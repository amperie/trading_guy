"""
OrderManager implementation backed by the Alpaca Trading API.

Submits orders to Alpaca's paper or live trading endpoints and polls
for fill status. Supports MARKET and BRACKET order types. Also provides
broker account state queries for portfolio synchronization.

Bracket order lifecycle on Alpaca:
    1. Submit a market BUY with order_class=BRACKET, attaching
       take_profit (limit) and stop_loss (stop) legs.
    2. Alpaca fills the entry and creates two child leg orders.
    3. When one leg fills, Alpaca auto-cancels the other.
    4. This OM maps the Alpaca leg fills back to the local
       BracketOrder's SOLD_ORDER for portfolio processing.

Broker state synchronization:
    get_broker_account_state() queries Alpaca for the current account
    cash balance and all open positions. The Portfolio calls this
    periodically (controlled by sync_with_broker / sync_interval config)
    to correct local drift from asynchronous fills and to recover
    positions from previous sessions.

ID mapping:
    Local order IDs (uuid-based) are mapped to Alpaca's remote IDs
    via _local_to_remote. Bracket child legs are tracked separately
    in _bracket_child_map so exit fills can be matched back to the
    correct STOP or PROFIT child order.

Requirements:
    pip install alpaca-py
"""
from __future__ import annotations

from datetime import datetime
from typing import Union, Optional

from trading.core.classes import Order, OrderType, OrderStatus, OrderAction, OrderTimeInForce, Position, BracketOrder
from trading.core.om.order_manager import OrderManager
from utils.logger import Logger

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, TakeProfitRequest, StopLossRequest, GetOrdersRequest,
        ReplaceOrderRequest,
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    TradingClient = None
    MarketOrderRequest = None
    TakeProfitRequest = None
    StopLossRequest = None
    GetOrdersRequest = None
    ReplaceOrderRequest = None
    OrderSide = None
    TimeInForce = None
    OrderClass = None

logger = Logger().get_logger(__name__)


class AlpacaOrderManager(OrderManager):
    """
    OrderManager backed by the Alpaca Trading API.

    Submits MARKET and BRACKET orders to Alpaca and polls for fill
    status. Designed for both paper trading and live trading. Also
    provides broker account state queries via get_broker_account_state()
    so the Portfolio can periodically sync its local cash and positions
    with the actual Alpaca account.

    Supported order types:
        - MARKET: Immediate market orders (BUY or SELL).
        - BRACKET: Market BUY entry with attached stop-loss and
          profit-taker exit legs. Alpaca manages the exit leg
          lifecycle (one fills, other auto-cancels).

    Broker synchronization:
        get_broker_account_state() returns the Alpaca account's current
        cash, buying power, and all open positions. This is used by
        Portfolio._sync_from_broker() to:
        - Correct cash drift from asynchronous fills
        - Recover positions from previous trading sessions
        - Remove stale local positions that were closed externally

    Configuration (cfg dict):
        api_key (str): Alpaca API key. Required.
        secret_key (str): Alpaca secret key. Required.
        paper (bool): Use paper trading endpoint. Default: True.
        time_in_force (str): Order duration — "day" or "gtc".
            Default: "day".

    Attributes:
        client (TradingClient): The alpaca-py trading client instance.
        time_in_force (str): Order duration setting.
        _local_to_remote (dict[str, str]): Maps local order_id to
            Alpaca's remote order ID (platform_id).
        _bracket_child_map (dict[str, dict[str, str]]): Maps parent
            bracket order_id to {"STOP": remote_id, "PROFIT": remote_id}
            for leg tracking.

    Example:
        cfg = {
            "api_key": "PK...",
            "secret_key": "...",
            "paper": True,
            "time_in_force": "day",
        }
        om = AlpacaOrderManager(cfg)

        # Submit a market order
        order = Order.create_market_order("AAPL", OrderAction.BUY, 10, 0.0, tick)
        om.submit_order(order, tick, positions, cash)

        # Submit a bracket order
        bo = BracketOrder.create_bracket_order(
            "AAPL", high_sell_price=160.0, low_sell_price=140.0,
            quantity=10, tx_cost=0.0, current_tick=tick)
        om.submit_order(bo, tick, positions, cash)

        # Query broker state for portfolio sync
        state = om.get_broker_account_state()
        # Returns: {"cash": 95000.0, "buying_power": 190000.0,
        #           "positions": {"AAPL": 50, "SPY": 100}}
    """

    def __init__(self, cfg: dict = None):
        """
        Initialize the Alpaca order manager and connect to the API.

        Creates a TradingClient with the provided credentials. Defaults
        to paper trading if not specified.

        Args:
            cfg: Configuration dict. Required keys:
                - api_key (str): Alpaca API key.
                - secret_key (str): Alpaca secret key.
                Optional keys:
                - paper (bool): Use paper trading. Default: True.
                - time_in_force (str): "day" or "gtc". Default: "day".

        Raises:
            ImportError: If alpaca-py is not installed.
            ValueError: If api_key or secret_key is missing.
        """
        super().__init__(cfg)
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py is required. Install with: pip install alpaca-py")

        cfg = cfg or {}
        api_key = cfg.get("api_key")
        secret_key = cfg.get("secret_key")
        paper = cfg.get("paper", True)
        if not api_key or not secret_key:
            raise ValueError("AlpacaOrderManager requires api_key and secret_key in config")

        self.time_in_force = cfg.get("time_in_force", "day").lower()
        self.client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)
        self._local_to_remote: dict[str, str] = {}
        self._bracket_child_map: dict[str, dict[str, str]] = {}
        self.trailing_stop_replace_min_increment = float(cfg.get("trailing_stop_replace_min_increment", 0.01))

    def get_broker_account_state(self) -> dict | None:
        """Query Alpaca for current account cash and open positions.

        Returns:
            Dict with "cash", "buying_power", and "positions" (symbol -> qty).
            Returns None if the API call fails.
        """
        try:
            account = self.client.get_account()
            positions_raw = self.client.get_all_positions()
            return {
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "positions": {p.symbol: int(float(p.qty)) for p in positions_raw},
            }
        except Exception as exc:
            logger.error(f"Failed to fetch Alpaca account state: {exc}")
            return None

    def sync_orders_from_broker(self, limit: int = 100) -> dict:
        """Pull all orders from Alpaca and sync into local order dicts.

        Fetches orders with nested=True so bracket legs are included.
        New orders are converted to local Order/BracketOrder objects.
        Already-tracked orders have their status refreshed.

        Returns:
            Dict with counts: {"new": N, "updated": N, "total": N}
        """
        try:
            request = GetOrdersRequest(status="all", nested=True, limit=limit)
            alpaca_orders = self.client.get_orders(filter=request)
        except Exception as exc:
            logger.error(f"Failed to fetch orders from Alpaca: {exc}")
            return {"new": 0, "updated": 0, "total": 0, "error": str(exc)}

        # Build reverse lookup: remote platform_id -> local order_id.
        # Prefer the explicit map, but also fall back to any already-loaded
        # local orders so resumed sessions do not recreate duplicates.
        remote_to_local = {
            str(v): k for k, v in self._local_to_remote.items() if v
        }
        for local_id, local_order in self._all_orders.items():
            platform_id_existing = getattr(local_order, "platform_id", None)
            if platform_id_existing:
                remote_to_local.setdefault(str(platform_id_existing), local_id)

        new_count = 0
        updated_count = 0

        for alpaca_order in alpaca_orders:
            platform_id = str(alpaca_order.id)

            if platform_id in remote_to_local:
                # Update path: refresh status on existing local order
                local_id = remote_to_local[platform_id]
                local_order = self._all_orders.get(local_id)
                if local_order is not None:
                    self._update_order_status_from_backend(local_order)
                    self._update_order_lists(local_order)
                    updated_count += 1
                continue

            # Create path: build new local Order/BracketOrder
            alpaca_status = _enum_val(getattr(alpaca_order, "status", ""))
            alpaca_side = _enum_val(getattr(alpaca_order, "side", ""))
            action = OrderAction.BUY if "buy" in alpaca_side else OrderAction.SELL
            quantity = int(float(alpaca_order.qty or 0))
            filled_price = float(alpaca_order.filled_avg_price or 0)
            submitted_at = _parse_alpaca_time(getattr(alpaca_order, "submitted_at", None))
            filled_at = _parse_alpaca_time(getattr(alpaca_order, "filled_at", None))
            alpaca_tif = _REVERSE_TIF_MAP.get(
                _enum_val(getattr(alpaca_order, "time_in_force", "")), OrderTimeInForce.GTC)

            order_class = _enum_val(getattr(alpaca_order, "order_class", ""))
            is_bracket = order_class in ("bracket", "oco")
            legs = getattr(alpaca_order, "legs", None) or []

            if is_bracket and action == OrderAction.BUY:
                # Create BracketOrder with children from legs
                main_order = BracketOrder(
                    action=action,
                    type=OrderType.BRACKET,
                    symbol=alpaca_order.symbol,
                    price=filled_price,
                    quantity=quantity,
                    cash=filled_price * quantity if filled_price else 0,
                    placed_datetime=submitted_at or datetime.now(),
                    executed_datetime=filled_at,
                    time_in_force=alpaca_tif,
                    platform_id=platform_id,
                )

                # Build child orders from Alpaca legs
                for leg in legs:
                    leg_order_type = _enum_val(getattr(leg, "order_type", ""))
                    leg_status = _enum_val(getattr(leg, "status", ""))
                    leg_platform_id = str(getattr(leg, "id", ""))
                    leg_filled_price = float(getattr(leg, "filled_avg_price", 0) or 0)
                    leg_filled_qty = int(float(getattr(leg, "filled_qty", 0) or 0))
                    leg_filled_at = _parse_alpaca_time(getattr(leg, "filled_at", None))

                    if "stop" in leg_order_type:
                        child_name = "STOP"
                        child_type = OrderType.STOP_LOSS
                        trigger_price = float(getattr(leg, "stop_price", 0) or 0)
                    elif "limit" in leg_order_type:
                        child_name = "PROFIT"
                        child_type = OrderType.PROFIT_TAKER
                        trigger_price = float(getattr(leg, "limit_price", 0) or 0)
                    else:
                        continue

                    child_price = leg_filled_price if leg_filled_price else trigger_price
                    child_order = Order(
                        action=OrderAction.SELL,
                        type=child_type,
                        symbol=alpaca_order.symbol,
                        price=child_price,
                        quantity=leg_filled_qty if leg_filled_qty else quantity,
                        cash=leg_filled_price * leg_filled_qty if leg_filled_price and leg_filled_qty else 0,
                        placed_datetime=submitted_at or datetime.now(),
                        executed_datetime=leg_filled_at,
                        time_in_force=alpaca_tif,
                        platform_id=leg_platform_id,
                        parent_id=main_order.order_id,
                        status=OrderStatus.FILLED if leg_status == "filled" else (
                            OrderStatus.CANCELED if leg_status in ("canceled", "expired", "rejected") else OrderStatus.PENDING
                        ),
                    )
                    main_order.add_child_order(child_name, child_order)

                    # Register leg in bracket child map
                    if main_order.order_id not in self._bracket_child_map:
                        self._bracket_child_map[main_order.order_id] = {}
                    self._bracket_child_map[main_order.order_id][child_name] = leg_platform_id
                    self._local_to_remote[child_order.order_id] = leg_platform_id

                    # If this leg filled, it's the exit order
                    if leg_status == "filled":
                        main_order.SOLD_ORDER = child_order
                        main_order.status = OrderStatus.FILLED

                # Add MANUAL_ORDER placeholder if not already present
                if "MANUAL_ORDER" not in main_order.get_child_order_names():
                    main_order.add_child_order("MANUAL_ORDER", None)

                # Determine bracket status if not already set by a filled leg
                if main_order.status != OrderStatus.FILLED:
                    # If all real child orders are canceled the bracket is done
                    children = [main_order.get_child_order(n)
                                for n in main_order.get_child_order_names()]
                    real_children = [c for c in children if c is not None]
                    if real_children and all(
                            c.status == OrderStatus.CANCELED for c in real_children):
                        main_order.status = OrderStatus.FILLED
                    elif alpaca_status == "filled":
                        # Entry filled but no exit leg filled yet
                        main_order.status = OrderStatus.PENDING_SALE
                    elif alpaca_status in ("canceled", "expired", "rejected"):
                        main_order.status = OrderStatus.CANCELED
                    else:
                        main_order.status = OrderStatus.PENDING

                order = main_order
            else:
                # Simple MARKET order
                if alpaca_status == "filled":
                    status = OrderStatus.FILLED
                elif alpaca_status in ("canceled", "expired", "rejected"):
                    status = OrderStatus.CANCELED
                else:
                    status = OrderStatus.PENDING

                order = Order(
                    action=action,
                    type=OrderType.MARKET,
                    symbol=alpaca_order.symbol,
                    price=filled_price,
                    quantity=quantity,
                    cash=filled_price * quantity if filled_price else 0,
                    placed_datetime=submitted_at or datetime.now(),
                    executed_datetime=filled_at,
                    time_in_force=alpaca_tif,
                    platform_id=platform_id,
                    status=status,
                )

            # Register mapping
            self._local_to_remote[order.order_id] = platform_id

            # Place into correct dicts
            self._all_orders[order.order_id] = order
            if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED):
                self._filled_orders_by_id[order.order_id] = order
                # For filled brackets, also register the SOLD_ORDER
                if order.type == OrderType.BRACKET and order.status == OrderStatus.FILLED and order.SOLD_ORDER:
                    self._filled_orders_by_id[order.SOLD_ORDER.order_id] = order.SOLD_ORDER
                    self._all_orders[order.SOLD_ORDER.order_id] = order.SOLD_ORDER
            else:
                self._pending_orders_by_id[order.order_id] = order

            new_count += 1
            logger.info(f"Synced order from Alpaca: {order.order_id[:12]}... {order.symbol} {order.type.name} {order.status.name}")

        total = len(alpaca_orders)
        logger.info(f"Order sync complete - New: {new_count}, Updated: {updated_count}, Total from Alpaca: {total}")
        return {"new": new_count, "updated": updated_count, "total": total}

    def _cancel_order(self, order_id: str) -> Order:
        """
        Cancel a pending order via the Alpaca API.

        Looks up the remote ID and sends a cancel request to Alpaca.
        Logs a warning if the cancel fails (e.g. order already filled)
        but does not raise — the order status will be corrected on the
        next poll.

        Args:
            order_id: Local order ID to cancel.

        Returns:
            The Order object (status may not yet reflect cancellation
            until the next backend poll).
        """
        remote_id = self._local_to_remote.get(order_id)
        if remote_id:
            try:
                self.client.cancel_order_by_id(remote_id)
            except Exception as exc:
                logger.warning(f"Failed to cancel Alpaca order {remote_id}: {exc}")
        return self._all_orders[order_id]

    def _clamp_buy_to_buying_power(self, order: Order, current_tick: list = None, pf_cash: float = 0.0) -> Order:
        """Reduce BUY order quantity if it exceeds Alpaca buying power.

        Queries the Alpaca account for current buying power and compares
        against estimated order cost. If cost exceeds buying power, reduces
        quantity to what the account can support. Updates order.quantity,
        order.cash, and bracket child order quantities to stay in sync.

        Args:
            order: A BUY order (MARKET or BRACKET). Modified in-place.
            current_tick: Current tick data used to estimate price.
            pf_cash: Portfolio cash used as fallback to derive implied price
                when the target symbol is absent from current_tick.

        Returns:
            The (possibly adjusted) order.
        """
        price = self._get_price_from_tick(order.symbol, current_tick)
        if (price is None or price <= 0) and pf_cash > 0 and order.quantity > 0:
            price = pf_cash / order.quantity
            logger.debug(
                f"No tick price for {order.symbol} — using implied price ${price:.4f} from pf_cash"
            )
        if price is None or price <= 0:
            logger.debug(
                f"No tick price for {order.symbol} — skipping buying power clamp"
            )
            return order

        try:
            account = self.client.get_account()
            # Use min(cash, buying_power) — on margin accounts buying_power is 2× cash,
            # but bracket/GTC orders are collateralized against actual cash balance.
            buying_power = min(float(account.cash), float(account.buying_power))
        except Exception as exc:
            logger.warning(f"Could not fetch buying power for pre-submit check: {exc}")
            return order

        cost = order.quantity * price
        if cost <= buying_power:
            return order

        new_qty = int(buying_power / price)
        if new_qty <= 0:
            logger.warning(
                f"Buying power ${buying_power:,.2f} insufficient for even 1 share "
                f"of {order.symbol} @ ${price:.2f} — order will likely be rejected"
            )
            return order

        logger.info(
            f"Clamping {order.symbol} BUY qty {order.quantity} -> {new_qty} "
            f"(buying_power=${buying_power:,.2f}, est_price=${price:.2f})"
        )
        order.quantity = new_qty
        order.cash = new_qty * price

        if order.type == OrderType.BRACKET:
            for name in order.get_child_order_names():
                child = order.get_child_order(name)
                if child is not None:
                    child.quantity = new_qty

        return order

    @staticmethod
    def _get_price_from_tick(symbol: str, current_tick: list) -> Optional[float]:
        """Extract closing price for *symbol* from the current tick list."""
        if not current_tick:
            return None
        pd = next((x for x in current_tick if x.symbol == symbol), None)
        return pd.close if pd else None

    def _submit_order_to_backend(
            self, order: Order, current_tick: list = None,
            positions: dict[str, Position] = None, pf_cash: float = 0.0) -> Order:
        """
        Submit an order to the Alpaca Trading API.

        Builds the appropriate Alpaca request based on order type:
        - MARKET: Simple market order (BUY or SELL).
        - BRACKET: Market BUY with attached take_profit (limit) and
          stop_loss (stop) legs. Only BUY entries are supported for
          bracket orders.

        After submission, maps the local order_id to Alpaca's remote ID
        and (for brackets) maps child leg IDs for exit tracking.

        The order is always set to PENDING after submission — Alpaca
        fills are detected asynchronously via _update_order_status_from_backend.

        Args:
            order: The Order or BracketOrder to submit.
            current_tick: Current tick data used for buying-power guard.
            positions: Current portfolio positions (unused).
            pf_cash: Current portfolio cash (unused).

        Returns:
            The submitted Order with platform_id set and status=PENDING.

        Raises:
            NotImplementedError: If order type is not MARKET or BRACKET,
                or if a BRACKET order has action != BUY.
            ValueError: If a BRACKET order is missing STOP or PROFIT
                child orders.
        """
        if order.action == OrderAction.BUY:
            self._clamp_buy_to_buying_power(order, current_tick, pf_cash)

        side = OrderSide.BUY if order.action == OrderAction.BUY else OrderSide.SELL
        tif = _ALPACA_TIF_MAP.get(order.time_in_force, TimeInForce.GTC)

        if order.type == OrderType.MARKET:
            req = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=tif,
            )
        elif order.type == OrderType.BRACKET:
            if order.action != OrderAction.BUY:
                raise NotImplementedError("Alpaca bracket orders are supported for BUY entries only")
            try:
                stop_leg = order.get_child_order("STOP")
                profit_leg = order.get_child_order("PROFIT")
            except KeyError as e:
                raise ValueError(f"Bracket order requires STOP and PROFIT child orders (missing: {e})")
            if stop_leg is None or profit_leg is None:
                raise ValueError("Bracket order requires STOP and PROFIT child orders")
            req = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=tif,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=profit_leg.price),
                stop_loss=StopLossRequest(stop_price=stop_leg.price),
            )
        else:
            raise NotImplementedError("AlpacaOrderManager supports MARKET and BRACKET orders only")

        logger.info(f"Local order to Alpaca: {order}")
        logger.info(f"Submitting order request to Alpaca: {req}")
        try:
            alpaca_order = self.client.submit_order(order_data=req)
        except Exception:
            logger.exception(f"Alpaca submit_order failed for local order {order.order_id}")
            raise
        logger.info(
            f"Alpaca accepted order: local={order.order_id} remote={alpaca_order.id} "
            f"status={_enum_val(getattr(alpaca_order, 'status', ''))}"
        )
        order.platform_id = str(alpaca_order.id)
        self._local_to_remote[order.order_id] = order.platform_id
        if order.type == OrderType.BRACKET and getattr(alpaca_order, "legs", None):
            stop_id = _find_leg_id(alpaca_order.legs, "stop")
            profit_id = _find_leg_id(alpaca_order.legs, "profit")
            self._bracket_child_map[order.order_id] = {
                "STOP": stop_id,
                "PROFIT": profit_id,
            }
            stop_order = order.get_child_order("STOP")
            profit_order = order.get_child_order("PROFIT")
            if stop_order is not None and stop_id:
                stop_order.platform_id = str(stop_id)
                self._local_to_remote[stop_order.order_id] = str(stop_id)
            if profit_order is not None and profit_id:
                profit_order.platform_id = str(profit_id)
                self._local_to_remote[profit_order.order_id] = str(profit_id)
        order.status = OrderStatus.PENDING
        return order

    def _refresh_bracket_child_ids(self, order: BracketOrder, legs) -> None:
        if not legs:
            return
        stop_id = _find_leg_id(legs, "stop")
        profit_id = _find_leg_id(legs, "profit")
        self._bracket_child_map.setdefault(order.order_id, {})
        if stop_id:
            self._bracket_child_map[order.order_id]["STOP"] = str(stop_id)
            stop_order = order.get_child_order("STOP")
            if stop_order is not None:
                stop_order.platform_id = str(stop_id)
                self._local_to_remote[stop_order.order_id] = str(stop_id)
        if profit_id:
            self._bracket_child_map[order.order_id]["PROFIT"] = str(profit_id)
            profit_order = order.get_child_order("PROFIT")
            if profit_order is not None:
                profit_order.platform_id = str(profit_id)
                self._local_to_remote[profit_order.order_id] = str(profit_id)

    def _maybe_replace_trailing_stop_leg(self, order: BracketOrder, current_tick: list = None) -> None:
        stop_order = order.get_child_order("STOP")
        if stop_order is None or stop_order.type != OrderType.TRAILING_STOP:
            return
        current_price = self._get_price_from_tick(order.symbol, current_tick) or order.price
        if current_price is None or current_price <= 0:
            return
        old_stop = stop_order.price
        old_hwm = stop_order.trail_hwm
        moved = stop_order.update_trailing_stop(current_price)
        if not moved or stop_order.price - old_stop < self.trailing_stop_replace_min_increment:
            return
        remote_stop_id = stop_order.platform_id or self._bracket_child_map.get(order.order_id, {}).get("STOP")
        if not remote_stop_id:
            logger.warning(f"Cannot replace trailing stop for {order.symbol}: missing Alpaca stop leg id")
            return
        try:
            self.client.replace_order_by_id(
                remote_stop_id,
                order_data=ReplaceOrderRequest(stop_price=stop_order.price),
            )
            logger.info(
                f"Raised trailing stop for {order.symbol}: {old_stop:.2f} -> {stop_order.price:.2f} "
                f"(hwm={stop_order.trail_hwm:.2f})"
            )
        except Exception as exc:
            stop_order.price = old_stop
            stop_order.trail_hwm = old_hwm
            logger.warning(f"Failed to replace Alpaca trailing stop leg {remote_stop_id}: {exc}")

    def _update_order_status_from_backend(
            self, order: Union[BracketOrder, Order], current_tick: list = None,
            positions: dict[str, Position] = None, pf_cash: float = 0.0) -> Order:
        """
        Poll Alpaca for the current status of a single order.

        Fetches the order by its remote ID and maps the Alpaca status
        back to the local OrderStatus enum.

        For BRACKET orders, handles the two-phase lifecycle:
            1. Entry fill (Alpaca "filled" + local PENDING):
               Sets status to PENDING_SALE, populates fill price/qty.
            2. Exit fill (a leg is filled):
               Creates/updates the SOLD_ORDER child, sets status to FILLED.
            3. Cancel/reject/expire: Sets status to CANCELED.

        For MARKET orders:
            - "filled" -> FILLED with price/qty populated.
            - "canceled"/"rejected"/"expired" -> CANCELED.
            - Everything else (new, accepted, partially_filled) -> PENDING.

        Args:
            order: The order to refresh. Modified in-place.
            current_tick: Current tick data (unused by Alpaca backend).
            positions: Current portfolio positions (unused).
            pf_cash: Current portfolio cash (unused).

        Returns:
            The updated Order object (same instance).
        """
        remote_id = order.platform_id or self._local_to_remote.get(order.order_id)
        if not remote_id:
            return order

        try:
            alpaca_order = self.client.get_order_by_id(remote_id)
        except Exception as exc:
            logger.warning(f"Failed to fetch Alpaca order {remote_id}: {exc}")
            return order

        status = _enum_val(alpaca_order.status)
        alpaca_tif = _REVERSE_TIF_MAP.get(_enum_val(getattr(alpaca_order, "time_in_force", "")), None)
        if alpaca_tif is not None:
            order.time_in_force = alpaca_tif

        if order.type == OrderType.BRACKET:
            # --- MANUAL_SALE handling (symbol switch) ---
            if order.status == OrderStatus.PENDING_SALE and order.MANUAL_SALE:
                manual_order = order.get_child_order("MANUAL_ORDER")

                if manual_order is None:
                    # Phase 1: Cancel bracket legs and submit standalone market SELL
                    try:
                        self.client.cancel_order_by_id(remote_id)
                    except Exception as exc:
                        logger.warning(f"Failed to cancel Alpaca bracket for manual sale: {exc}")

                    tif = _ALPACA_TIF_MAP.get(order.time_in_force, TimeInForce.GTC)
                    sell_req = MarketOrderRequest(
                        symbol=order.symbol,
                        qty=order.quantity,
                        side=OrderSide.SELL,
                        time_in_force=tif,
                    )
                    try:
                        alpaca_sell = self.client.submit_order(order_data=sell_req)
                        mo = Order.create_market_order(
                            order.symbol, OrderAction.SELL, order.quantity, order.tx_cost,
                            current_tick
                        )
                        mo.platform_id = str(alpaca_sell.id)
                        mo.status = OrderStatus.PENDING
                        self._local_to_remote[mo.order_id] = mo.platform_id
                        self._all_orders[mo.order_id] = mo
                        self._pending_orders_by_id[mo.order_id] = mo

                        order.add_child_order("MANUAL_ORDER", mo)
                        order.SOLD_ORDER = mo

                        so = order.get_child_order("STOP")
                        po = order.get_child_order("PROFIT")
                        if so:
                            so.status = OrderStatus.CANCELED
                        if po:
                            po.status = OrderStatus.CANCELED

                        logger.info(f"Manual sale submitted to Alpaca for {order.symbol} qty={order.quantity}, awaiting fill")
                    except Exception as exc:
                        logger.error(f"Failed to submit manual sale to Alpaca: {exc}")

                    # Bracket stays PENDING_SALE until the manual sell fills
                    return order

                else:
                    # Phase 2: Poll Alpaca for manual sell fill
                    mo_remote_id = manual_order.platform_id or self._local_to_remote.get(manual_order.order_id)
                    if mo_remote_id:
                        try:
                            alpaca_mo = self.client.get_order_by_id(mo_remote_id)
                            mo_status = _enum_val(alpaca_mo.status)

                            if mo_status == "filled":
                                manual_order.status = OrderStatus.FILLED
                                manual_order.price = float(alpaca_mo.filled_avg_price or 0)
                                filled_qty = int(float(alpaca_mo.filled_qty or 0))
                                if filled_qty > 0:
                                    manual_order.quantity = filled_qty
                                    manual_order.cash = manual_order.price * manual_order.quantity
                                manual_order.executed_datetime = _parse_alpaca_time(alpaca_mo.filled_at) or datetime.now()

                                order.status = OrderStatus.FILLED
                                logger.info(
                                    f"Manual sale filled for {order.symbol} qty={manual_order.quantity} "
                                    f"@ ${manual_order.price:.2f}"
                                )
                            elif mo_status in {"canceled", "rejected", "expired"}:
                                manual_order.status = OrderStatus.CANCELED
                                order.status = OrderStatus.CANCELED
                                logger.warning(f"Manual sale {mo_status} for {order.symbol}")
                        except Exception as exc:
                            logger.warning(f"Failed to poll manual sale order: {exc}")

                    return order

            # Entry fill -> move to PENDING_SALE to update portfolio buy.
            if status == "filled" and order.status == OrderStatus.PENDING:
                order.status = OrderStatus.PENDING_SALE
                order.price = float(alpaca_order.filled_avg_price or 0)
                filled_qty = int(float(alpaca_order.filled_qty or 0))
                if filled_qty > 0:
                    order.quantity = filled_qty
                    order.cash = order.price * order.quantity
                    for name in order.get_child_order_names():
                        child = order.get_child_order(name)
                        if child is not None:
                            child.quantity = filled_qty
                order.executed_datetime = _parse_alpaca_time(alpaca_order.filled_at) or datetime.now()
                stop_order = order.get_child_order("STOP")
                if stop_order is not None and stop_order.type == OrderType.TRAILING_STOP and order.price > 0:
                    stop_order.trail_hwm = order.price

            # Check legs for exit fill.
            legs = getattr(alpaca_order, "legs", None) or []
            self._refresh_bracket_child_ids(order, legs)
            filled_leg = _find_filled_leg(legs)
            if filled_leg is not None:
                exit_order = _update_exit_order_from_leg(order, filled_leg)
                order.SOLD_ORDER = exit_order
                order.status = OrderStatus.FILLED
            elif order.status == OrderStatus.PENDING_SALE:
                self._maybe_replace_trailing_stop_leg(order, current_tick)
            elif status in {"canceled", "rejected", "expired"}:
                order.status = OrderStatus.CANCELED
        else:
            if status == "filled":
                order.status = OrderStatus.FILLED
                order.price = float(alpaca_order.filled_avg_price or 0)
                filled_qty = int(float(alpaca_order.filled_qty or 0))
                if filled_qty > 0:
                    order.quantity = filled_qty
                    order.cash = order.price * order.quantity
                order.executed_datetime = _parse_alpaca_time(alpaca_order.filled_at) or datetime.now()
            elif status in {"canceled", "rejected", "expired"}:
                order.status = OrderStatus.CANCELED
            else:
                # new, accepted, partially_filled, etc.
                order.status = OrderStatus.PENDING

        return order

    def _update_orders_statuses_from_backend(
            self, orders: dict[str, Order], current_tick: list = None,
            positions: dict[str, Position] = None, pf_cash: float = 0.0) -> list[str]:
        """
        Poll Alpaca for status changes on all pending orders.

        Iterates through each pending order and calls
        _update_order_status_from_backend(). Tracks which orders
        changed status.

        Args:
            orders: Dict of pending orders keyed by order_id.
            current_tick: Current tick data (unused).
            positions: Current portfolio positions (unused).
            pf_cash: Current portfolio cash (unused).

        Returns:
            List of order_id strings for orders whose status changed.
        """
        changed = []
        for order in list(orders.values()):
            original_status = order.status
            self._update_order_status_from_backend(order, current_tick, positions, pf_cash)
            if order.status != original_status:
                changed.append(order.order_id)
        return changed


def _enum_val(attr) -> str:
    """Get the lowercase string value from an Alpaca enum or string.

    Alpaca-py enums stringify as 'ClassName.VALUE' via str(), but their
    .value attribute holds the bare string (e.g. 'bracket', 'filled').
    """
    if hasattr(attr, "value"):
        return str(attr.value).lower()
    return str(attr).lower()


_ALPACA_TIF_MAP = {
    OrderTimeInForce.DAY: TimeInForce.DAY if TimeInForce else None,
    OrderTimeInForce.GTC: TimeInForce.GTC if TimeInForce else None,
    OrderTimeInForce.IOC: TimeInForce.IOC if TimeInForce else None,
    OrderTimeInForce.FOK: TimeInForce.FOK if TimeInForce else None,
    OrderTimeInForce.OPG: TimeInForce.OPG if TimeInForce else None,
    OrderTimeInForce.CLS: TimeInForce.CLS if TimeInForce else None,
} if TimeInForce else {}

_REVERSE_TIF_MAP = {
    "day": OrderTimeInForce.DAY,
    "gtc": OrderTimeInForce.GTC,
    "ioc": OrderTimeInForce.IOC,
    "fok": OrderTimeInForce.FOK,
    "opg": OrderTimeInForce.OPG,
    "cls": OrderTimeInForce.CLS,
}


def _parse_alpaca_time(value: Optional[datetime]) -> Optional[datetime]:
    """
    Parse an Alpaca timestamp into a Python datetime.

    Handles datetime objects directly, or ISO format strings
    (replacing "Z" suffix with "+00:00" for fromisoformat compatibility).

    Args:
        value: An Alpaca timestamp (datetime, string, or None).

    Returns:
        A datetime object, or None if parsing fails or value is None.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _find_leg_id(legs, leg_type: str) -> Optional[str]:
    """
    Find the Alpaca remote ID for a bracket order leg.

    Searches the Alpaca order's legs list for a matching leg type
    by inspecting each leg's order_type attribute.

    Args:
        legs: List of Alpaca leg objects from the parent order.
        leg_type: Either "stop" (matches order_type containing "stop")
            or "profit" (matches order_type containing "limit").

    Returns:
        The leg's remote ID string, or None if not found.
    """
    for leg in legs or []:
        order_type = _enum_val(getattr(leg, "order_type", ""))
        if leg_type == "stop" and "stop" in order_type:
            return getattr(leg, "id", None)
        if leg_type == "profit" and "limit" in order_type:
            return getattr(leg, "id", None)
    return None


def _find_filled_leg(legs):
    """
    Find the first filled leg in a bracket order's legs list.

    Args:
        legs: List of Alpaca leg objects from the parent order.

    Returns:
        The first leg object with status "filled", or None if no
        leg has been filled yet.
    """
    for leg in legs or []:
        status = _enum_val(getattr(leg, "status", ""))
        if status == "filled":
            return leg
    return None


def _update_exit_order_from_leg(order: BracketOrder, leg) -> Order:
    """
    Create or update the exit Order from an Alpaca filled leg.

    Maps the Alpaca leg's fill details (price, quantity, timestamp)
    back to the local BracketOrder's child order (STOP or PROFIT).
    If the child order doesn't exist locally (edge case), creates
    a new Order with the appropriate type.

    Args:
        order: The parent BracketOrder whose exit leg has filled.
        leg: The Alpaca leg object with fill details.

    Returns:
        The exit Order populated with fill price, quantity, cash,
        platform_id, and executed_datetime. Status is set to FILLED.
    """
    order_type = _enum_val(getattr(leg, "order_type", ""))
    if "stop" in order_type:
        exit_order = order.get_child_order("STOP")
        other_order = order.get_child_order("PROFIT")
    else:
        exit_order = order.get_child_order("PROFIT")
        other_order = order.get_child_order("STOP")

    if other_order is not None:
        other_order.status = OrderStatus.CANCELED

    if exit_order is None:
        exit_order = Order(
            action=OrderAction.SELL,
            type=OrderType.STOP_LOSS if "stop" in order_type else OrderType.PROFIT_TAKER,
            symbol=order.symbol,
            price=0.0,
            quantity=0,
            cash=0.0,
            placed_datetime=order.placed_datetime,
            parent_id=order.order_id,
        )

    exit_order.platform_id = str(getattr(leg, "id", "")) or exit_order.platform_id
    exit_order.status = OrderStatus.FILLED
    exit_order.price = float(getattr(leg, "filled_avg_price", 0) or 0)
    filled_qty = int(float(getattr(leg, "filled_qty", 0) or 0))
    if filled_qty > 0:
        exit_order.quantity = filled_qty
        exit_order.cash = exit_order.price * exit_order.quantity
    exit_order.executed_datetime = _parse_alpaca_time(getattr(leg, "filled_at", None)) or datetime.now()
    return exit_order
