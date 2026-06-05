from trading.core.portfolio import Portfolio
from trading.core.classes import MarketSignal, PriceData, SignalType, OrderAction, Order, TickResults, BracketOrder, OrderType, OrderStatus

from typing import Optional
from datetime import datetime
from utils.utils import Logger

logger = Logger().get_logger(__name__)

class DualSymbolSwitchPortfolio(Portfolio):
    """
    Switch portfolio between two leveraged ETFs (e.g. UPRO/SPXU) using
    bracket orders with stop-loss and profit-taker exits.

    Designed for use with SpyTrendMACDAlgorithm which emits BUY signals
    for the target symbol based on SPY MACD regime. Never holds both
    symbols simultaneously.

    Strategy flow:
        1. No position + BUY signal → enter target with bracket order
        2. Holding same symbol + BUY signal → ignore (no pyramiding)
        3. Holding different symbol + BUY signal → if holding period
           elapsed, trigger manual sale and queue switch for next tick
        4. Bracket stop/profit triggers → exit immediately regardless
           of holding period

    Live trading support:
        Uses self.get_price(symbol, tick) for price lookups, which falls
        back to the cached previous_price when the target symbol is not
        in the current tick. This is essential for live trading via
        AlpacaRealTimeEngine where each websocket bar event produces a
        tick with only one symbol.

    Config keys:
        upro_symbol (str): Bullish symbol (default: "UPRO")
        spxu_symbol (str): Bearish symbol (default: "SPXU")
        stop_pct (float): Stop-loss percentage (default: 5.0)
        profit_pct (float): Profit-taker percentage (default: 10.0)
        tx_cost (float): Transaction cost per order (default: 0.0)
        holding_period_hours (float): Minimum hours before switching (default: 2)
        min_signal_strength (int): Minimum signal strength to act on (default: 0)
    """

    def __init__(self, cfg: dict = None, order_manager=None, starting_cash: float = 0.0,
                 starting_positions=None, keep_history: bool = False):
        super().__init__(cfg, order_manager, starting_cash, starting_positions, keep_history)
        self.upro_symbol = self.cfg.get("upro_symbol", "UPRO")
        self.spxu_symbol = self.cfg.get("spxu_symbol", "SPXU")
        self.tx_cost = self.cfg.get("tx_cost", 0.0)
        self.min_signal_strength = self.cfg.get("min_signal_strength", 0)

        # Bracket order parameters
        self._refresh_bracket_config()

        # Holding period (in hours)
        self.holding_period_hours = self.cfg.get("holding_period_hours", 2)

        # State variables
        self._pending_target = None
        self._pending_switch_state = None  # None or "SWITCHING"
        self._symbol_entry_time = {}  # symbol -> datetime

    def _refresh_bracket_config(self) -> None:
        self.bracket_cfg = self.get_bracket_config({
            "stop_pct": self.cfg.get("stop_pct", 5.0),
            "profit_pct": self.cfg.get("profit_pct", 10.0),
        })
        self.stop_pct = float(self.bracket_cfg["stop_pct"])
        self.profit_pct = float(self.bracket_cfg["profit_pct"])

    def reconfigure(self, new_params: dict) -> None:
        super().reconfigure(new_params)
        self._refresh_bracket_config()

    def _current_symbol(self) -> str | None:
        for symbol in (self.upro_symbol, self.spxu_symbol):
            if symbol in self.positions and self.positions[symbol].quantity > 0:
                return symbol
        return None

    def _entry_fills_now(self, current_timestamp: datetime) -> bool:
        if not getattr(self.om, "_market_hours_only", False):
            return True
        market_time = current_timestamp.time()
        return datetime.strptime("09:30", "%H:%M").time() <= market_time < datetime.strptime("16:00", "%H:%M").time()

    def _holding_period_elapsed(self, symbol: str, current_timestamp: datetime) -> bool:
        """
        Check if holding period has elapsed since entering this position.

        Args:
            symbol: Symbol to check
            current_timestamp: Current tick timestamp

        Returns:
            True if holding period has elapsed or no entry time recorded
        """
        if symbol not in self._symbol_entry_time:
            return True  # No entry time means we can enter/exit freely

        from datetime import timedelta
        entry_time = self._symbol_entry_time[symbol]
        holding_delta = timedelta(hours=self.holding_period_hours)

        return current_timestamp >= entry_time + holding_delta

    def _find_active_bracket(self, symbol: str) -> Optional[BracketOrder]:
        """
        Find the active bracket order for a symbol.

        Args:
            symbol: Symbol to search for

        Returns:
            BracketOrder if found, None otherwise
        """
        for order_id in self.om.pending_orders_by_id:
            order = self.om.pending_orders_by_id[order_id]
            if (isinstance(order, BracketOrder) and
                order.symbol == symbol and
                order.status in (OrderStatus.PENDING_SALE, OrderStatus.PENDING)):
                return order

        return None

    def process_tick_market_signals_logic(
            self, signals: list[MarketSignal],
            tick: list[PriceData]) -> TickResults:
        """
        Process signals with bracket orders and holding period enforcement.

        Strategy:
        1. Bracket triggers (stop/profit) → Exit immediately at any time
        2. Signal changes → Exit via manual sale ONLY if holding period elapsed
        3. During holding period → Ignore new signals for different symbols
        4. Always use bracket orders for entries
        """
        orders: list[Order] = []

        # Get current timestamp from tick
        current_timestamp = tick[0].timestamp if tick else None
        if current_timestamp is None:
            return TickResults(orders=[])

        # Check current position
        current_symbol = self._current_symbol()

        # Record entry time on the first tick we observe the position is open.
        # This must happen here (not at order-submission time) so that orders
        # which pend overnight (market_hours_only) don't start the holding-period
        # clock before the position actually fills at market open.
        if current_symbol is not None and current_symbol not in self._symbol_entry_time:
            self._symbol_entry_time[current_symbol] = current_timestamp

        # Phase 1: Find target from current tick signals
        target_signal = None
        for signal in signals:
            if signal.type != SignalType.BUY:
                continue
            if signal.symbol not in (self.upro_symbol, self.spxu_symbol):
                continue
            if signal.strength < self.min_signal_strength:
                continue
            if target_signal is None or signal.strength > target_signal.strength:
                target_signal = signal

        # No signal → nothing to do
        if target_signal is None:
            return TickResults(orders=[])

        target = target_signal.symbol

        # Phase 3: Execute pending switch FIRST (takes precedence)
        if self._pending_switch_state == "SWITCHING" and self._pending_target is not None:
            # Check if previous position is closed
            if current_symbol is None:  # Position exited
                pending_target = self._pending_target

                # If a bracket for the target is already PENDING (stale pre-market
                # order), let it fill naturally instead of adding a second one.
                if self._find_active_bracket(pending_target) is not None:
                    self._pending_target = None
                    self._pending_switch_state = None
                    return TickResults(orders=orders)

                entry_price = self.get_price(pending_target, tick)

                if entry_price is not None:
                    qty = int(self.cash * 0.99 / entry_price)

                    if qty > 0:
                        bo = self.create_configured_bracket_order(
                            pending_target, entry_price, qty, self.tx_cost, tick, self.bracket_cfg
                        )
                        bo.intended_entry_price = entry_price
                        logger.debug(
                            f"Switching position to {pending_target} - qty: {qty}, "
                            f"entry price: {entry_price}, profit_pct: {self.profit_pct}, "
                            f"stop_pct: {self.stop_pct}")

                        # Link to signal if available
                        if target_signal and target_signal.symbol == pending_target:
                            target_signal.metadata['order_id'] = bo.order_id
                            target_signal.metadata['entry_time'] = current_timestamp
                        if self._entry_fills_now(current_timestamp):
                            self._symbol_entry_time[pending_target] = current_timestamp

                        orders.append(bo)

                # Reset state
                self._pending_target = None
                self._pending_switch_state = None

                # Return early - don't process other cases this tick
                return TickResults(orders=orders)

        # Phase 2: Handle position switching logic

        # Case A: No current position → Enter immediately
        if current_symbol is None:
            # Don't submit if ANY bracket is active for either symbol.
            # With market_hours_only, a bracket submitted pre-market stays PENDING
            # until 9:30 AM.  If the signal flips during that wait, submitting a
            # second bracket for the new symbol means both fill at open with the
            # same pf_cash → double-buy → negative cash → no more trades.
            for sym in (self.upro_symbol, self.spxu_symbol):
                if self._find_active_bracket(sym) is not None:
                    return TickResults(orders=[])

            entry_price = self.get_price(target, tick)
            if entry_price is None:
                return TickResults(orders=[])

            # Create bracket order
            qty = int(self.cash * 0.99 / entry_price)

            if qty > 0:
                bo = self.create_configured_bracket_order(
                    target, entry_price, qty, self.tx_cost, tick, self.bracket_cfg
                )
                bo.intended_entry_price = entry_price

                logger.debug(
                    f"No Current position, entering position for {target} - qty: {qty}, "
                    f"entry price: {entry_price}, profit_pct: {self.profit_pct}, "
                    f"stop_pct: {self.stop_pct}")

                # Link signal to order for analysis
                target_signal.metadata['order_id'] = bo.order_id
                target_signal.metadata['entry_time'] = current_timestamp
                if self._entry_fills_now(current_timestamp):
                    self._symbol_entry_time[target] = current_timestamp

                orders.append(bo)

        # Case B: Holding same symbol → Ignore (don't add to position)
        elif current_symbol == target:
            # Already holding target, no action needed
            logger.debug(f"Skipping position switch for {target} - already holding")
            pass

        # Case C: Different symbol → Switch if holding period elapsed
        elif current_symbol != target:
            # Check if holding period has elapsed
            if self._holding_period_elapsed(current_symbol, current_timestamp):
                # Find active bracket for current position
                active_bracket = self._find_active_bracket(current_symbol)

                if active_bracket is not None:
                    logger.debug(f"Holding period elapsed, triggering manual sale for {current_symbol}")

                    # Queue next target for next tick
                    self._pending_target = target
                    self._pending_switch_state = "SWITCHING"

                    # Clear entry time for old symbol
                    if current_symbol in self._symbol_entry_time:
                        del self._symbol_entry_time[current_symbol]

                    # Request manual sale — base class handles OM calls and portfolio updates
                    return TickResults(orders=orders, trigger_manual_sales=[active_bracket])
            else:
                # Holding period not elapsed → Ignore signal
                # Let bracket exit naturally via stop/profit
                logger.debug(f"Holding period in force for: {current_symbol} - Target = {target}")
                pass

        return TickResults(orders=orders)
