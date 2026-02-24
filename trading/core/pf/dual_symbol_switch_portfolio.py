from trading.core.portfolio import Portfolio
from trading.core.classes import MarketSignal, PriceData, SignalType, OrderAction, Order, TickResults, BracketOrder, OrderType, OrderStatus

from typing import Optional
from datetime import datetime


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
        self.stop_pct = self.cfg.get("stop_pct", 5.0)
        self.profit_pct = self.cfg.get("profit_pct", 10.0)

        # Holding period (in hours)
        self.holding_period_hours = self.cfg.get("holding_period_hours", 2)

        # State variables
        self._pending_target = None
        self._pending_switch_state = None  # None or "SWITCHING"
        self._symbol_entry_time = {}  # symbol -> datetime

    def _current_symbol(self) -> str | None:
        for symbol in (self.upro_symbol, self.spxu_symbol):
            if symbol in self.positions and self.positions[symbol].quantity > 0:
                return symbol
        return None

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
                order.status == OrderStatus.PENDING_SALE):
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
                entry_price = self.get_price(pending_target, tick)

                if entry_price is not None:
                    qty = int(self.cash / entry_price)

                    if qty > 0:
                        bo = BracketOrder.create_bracket_order(
                            symbol=pending_target,
                            high_sell_price=round(entry_price * (1.0 + self.profit_pct / 100.0), 2),
                            low_sell_price=round(entry_price * (1.0 - self.stop_pct / 100.0), 2),
                            quantity=qty,
                            tx_cost=self.tx_cost,
                            current_tick=tick
                        )

                        # Track entry time
                        self._symbol_entry_time[pending_target] = current_timestamp

                        # Link to signal if available
                        if target_signal and target_signal.symbol == pending_target:
                            target_signal.metadata['order_id'] = bo.order_id
                            target_signal.metadata['entry_time'] = current_timestamp

                        orders.append(bo)

                # Reset state
                self._pending_target = None
                self._pending_switch_state = None

                # Return early - don't process other cases this tick
                return TickResults(orders=orders)

        # Phase 2: Handle position switching logic

        # Case A: No current position → Enter immediately
        if current_symbol is None:
            entry_price = self.get_price(target, tick)
            if entry_price is None:
                return TickResults(orders=[])

            # Create bracket order
            qty = int(self.cash / entry_price)

            if qty > 0:
                bo = BracketOrder.create_bracket_order(
                    symbol=target,
                    high_sell_price=round(entry_price * (1.0 + self.profit_pct / 100.0), 2),
                    low_sell_price=round(entry_price * (1.0 - self.stop_pct / 100.0), 2),
                    quantity=qty,
                    tx_cost=self.tx_cost,
                    current_tick=tick
                )

                # Track entry time for holding period
                self._symbol_entry_time[target] = current_timestamp

                # Link signal to order for analysis
                target_signal.metadata['order_id'] = bo.order_id
                target_signal.metadata['entry_time'] = current_timestamp

                orders.append(bo)

        # Case B: Holding same symbol → Ignore (don't add to position)
        elif current_symbol == target:
            # Already holding target, no action needed
            pass

        # Case C: Different symbol → Switch if holding period elapsed
        elif current_symbol != target:
            # Check if holding period has elapsed
            if self._holding_period_elapsed(current_symbol, current_timestamp):
                # Find active bracket for current position
                active_bracket = self._find_active_bracket(current_symbol)

                if active_bracket is not None:
                    # Trigger manual sale
                    active_bracket.MANUAL_SALE = True

                    # Queue next target for next tick
                    self._pending_target = target
                    self._pending_switch_state = "SWITCHING"

                    # Clear entry time for old symbol
                    if current_symbol in self._symbol_entry_time:
                        del self._symbol_entry_time[current_symbol]

                    # Note: No orders created this tick
                    # Manual sale will process, then next tick we'll buy
            else:
                # Holding period not elapsed → Ignore signal
                # Let bracket exit naturally via stop/profit
                pass

        return TickResults(orders=orders)
