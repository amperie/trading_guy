from trading.core.portfolio import Portfolio
from trading.core.classes import MarketSignal, PriceData, SignalType, OrderAction, Order, TickResults
from utils.utils import find_pricedata_in_list


class DualSymbolSwitchPortfolio(Portfolio):
    """
    Switch portfolio between UPRO and SPXU with next-bar execution.
    Uses BUY signals for the target symbol and never holds both at once.
    """

    def __init__(self, cfg: dict = None, order_manager=None, starting_cash: float = 0.0,
                 starting_positions=None, keep_history: bool = False):
        super().__init__(cfg, order_manager, starting_cash, starting_positions, keep_history)
        self.upro_symbol = self.cfg.get("upro_symbol", "UPRO")
        self.spxu_symbol = self.cfg.get("spxu_symbol", "SPXU")
        self.tx_cost = self.cfg.get("tx_cost", 0.0)
        self.min_signal_strength = self.cfg.get("min_signal_strength", 0)
        self._pending_target = None
        self._pending_switch_state = None  # None or "SELLING"

    def _current_symbol(self) -> str | None:
        for symbol in (self.upro_symbol, self.spxu_symbol):
            if symbol in self.positions and self.positions[symbol].quantity > 0:
                return symbol
        return None

    def process_tick_market_signals_logic(
            self, signals: list[MarketSignal],
            tick: list[PriceData]) -> TickResults:
        orders: list[Order] = []

        # If we're waiting for a SELL to complete, only BUY when the position is gone.
        if self._pending_target is not None:
            current_symbol = self._current_symbol()
            if current_symbol is None:
                target = self._pending_target
                pd = find_pricedata_in_list(target, tick)
                if pd is not None:
                    qty = int(self.cash / pd.close)
                    if qty > 0:
                        orders.append(Order.create_market_order(
                            target, OrderAction.BUY, qty, self.tx_cost, tick
                        ))
                self._pending_target = None
                self._pending_switch_state = None
                return TickResults(orders=orders)
            return TickResults(orders=orders)

        # Select target from current tick signals.
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

        if target_signal is not None:
            target = target_signal.symbol
            current_symbol = self._current_symbol()

            # Case 1: No position -> buy immediately.
            if current_symbol is None:
                pd = find_pricedata_in_list(target, tick)
                if pd is not None:
                    qty = int(self.cash / pd.close)
                    if qty > 0:
                        orders.append(Order.create_market_order(
                            target, OrderAction.BUY, qty, self.tx_cost, tick
                        ))

            # Case 2: Different position -> sell now, buy after sell fills.
            elif current_symbol != target:
                qty = self.positions[current_symbol].quantity
                if qty > 0:
                    orders.append(Order.create_market_order(
                        current_symbol, OrderAction.SELL, qty, self.tx_cost, tick
                    ))
                    self._pending_target = target
                    self._pending_switch_state = "SELLING"
                    target_signal.metadata["order_target_after_sell"] = True

            # Case 3: Already in target -> no action.

        return TickResults(orders=orders)
