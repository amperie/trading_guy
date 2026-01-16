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

    def _current_symbol(self) -> str | None:
        for symbol in (self.upro_symbol, self.spxu_symbol):
            if symbol in self.positions and self.positions[symbol].quantity > 0:
                return symbol
        return None

    def process_tick_market_signals_logic(
            self, signals: list[MarketSignal],
            tick: list[PriceData]) -> TickResults:
        orders: list[Order] = []

        # Execute queued target from previous tick (next-bar execution).
        if self._pending_target is not None:
            target = self._pending_target
            current_symbol = self._current_symbol()

            if current_symbol != target:
                if current_symbol is not None:
                    qty = self.positions[current_symbol].quantity
                    if qty > 0:
                        orders.append(Order.create_market_order(
                            current_symbol, OrderAction.SELL, qty, self.tx_cost, tick
                        ))

                pd = find_pricedata_in_list(target, tick)
                if pd is not None:
                    qty = int(self.cash / pd.close)
                    if qty > 0:
                        orders.append(Order.create_market_order(
                            target, OrderAction.BUY, qty, self.tx_cost, tick
                        ))

            self._pending_target = None

        # Queue next target based on current tick signals.
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
            self._pending_target = target_signal.symbol
            target_signal.metadata["order_target_next_tick"] = True

        return TickResults(orders=orders)
