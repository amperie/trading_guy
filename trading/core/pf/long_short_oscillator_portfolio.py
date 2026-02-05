from trading.core.portfolio import Portfolio
from trading.core.classes import MarketSignal, PriceData, SignalType, OrderAction, Order, TickResults
from utils.utils import find_pricedata_in_list
from enum import Enum
from datetime import datetime

class SignalState(Enum):
    NeutralState = 0
    LongState = 1
    ShortState = 2

class TransactionState(Enum):
    NothingPending = 0
    SalePending = 1

class LongShortOscillatorPortfolio(Portfolio):

    def __init__(self, cfg: dict = None, order_manager=None, starting_cash: float = 0.0,
                 starting_positions=None, keep_history: bool = False):

        super().__init__(cfg, order_manager, starting_cash, starting_positions, keep_history)
        self.long_symbol = self.cfg["long_symbol"]
        self.short_symbol = self.cfg["short_symbol"]
        self.benchmark_symbol = self.cfg.get("benchmark_symbol", None)
        self.use_benchmark_symbol = self.cfg.get("use_benchmark_symbol", False)
        self.min_signal_strength = self.cfg.get("min_signal_strength", 0)
        self.tx_cost = self.cfg["tx_cost"]
        self.profit_pct = self.cfg["profit_pct"]
        self.stop_loss_pct = self.cfg["stop_loss_pct"]
        self.holding_period_hours = self.cfg.get("holding_period_hours", 0)

        # Internal variables
        self._last_entry = datetime(1970, 1, 1, 0, 0, 0)
        self._pf_state = SignalState.NeutralState
        self._tx_state = TransactionState.NothingPending
        self._last_tx = None

    def process_tick_market_signals_logic(self, signals: list[MarketSignal], tick: list[PriceData]) -> TickResults:
        pass