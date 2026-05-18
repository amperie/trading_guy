"""
Base class for trading algorithms.

All trading algorithms inherit from Algorithm and implement on_data_logic()
to define their strategy. The base class manages price history tracking and
provides a consistent interface for the engine to call.

Data flow:
    Engine calls on_data(tick) each tick
        -> Algorithm updates price history automatically
        -> Algorithm calls on_data_logic(tick) (your strategy)
        -> Returns list of MarketSignal objects
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, final, List

from trading.core.classes import MarketSignal, PriceData
from collections import defaultdict, deque
from utils.logger import Logger
from utils.utils import merge_nested_config

logger = Logger().get_logger(__name__)


class Algorithm(ABC):
    """
    Base class for trading algorithms.

    Provides automatic price history management, a warmup gate, and a consistent
    interface for the engine to call on each tick of market data.  Subclasses
    implement their strategy logic in on_data_logic().

    Subclassing notes:
        - Implement on_data_logic(data) with your strategy logic.
        - Do NOT override on_data(); it is @final and manages history updates,
          the tick counter, and the warmup gate before delegating to on_data_logic().
        - Access rolling price history via self.price_history[symbol] (deque of
          closing prices) and self.price_data_history[symbol] (deque of PriceData).
        - History deques are automatically capped at history_length entries.
        - Pass algorithm-specific parameters through cfg dict.
        - Override required_warmup_bars (property) when your algorithm needs more
          bars than history_length before it can generate reliable signals.

    Warmup gate:
        on_data() always calls on_data_logic() so that algorithm-specific internal
        state (e.g. indicator histories) is built up from the very first tick.
        However, it returns an empty list until is_warmed_up is True — i.e. until
        _ticks_seen >= required_warmup_bars.

        This gate is the primary mechanism for session replay: the BacktestingEngine
        feeds both warmup bars and live-session bars in one continuous stream, and
        the gate suppresses signals for the warmup portion automatically.

        For live trading, call warm_up(historical_ticks) before the engine loop.
        Each warmup tick increments _ticks_seen, so the gate opens naturally after
        the configured number of bars.  BaseEngine.warm_up(data_provider) provides
        a convenience wrapper that iterates a DataProvider automatically.

    Attributes:
        cfg (dict): Algorithm configuration. Merged with default_cfg on init.
        history_length (int): Max number of ticks to retain per symbol.
            Set to 0 to disable history tracking.
        _ticks_seen (int): Total ticks processed via on_data() (including warmup).
        price_history (dict[str, deque]): Rolling closing prices keyed by symbol.
            Only populated when history_length > 0.
        price_data_history (dict[str, deque]): Rolling PriceData objects keyed by symbol.
            Only populated when history_length > 0.
        full_history (list): Every tick received, appended unconditionally when
            cfg["full_history"] is True.

    Example — basic algorithm::

        class MyAlgo(Algorithm):
            def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
                for pd in data:
                    prices = self.price_history.get(pd.symbol)
                    if prices and len(prices) >= 20:
                        sma = sum(prices) / len(prices)
                        if pd.close > sma:
                            return [MarketSignal(SignalType.BUY, pd.symbol, 75)]
                return []

        algo = MyAlgo(cfg={"history_length": 20})

        # Pre-populate history before live trading:
        algo.warm_up(list(data_provider.iterate()))

    Example — custom warmup threshold::

        class MyMACDAlgo(Algorithm):
            @property
            def required_warmup_bars(self) -> int:
                # MACD needs slow_period + signal_period bars before it stabilises
                return self.macd_slow + self.macd_signal

            def on_data_logic(self, data):
                ...
    """

    default_cfg = {
        "history_length": 0,
        "full_history": False,
    }
    _reconfigure_reserved_attrs = {
        "cfg",
        "history_length",
        "warm_up_ticks",
        "_ticks_seen",
        "_required_warmup_bars_override",
        "price_history",
        "price_data_history",
        "full_history",
    }

    def __init__(self, cfg: Dict[str, Any]=None, history_length: int=0):
        """
        Initialize the algorithm with configuration and history settings.

        Args:
            cfg: Algorithm configuration dict. Merged on top of default_cfg.
                Common keys:
                - history_length (int): Rolling window size per symbol (default: 0).
                - full_history (bool): Whether to store every tick (default: False).
                Any additional keys are available to the subclass via self.cfg.
            history_length: Rolling window size. Overrides cfg["history_length"]
                if provided (i.e. > 0). When > 0, price_history and
                price_data_history are populated as deques with maxlen.
                Also sets the default value of required_warmup_bars, which
                controls when the warmup gate opens.
        """
        super().__init__()
        if cfg is None:
            cfg = self.default_cfg
        else:
            cfg = {**self.default_cfg, **cfg}

        self.cfg = cfg
        self.history_length = history_length
        self.warm_up_ticks = cfg.get("warm_up_ticks", history_length)
        self._ticks_seen: int = 0
        self._required_warmup_bars_override: int | None = None
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque())
        self.price_data_history: Dict[str, deque] = defaultdict(lambda: deque())
        self.full_history: List[Dict[str, PriceData]] = []

        if self.history_length > 0:
            # Initialize dequeues with maxlen for automatic size management
            self.price_history = defaultdict(lambda: deque(maxlen=self.history_length))
            self.price_data_history = defaultdict(lambda: deque(maxlen=self.history_length))
        else:
            self.price_history = {}

        self.full_history = []
        logger.debug(f"{self.__class__.__name__} initialized: history_length={self.history_length} full_history={self.cfg.get('full_history', False)}")

    def _update_history(self, data: list[PriceData]):
        """
        Append current tick data to the rolling and full history stores.

        Called automatically by on_data() before on_data_logic(). When
        history_length > 0, appends each symbol's closing price to
        price_history and the full PriceData to price_data_history. When
        cfg["full_history"] is True, the entire tick is appended to
        full_history.

        Args:
            data: List of PriceData for the current tick (one per symbol).
        """
        for pd in data:
            symbol = pd.symbol
            # Append to limited history
            if self.history_length > 0:
                self.price_history[symbol].append(pd.close)
                self.price_data_history[symbol].append(pd)
            # Append full history if configured to do so
        if "full_history" in self.cfg and self.cfg["full_history"] is True:
            self.full_history.append(data)

    @final
    def get_price_history(self) -> Dict[str, deque]:
        """
        Return the rolling closing-price history for all symbols.

        Returns:
            Dict mapping symbol -> deque of float closing prices.
            Empty dict if history_length is 0.
        """
        return self.price_history

    @final
    def get_price_data_history(self) -> Dict[str, deque]:
        """
        Return the rolling PriceData history for all symbols.

        Returns:
            Dict mapping symbol -> deque of PriceData objects.
            Empty defaultdict if history_length is 0.
        """
        return self.price_data_history

    @final
    def get_full_history(self) -> List:
        """
        Return the complete tick history.

        Only populated when cfg["full_history"] is True. Each entry is the
        full list[PriceData] from a single tick.

        Returns:
            List of list[PriceData], one entry per tick received.
        """
        return self.full_history

    def warm_up(self, historical_ticks: list[list[PriceData]]):
        """Pre-populate price history and internal algorithm state.

        Feeds each tick through on_data() (the full pipeline: history update +
        on_data_logic()) so that algorithm-specific internal state (e.g. indicator
        histories, rolling calculations) is built up automatically.

        Each tick increments _ticks_seen.  After warm_up() completes with N ticks,
        _ticks_seen == N.  If N >= required_warmup_bars, is_warmed_up is True and
        the very next on_data() call in the live engine loop will emit signals.

        The signals returned by on_data() during warm_up are discarded by the gate
        (is_warmed_up is False until the threshold is reached), so no spurious
        orders are generated even if on_data_logic() tries to signal early.

        Args:
            historical_ticks: List of ticks (each tick is a list[PriceData])
                in chronological order, as returned by DataProvider.iterate().
        """
        for tick in historical_ticks:
            self.on_data(tick)  # full pipeline; gate suppresses early signals
        logger.info(f"Algorithm warmed up with {len(historical_ticks)} historical ticks")

    @property
    def required_warmup_bars(self) -> int:
        """Number of ticks that must be processed before signals are emitted.

        Defaults to history_length. Subclasses can override to set a larger
        value when additional bars are needed beyond the rolling window.
        Set _required_warmup_bars_override to pin this value externally
        (e.g. session replay to align the gate to exact warmup bars fetched).
        """
        if self._required_warmup_bars_override is not None:
            return self._required_warmup_bars_override
        return self.history_length

    @property
    def is_warmed_up(self) -> bool:
        """True once enough ticks have been seen for signals to be valid."""
        return self._ticks_seen >= self.required_warmup_bars

    @final
    def on_data(self, data: list[PriceData]) -> list[MarketSignal]:
        """
        Process a tick of market data. Called by the engine each tick.

        Updates price history, then delegates to on_data_logic() for the
        subclass strategy. Do NOT override this method.

        Args:
            data: List of PriceData for the current tick (one per symbol
                present in the data stream).

        Returns:
            List of MarketSignal objects generated by the strategy.
            Returns [] until is_warmed_up is True (warmup gate).
        """
        self._update_history(data)
        self._ticks_seen += 1
        signals = self.on_data_logic(data)
        if not self.is_warmed_up:
            return []
        if signals:
            logger.debug(f"Signals generated: {[(s.type.name, s.symbol, s.strength) for s in signals]}")
        return signals

    @abstractmethod
    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        """
        Strategy logic implementation. Override this in your subclass.

        Called once per tick after history has been updated. Access rolling
        price data via self.price_history[symbol] and
        self.price_data_history[symbol].

        Args:
            data: List of PriceData for the current tick (one per symbol).

        Returns:
            List of MarketSignal objects. Return an empty list if the
            strategy has no signal for this tick.
        """
        raise NotImplementedError("on_data_logic needs to be overriden")

    def get_indicator_dataframe(self):
        """Return a DataFrame of calculated indicator values for debug dumps.

        Override in your algorithm to expose indicator series to the debug
        REPL's 'dump al.indicators' command. Return a pandas DataFrame with
        at minimum a 'timestamp' column plus one column per indicator.

        Returns None by default (no indicators exposed).
        """
        return None

    def get_bias(self, data: list[PriceData] | None = None) -> float:
        """Return the algorithm's current directional bias on a [-1, 1] scale.

        This hook is optional and exists primarily for ensemble/meta-strategy
        use cases. A value of +1 indicates maximum bullish conviction, -1
        indicates maximum bearish conviction, and 0 indicates neutral/no edge.

        Existing algorithms do not need to override this method; the default
        implementation returns 0.0 so legacy behavior remains unchanged.
        """
        return 0.0

    def _iter_reconfigure_leaf_values(self, params: dict) -> list[tuple[str, Any]]:
        leaf_values: list[tuple[str, Any]] = []
        for key, value in params.items():
            if isinstance(value, dict):
                leaf_values.extend(self._iter_reconfigure_leaf_values(value))
            else:
                leaf_values.append((key, value))
        return leaf_values

    def _sync_attrs_from_reconfigure(self, new_params: dict) -> None:
        for attr_name, value in self._iter_reconfigure_leaf_values(new_params):
            if attr_name in self._reconfigure_reserved_attrs:
                continue
            if hasattr(self, attr_name):
                setattr(self, attr_name, value)

    def reconfigure(self, new_params: dict) -> None:
        """Update algorithm parameters without losing price history.

        Updates self.cfg, auto-syncs matching existing instance attributes,
        and logs the change. Subclasses only need to override this when they
        maintain derived state that cannot be refreshed from plain config
        leaves alone.
        """
        merge_nested_config(self.cfg, new_params)
        self._sync_attrs_from_reconfigure(new_params)
        logger.info(f"{self.__class__.__name__} reconfigured: {new_params}")
