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

logger = Logger().get_logger(__name__)


class Algorithm(ABC):
    """
    Base class for trading algorithms.

    Provides automatic price history management and a consistent interface
    for the engine to call on each tick of market data. Subclasses implement
    their strategy logic in on_data_logic().

    Subclassing notes:
        - Implement on_data_logic(data) with your strategy logic.
        - Do NOT override on_data(); it is @final and manages history updates
          before delegating to on_data_logic().
        - Access rolling price history via self.price_history[symbol] (deque of
          closing prices) and self.price_data_history[symbol] (deque of PriceData).
        - History deques are automatically capped at history_length entries.
        - Pass algorithm-specific parameters through cfg dict.

    Warm-up:
        Call warm_up(historical_ticks) before the live run loop to pre-populate
        price_history and price_data_history from historical data. This avoids
        a cold-start period where the algorithm cannot generate signals because
        its history deques are empty.

        warm_up() feeds ticks through _update_history() without calling
        on_data_logic(), so no spurious signals are generated. It is not
        @final — subclasses can override it to initialize additional state
        (e.g. indicator objects, rolling calculations) from historical data.

        BaseEngine.warm_up(data_provider) provides a convenience wrapper
        that iterates a DataProvider and calls this method automatically.

    Attributes:
        cfg (dict): Algorithm configuration. Merged with default_cfg on init.
        history_length (int): Max number of ticks to retain per symbol.
            Set to 0 to disable history tracking.
        price_history (dict[str, deque]): Rolling closing prices keyed by symbol.
            Only populated when history_length > 0.
        price_data_history (dict[str, deque]): Rolling PriceData objects keyed by symbol.
            Only populated when history_length > 0.
        full_history (list): Every tick received, appended unconditionally when
            cfg["full_history"] is True.

    Example:
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
    """

    default_cfg = {
        "history_length": 0,
        "full_history": False,
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
        """
        super().__init__()
        if cfg is None:
            cfg = self.default_cfg
        else:
            cfg = {**self.default_cfg, **cfg}

        self.cfg = cfg
        self.history_length = history_length
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
        """Pre-populate price history without generating signals.

        Feeds each tick through _update_history() so the algorithm has
        sufficient history from the very first live tick. Not @final —
        subclasses can extend this to initialize additional state (e.g.
        indicator objects, rolling calculations).

        Args:
            historical_ticks: List of ticks (each tick is a list[PriceData])
                in chronological order, as returned by DataProvider.iterate().
        """
        for tick in historical_ticks:
            self._update_history(tick)
        logger.info(f"Algorithm warmed up with {len(historical_ticks)} historical ticks")

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
        """
        self._update_history(data)
        signals = self.on_data_logic(data)
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
