from trading.core.classes import PriceData, TickResults
from trading.core.algorithm import Algorithm
from trading.core.portfolio import Portfolio
from trading.core.om.order_manager import OrderManager
from trading.data_providers.data_provider import DataProvider
from abc import ABC, abstractmethod


class BaseEngine(ABC):
    """
    Base class for execution engines.

    Subclassing notes:
    - Implement run() as the main loop or orchestration entrypoint.
    - Implement on_tick() to handle a single tick of data.
    - Implement finalize() for any end-of-run cleanup or summaries.
    - Engines coordinate DataProvider, Algorithm, Portfolio, and OrderManager.
    """
    dp = DataProvider
    al = Algorithm
    pf = Portfolio
    om = OrderManager

    def __init__(
            self, cfg:dict= None, dp: DataProvider = None,
            al: Algorithm=None, om: OrderManager=None,
            pf: Portfolio=None
        ):

        if cfg is None: cfg = {}
        self.cfg = cfg
        self.dp = dp
        self.al = al
        self.om = om
        self.pf = pf

    @abstractmethod
    def run(self):
        """Run the engine's main loop."""
        raise NotImplementedError()

    @abstractmethod
    def on_tick(self, tick: list[PriceData]) -> TickResults:
        """Process one tick and return TickResults."""
        raise NotImplementedError()

    @abstractmethod
    def finalize(self):
        """Finalize the engine (cleanup, summaries, reports)."""
        raise NotImplementedError()
