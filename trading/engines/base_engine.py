from trading.core.classes import PriceData, TickResults
from trading.core.algorithm import Algorithm
from trading.core.portfolio import Portfolio
from trading.core.order_manager import OrderManager
from trading.data_providers.data_provider import DataProvider
from abc import ABC, abstractmethod


class BaseEngine(ABC):
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
        raise NotImplementedError()

    @abstractmethod
    def on_tick(self, tick: list[PriceData]) -> TickResults:
        raise NotImplementedError()

    @abstractmethod
    def finalize(self):
        raise NotImplementedError()