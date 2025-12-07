from core.classes import PriceData, Order, TickResults
from core.algorithm import Algorithm
from core.portfolio import Portfolio
from core.order_manager import OrderManager
from data_providers.data_provider import DataProvider
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