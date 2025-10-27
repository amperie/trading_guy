"""
Simulator for backtesting
Iterates and feeds data through the system as if it were coming from a real time source
"""
from core.classes import PriceData
from core.algorithm import Algorithm
from core.portfolio import Portfolio
from core.order_manager import OrderManager
from data_providers.data_provider import DataProvider
from data_providers.test_data_provider import TestDataProvider
from core.classes import MarketSignal
from utils.utils import instantiate_from_string
from utils.config_manager import ConfigManager

class Simulator:
    dp = DataProvider
    al = Algorithm
    pf = Portfolio
    om = OrderManager

    def __init__(
            self, cfg_section_to_use:str="simulator", cfg:dict=None,
            al: Algorithm=None, om: OrderManager=None,
            pf: Portfolio=None
        ):
        #TODO: Get rid of all this dynamic typing stuff

        if cfg is None:
            cfg = {}
        cfg_dict = {**ConfigManager().get(cfg_section_to_use), **cfg}
        self.cfg_dict = cfg_dict
        self.cfg = ConfigManager.dict_to_namespace(cfg_dict)

        dp_cfg_dict = cfg_dict["data_provider"]
        self.dp = instantiate_from_string(
            self.cfg.data_provider.provider, cfg=dp_cfg_dict)

        if al is None:
            # Load from config file
            al_cfg = cfg_dict["algorithm"]
            self.al = instantiate_from_string(
                al_cfg['algorithm'], cfg=al_cfg
            )
        else:
            self.al = al

        if om is None:
            om_cfg = cfg_dict["order_manager"]
            self.om = instantiate_from_string(
                om_cfg['order_manager'], cfg=om_cfg
            )
        else:
            self.om = om

        if pf is None:
            pf_cfg = cfg_dict["portfolio"]
            pf = Portfolio(pf_cfg)
        pf.set_order_manager(self.om)
        self.pf = pf

    def run(self):
        """Run through the data, feed it into the algorithm
        Feed all the market signals into the portfolio for it to execute on signals
        Portfolio outputs a list of orders, feed those to the OrderManager
        """

        for tick in self.dp.iterate():
            self.process_tick(tick)

    def process_tick(self, tick: list[PriceData]):

        market_signals = self.al.on_data(tick)
        orders = self.pf.process_tick_market_signals(market_signals, tick)