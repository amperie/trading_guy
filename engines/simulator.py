"""
Simulator for backtesting
Iterates and feeds data through the system as if it were coming from a real time source
"""
from core.classes import PriceData
from data_providers.data_provider import DataProvider
from data_providers.test_data_provider import TestDataProvider
from utils.utils import instantiate_from_string
from utils.config_manager import ConfigManager

class Simulator:
    dp = DataProvider

    def __init__(self, cfg_section_to_use:str="simulator", cfg:dict=None):
        if cfg is None:
            cfg = {}
        cfg_dict = {**ConfigManager().get(cfg_section_to_use), **cfg}
        self.cfg_dict = cfg_dict
        self.cfg = ConfigManager.dict_to_namespace(cfg_dict)

        dp_cfg_dict = cfg_dict["data_provider"]
        self.dp = instantiate_from_string(
            self.cfg.data_provider.provider, cfg=dp_cfg_dict)