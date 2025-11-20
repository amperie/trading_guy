import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.simulator import Simulator
from core.om.backtesting_om import BacktestingOM
from core.pf.single_symbol_portfolio import SingleSymbolPortfolio
from core.portfolio import Portfolio
from algorithms.test_algorithm import TestAlgorithm
from data_providers.test_data_provider import TestDataProvider

# Get absolute path to data file relative to project root
project_root = Path(__file__).parent.parent
data_path = str(project_root / "data" / "test_data.csv")

om = BacktestingOM()
al = TestAlgorithm({"history_length": 10, "full_history": True})
dp_cfg = {"path": data_path, "provider":"data_providers.test_data_provider.TestDataProvider"}
dp = TestDataProvider(dp_cfg)
pf = SingleSymbolPortfolio({'symbol':'AAPL', 'keep_history': True, "cash": 1000})
pf.set_order_manager(om)

s = Simulator(om=om, al=al, pf=pf)
s.run()

error_out_here

dp_cfg = {"path": data_path, "provider":"data_providers.test_data_provider.TestDataProvider"}
# al_cfg = {"algorithm":"override"}
s_cfg = {"data_provider": dp_cfg}
s = Simulator(cfg=s_cfg)
s.run()
pass