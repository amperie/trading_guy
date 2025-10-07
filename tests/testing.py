import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.simulator import Simulator

dp_cfg = {"path":"../data/test_data.csv", "provider":"data_providers.test_data_provider.TestDataProvider"}
# al_cfg = {"algorithm":"override"}
s_cfg = {"data_provider": dp_cfg}
s = Simulator(cfg=s_cfg)
s.run()
pass