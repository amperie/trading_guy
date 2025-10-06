from engines.simulator import Simulator

dp_cfg = {"path":"../data/SPXU.csv", "provider":"data_providers.test_data_provider.TestDataProvider"}
al_cfg = {"algorithm":"override"}
s_cfg = {"data_provider": dp_cfg, "algorithm": al_cfg}
s = Simulator(cfg=s_cfg)
pass