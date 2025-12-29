from dataset_builder_engines import IndicatorLagDataSetEngine
from data_providers.test_data_provider import TestDataProvider

symbol = "GDXU"
cfg = {
    "symbol": symbol,
    "lag_values": 10,
    "loss_pct": 0.05,
    "profit_pct": 0.1,
    "look_ahead_period": 100,
    "output_file": f"../../data/ml/{symbol}_test_ml.csv",
    "history_length": 100,
}

data_path = f"data/{symbol}_5min_MarketHours.csv"
dp_cfg = {
    "path": data_path, "provider":"data_providers.test_data_provider.TestDataProvider",
    "truncate": 10000000
}
dp = TestDataProvider(dp_cfg)

dsbe = IndicatorLagDataSetEngine(cfg=cfg, dp=dp)
dsbe.run()
