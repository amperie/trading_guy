from dataset_builder_engines import IndicatorLagDataSetEngine
from data_providers.test_data_provider import TestDataProvider

symbol = "GDXU"
cfg = {
    "symbol": symbol,
    "target_configs": [
        {'loss_pct': 0.05, 'profit_pct': 0.10, 'look_ahead_period': 100, 'name': 'target_5_10_100'},
        {'loss_pct': 0.03, 'profit_pct': 0.06, 'look_ahead_period': 50, 'name': 'target_3_6_50'},
        {'loss_pct': 0.10, 'profit_pct': 0.20, 'look_ahead_period': 200, 'name': 'target_10_20_200'}
    ],
    "lag_values": 10,
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
