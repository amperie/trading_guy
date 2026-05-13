from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AllowExtraModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class SpyTrendMACDAlgorithmParams(BaseModel):
    spy_symbol: str = "SPY"
    upro_symbol: str = "UPRO"
    spxu_symbol: str = "SPXU"
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    strength_scale: float = 20.0
    history_length: int = 0


class DualSymbolSwitchPortfolioParams(BaseModel):
    upro_symbol: str = "UPRO"
    spxu_symbol: str = "SPXU"
    cash: float = 0.0
    keep_history: bool = False
    min_signal_strength: int = 0
    stop_pct: float = 5.0
    profit_pct: float = 10.0
    holding_period_hours: float = 2.0
    tx_cost: float = 0.0
    sync_with_broker: bool = False
    sync_interval: int = 60
    order_sync_limit: int = 0


class SingleSymbolPortfolioParams(BaseModel):
    symbol: str
    cash: float = 0.0
    keep_history: bool = False
    stop_pct: float = 0.0
    profit_pct: float = 0.0
    tx_cost: float = 0.0
    sync_with_broker: bool = False
    sync_interval: int = 60


class BacktestingOrderManagerParams(BaseModel):
    market_hours_only: bool = False


class AlpacaOrderManagerParams(BaseModel):
    api_key: str = ""
    secret_key: str = ""
    paper: bool = True
    time_in_force: str = "day"


class TestDataProviderParams(BaseModel):
    path: str
    truncate: int = 0


class AlpacaDataProviderParams(BaseModel):
    api_key: str = ""
    secret_key: str = ""
    symbols: list[str]
    timeframe: str = "Minute"
    start_date: str | None = None
    end_date: str | None = None
    limit: int | None = None


KNOWN_COMPONENT_MODELS: dict[str, type[BaseModel]] = {
    "trading.algorithms.spy_trend_macd_algorithm.SpyTrendMACDAlgorithm": SpyTrendMACDAlgorithmParams,
    "trading.core.pf.dual_symbol_switch_portfolio.DualSymbolSwitchPortfolio": DualSymbolSwitchPortfolioParams,
    "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio": SingleSymbolPortfolioParams,
    "trading.core.om.backtesting_om.BacktestingOrderManager": BacktestingOrderManagerParams,
    "trading.core.om.alpaca_om.AlpacaOrderManager": AlpacaOrderManagerParams,
    "trading.data_providers.test_data_provider.TestDataProvider": TestDataProviderParams,
    "trading.data_providers.alpaca_data_provider.AlpacaDataProvider": AlpacaDataProviderParams,
}
