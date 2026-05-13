from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    implementation: str
    source_url: str | None = None
    source_path: str | None = None
    class_name: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class MLflowConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    tracking_uri: str | None = None
    experiment_name: str | None = None
    auto_log_system_info: bool | None = None


class StateStoreConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    connection_uri: str | None = None
    database: str | None = None
    session_id: str | None = None


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    log_to_mlflow: bool = True
    experiment_name: str | None = None
    run_name: str | None = None
    description: str | None = None
    benchmarks: dict[str, str] = Field(default_factory=dict)


class AggregationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    aggregation_period_minutes: int = 5
    aggregation_start_minutes: int | None = None
    use_market_open: bool = True
    market_open_hour: int = 9
    market_open_minute: int = 30
    batch_symbols: bool = False
    expected_symbols: list[str] = Field(default_factory=list)
    batch_timeout_seconds: float = 2.0


class AlpacaConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    api_key: str = ""
    secret_key: str = ""
    paper: bool | None = None
    time_in_force: str | None = None
    override_url: str | None = None
    symbols_to_subscribe: list[str] = Field(default_factory=list)
    subscribe_to_bars: bool = True
    subscribe_to_quotes: bool = False
    subscribe_to_trades: bool = False
    warmup: dict[str, Any] | None = None


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False


class WalkForwardConfig(BaseModel):
    model_config = ConfigDict(extra="allow")


class HPOConfig(BaseModel):
    model_config = ConfigDict(extra="allow")


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["backtest", "live", "walk-forward", "hpo", "session-replay"]
    algorithm: ComponentConfig
    portfolio: ComponentConfig
    order_manager: ComponentConfig
    data_provider: ComponentConfig | None = None
    alpaca: AlpacaConfig | None = None
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    hpo: HPOConfig = Field(default_factory=HPOConfig)
    state_store: StateStoreConfig = Field(default_factory=StateStoreConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    logging: dict[str, Any] = Field(default_factory=dict)

    def execution_mode(self) -> str:
        if self.mode == "live":
            return "broker"
        if self.mode == "session-replay":
            return "replay"
        return "simulation"
