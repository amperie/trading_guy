from __future__ import annotations

import os
import signal
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from trading.engines import walk_forward_engine as wf_module
from trading.engines.walk_forward_engine import WalkForwardEngine
from trading.engines.walk_forward_policy import compute_walk_forward_periods, metric_value
from trading.data_providers.data_provider import DataProvider
from trading.core.algorithm import Algorithm
from trading.core.portfolio import Portfolio
from trading.core.om.order_manager import OrderManager
from trading.launchers import run_backtest_ray


class DummyDataProvider(DataProvider):
    def load_data(self):
        self.data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=200, freq="D"),
                "symbol": ["SPY"] * 200,
                "open": [1.0] * 200,
                "high": [1.0] * 200,
                "low": [1.0] * 200,
                "close": [1.0] * 200,
                "volume": [1.0] * 200,
            }
        )


class DummyAlgorithm(Algorithm):
    def __init__(self, cfg=None, history_length: int = 0):
        super().__init__(cfg=cfg, history_length=history_length)

    def on_data_logic(self, data):
        return []


class DummyNestedConfigAlgorithm(Algorithm):
    def __init__(self, cfg=None, history_length: int = 0):
        super().__init__(cfg=cfg or {}, history_length=history_length)
        self.alpha = self.cfg.get("alpha", 1)
        nested = self.cfg.get("nested", {})
        self.beta = nested.get("beta", 2)

    def on_data_logic(self, data):
        return []


class DummyOrderManager(OrderManager):
    def _update_order_status_from_backend(self, order, current_tick=None, positions=None, pf_cash: float = 0.0):
        return order

    def _update_orders_statuses_from_backend(self, orders, current_tick=None, positions=None, pf_cash: float = 0.0):
        return []

    def _submit_order_to_backend(self, order, current_tick=None, positions=None, pf_cash: float = 0.0):
        return order

    def _cancel_order(self, order_id: str):
        return self.all_orders[order_id]


class DummyPortfolio(Portfolio):
    def process_tick_market_signals_logic(self, signals, tick):
        return SimpleNamespace(orders=[])


def _build_engine(cfg: dict | None = None) -> WalkForwardEngine:
    dp = DummyDataProvider({"path": "ignored"})
    al = DummyAlgorithm({"history_length": 7}, history_length=7)
    om = DummyOrderManager()
    pf = DummyPortfolio({"symbol": "SPY", "cash": 1000.0, "keep_history": True}, om, 1000.0, {}, True)
    base_cfg = {
        "walk_forward": {
            "optimization_window_days": 30,
            "validation_window_days": 5,
            "trading_window_days": 10,
            "algorithm_param_keys": ["alpha"],
            "portfolio_param_keys": ["risk"],
        },
        "experiment_name": "wf-exp",
        "run_name": "wf-run",
        "description": "wf-desc",
        "state_store": {"enabled": False},
    }
    if cfg:
        for key, value in cfg.items():
            if isinstance(value, dict) and isinstance(base_cfg.get(key), dict):
                base_cfg[key] = {**base_cfg[key], **value}
            else:
                base_cfg[key] = value
    return WalkForwardEngine(
        cfg=base_cfg,
        dp=dp,
        al=al,
        om=om,
        pf=pf,
    )


def test_compute_periods_do_not_overlap():
    engine = _build_engine()

    periods = engine._compute_periods()

    assert periods
    assert periods[0] == wf_module.WalkForwardPeriod(
        optimization_start=datetime(2024, 1, 1),
        optimization_end=datetime(2024, 1, 31),
        validation_start=datetime(2024, 1, 31),
        validation_end=datetime(2024, 2, 5),
        trading_start=datetime(2024, 2, 5),
        trading_end=datetime(2024, 2, 15),
    )
    assert periods[1] == wf_module.WalkForwardPeriod(
        optimization_start=datetime(2024, 1, 11),
        optimization_end=datetime(2024, 2, 10),
        validation_start=datetime(2024, 2, 10),
        validation_end=datetime(2024, 2, 15),
        trading_start=datetime(2024, 2, 15),
        trading_end=datetime(2024, 2, 25),
    )
    for period in periods:
        assert period.validation_start == period.optimization_end
        assert period.trading_start == period.validation_end
    for previous, current in zip(periods, periods[1:]):
        assert current.optimization_start > previous.optimization_start
        assert current.optimization_start == previous.optimization_start + pd.Timedelta(days=engine.trading_window_days)
        assert current.trading_start == previous.trading_end


def test_compute_periods_final_window_includes_last_data_timestamp():
    engine = _build_engine(
        {
            "walk_forward": {
                "optimization_window_days": 30,
                "validation_window_days": 5,
                "trading_window_days": 10,
            }
        }
    )

    periods = engine._compute_periods()

    data_end = engine._get_date_range()[1]
    assert periods[-1].trading_end == data_end + pd.Timedelta(microseconds=1)


def test_compute_periods_rejects_non_positive_windows():
    with pytest.raises(ValueError, match="trading_window_days must be > 0"):
        compute_walk_forward_periods(
            data_start=datetime(2024, 1, 1),
            data_end=datetime(2024, 2, 1),
            optimization_window_days=30,
            validation_window_days=5,
            trading_window_days=0,
        )


def test_create_dp_for_range_preserves_intraday_end_boundaries():
    engine = _build_engine()

    dp = engine._create_dp_for_range(
        datetime(2024, 1, 1, 9, 30),
        datetime(2024, 1, 2, 9, 30),
    )

    assert dp.cfg["start_date"] == "2024-01-01 09:30:00"
    assert dp.cfg["end_date"] == "2024-01-02 09:29:59.999999"


def test_evaluate_period_compares_with_portfolio_params(monkeypatch):
    engine = _build_engine()
    captured = []

    monkeypatch.setattr(engine, "_create_dp_for_range", lambda start, end: f"{start.date()}->{end.date()}")
    monkeypatch.setattr(engine, "_get_date_range", lambda: (datetime(2023, 12, 1), datetime(2024, 3, 1)))
    monkeypatch.setattr(engine, "_run_optimization", lambda dp, warmup_dp=None: {"alpha": 2, "risk": 9})

    def fake_run_backtest(al_cfg, pf_cfg, dp, warmup_dp=None):
        captured.append((dict(al_cfg), dict(pf_cfg), dp, warmup_dp))
        annualized_return = 10.0 if pf_cfg["risk"] == 1 else 25.0
        return {
            "metrics": SimpleNamespace(
                annualized_return=annualized_return,
                total_trades=12,
                total_return_pct=1.0,
                sharpe_ratio=2.0,
                win_rate=0.5,
            ),
            "trades": [],
        }

    monkeypatch.setattr(engine, "_run_backtest", fake_run_backtest)
    monkeypatch.setattr(engine, "_build_algorithm", lambda cfg: SimpleNamespace(cfg=dict(cfg)))
    monkeypatch.setattr(engine, "_build_portfolio", lambda cfg, om: SimpleNamespace(cfg=dict(cfg)))

    class FakeBacktestingEngine:
        def __init__(self, cfg, dp, al, om, pf):
            self.dp = dp
            self.al = al
            self.om = om
            self.pf = pf

        def run(self):
            return None

    class FakeAnalysisEngine:
        def __init__(self, pf, om):
            self.pf = pf
            self.om = om

        def extract_trades(self):
            return []

        def calculate_metrics(self):
            return SimpleNamespace(total_return_pct=1.0, sharpe_ratio=2.0, total_trades=3)

    monkeypatch.setattr(wf_module, "BacktestingEngine", FakeBacktestingEngine)
    monkeypatch.setattr(wf_module, "AnalysisEngine", FakeAnalysisEngine)

    period = wf_module.WalkForwardPeriod(
        optimization_start=datetime(2024, 1, 1),
        optimization_end=datetime(2024, 1, 31),
        validation_start=datetime(2024, 2, 1),
        validation_end=datetime(2024, 2, 5),
        trading_start=datetime(2024, 2, 6),
        trading_end=datetime(2024, 2, 10),
    )
    current_al_cfg = {"alpha": 1}
    current_pf_cfg = {"symbol": "SPY", "cash": 1000.0, "keep_history": True, "risk": 1}

    result = engine._evaluate_period(0, period, current_al_cfg, current_pf_cfg)

    assert captured[0][1]["risk"] == 1
    assert captured[1][1]["risk"] == 9
    assert captured[0][2] == "2024-02-01->2024-02-05"
    assert captured[0][3] == "2024-01-01->2024-02-01"
    assert captured[1][3] == "2024-01-01->2024-02-01"
    assert result["adopted"] is True
    assert result["pf_cfg"]["risk"] == 9
    assert result["decision"]["challenger_metric"] == 25.0


def test_run_optimization_passes_warmup_provider_config(monkeypatch):
    engine = _build_engine()
    captured = {}

    def fake_tune(**kwargs):
        captured.update(kwargs)
        return {"alpha": 2}

    monkeypatch.setattr("trading.launchers.run_backtest_ray.tune_backtest_hyperparameters", fake_tune)

    result = engine._run_optimization(
        SimpleNamespace(cfg={"path": "opt.csv"}),
        warmup_dp=SimpleNamespace(cfg={"path": "warmup.csv"}),
    )

    assert result == {"alpha": 2}
    assert captured["base_data_provider_config"] == {"path": "opt.csv"}
    assert captured["warmup_data_provider_config"] == {"path": "warmup.csv"}


def test_apply_config_supports_nested_algorithm_keys():
    engine = _build_engine(
        {
            "walk_forward": {
                "optimization_window_days": 30,
                "validation_window_days": 5,
                "trading_window_days": 10,
                "algorithm_param_keys": ["regime.alpha", "rsi.period"],
                "portfolio_param_keys": ["risk"],
            },
        }
    )
    engine.original_al_cfg = {
        "regime": {"alpha": 1, "beta": 2},
        "rsi": {"period": 14, "threshold": 30},
    }
    engine.original_pf_cfg = {"risk": 1, "mode": "default"}

    al_cfg, pf_cfg = engine._apply_config({"regime.alpha": 5, "rsi.period": 9, "risk": 4})

    assert al_cfg["regime"] == {"alpha": 5, "beta": 2}
    assert al_cfg["rsi"] == {"period": 9, "threshold": 30}
    assert pf_cfg == {"risk": 4, "mode": "default"}


def test_create_mlflow_client_honors_disable_and_config(monkeypatch):
    engine = _build_engine({"walk_forward": {}, "log_to_mlflow": False})
    assert engine._create_mlflow_client() is None

    created = {}

    class FakeClient:
        def __init__(self, experiment_name=None, tracking_uri=None, enabled=True):
            created["experiment_name"] = experiment_name
            created["tracking_uri"] = tracking_uri
            created["enabled"] = enabled

    monkeypatch.setattr("utils.mlflow_client.MLflowClient", FakeClient)

    engine = _build_engine(
        {
            "walk_forward": {},
            "experiment_name": "wf-exp",
            "tracking_uri": "http://mlflow.local",
            "log_to_mlflow": True,
        }
    )
    engine._create_mlflow_client()

    assert created == {
        "experiment_name": "wf-exp",
        "tracking_uri": "http://mlflow.local",
        "enabled": True,
    }


def test_run_logs_single_mlflow_run(monkeypatch):
    engine = _build_engine(
        {
            "walk_forward": {
                "optimization_window_days": 30,
                "validation_window_days": 5,
                "trading_window_days": 10,
                "num_trials": 1,
                "algorithm_param_keys": [],
                "portfolio_param_keys": [],
            },
            "experiment_name": "wf-exp",
            "run_name": "wf-run",
            "description": "wf-desc",
            "log_to_mlflow": True,
        }
    )

    monkeypatch.setattr(
        engine,
        "_compute_periods",
        lambda: [
            wf_module.WalkForwardPeriod(
                optimization_start=datetime(2024, 1, 1),
                optimization_end=datetime(2024, 1, 31),
                validation_start=datetime(2024, 2, 1),
                validation_end=datetime(2024, 2, 5),
                trading_start=datetime(2024, 2, 6),
                trading_end=datetime(2024, 2, 10),
            )
        ],
    )

    captured = {}

    class FakeClient:
        def __init__(self):
            self._run_id = None

        def start_run(self, run_name=None, description=None):
            self._run_id = "parent-run-123"
            return self

        @property
        def is_active(self):
            return self._run_id is not None

        @property
        def run_id(self):
            return self._run_id

        def log_metrics(self, metrics):
            captured["aggregate_metrics"] = metrics

        def log_params(self, params):
            captured["aggregate_params"] = params

        def end_run(self):
            self._run_id = None

    monkeypatch.setattr(engine, "_create_mlflow_client", lambda: FakeClient())

    monkeypatch.setattr(
        engine,
        "_evaluate_period",
        lambda period_idx, period, current_al_cfg, current_pf_cfg: {
            "period_idx": period_idx,
            "period": period,
            "adopted": True,
            "al_cfg": dict(current_al_cfg),
            "pf_cfg": dict(current_pf_cfg),
            "algo_patch": {},
            "pf_patch": {},
            "event_id": "evt-123",
            "best_config": {},
            "decision": None,
        },
    )
    monkeypatch.setattr(
        engine,
        "_run_continuous_backtest",
        lambda plans: {
            "analysis": SimpleNamespace(),
            "results": {
                "metrics": SimpleNamespace(
                    total_return_pct=2.0,
                    annualized_return=3.0,
                    sharpe_ratio=1.5,
                    max_drawdown_pct=-4.0,
                    total_trades=1,
                    final_equity=1020.0,
                ),
                "report": "wf report",
            },
        },
    )
    monkeypatch.setattr(engine, "_build_optimization_events_rows", lambda plans: [])
    monkeypatch.setattr(engine, "_log_full_run_to_mlflow", lambda analysis, analysis_results, plans, aggregate: captured.setdefault("logged", aggregate))

    results = engine.run()

    assert "logged" in captured
    assert results["aggregate"]["num_periods"] == 1


def test_log_full_run_to_mlflow_ignores_start_run_failures(monkeypatch):
    engine = _build_engine(
        {
            "experiment_name": "wf-exp",
            "run_name": "wf-run",
            "description": "wf-desc",
            "log_to_mlflow": True,
        }
    )

    class FailingClient:
        def start_run(self, run_name=None, description=None):
            raise RuntimeError("mlflow unavailable")

    monkeypatch.setattr(engine, "_create_mlflow_client", lambda: FailingClient())
    monkeypatch.setattr(engine, "_build_optimization_events_rows", lambda plans: [])

    engine._log_full_run_to_mlflow(
        analysis=SimpleNamespace(),
        analysis_results={
            "metrics": SimpleNamespace(
                total_return_pct=2.0,
                annualized_return=3.0,
                sharpe_ratio=1.5,
                max_drawdown_pct=-4.0,
                total_trades=1,
                final_equity=1020.0,
            ),
            "report": "wf report",
        },
        plans=[],
        aggregate={"num_periods": 0},
    )


def test_log_full_run_to_mlflow_uses_full_analysis_logging(monkeypatch):
    engine = _build_engine(
        {
            "experiment_name": "wf-exp",
            "run_name": "wf-run",
            "description": "wf-desc",
            "log_to_mlflow": True,
            "mlflow_parameters": {"config.mode": "walk-forward"},
            "mlflow_artifact_paths": ["cfg.yaml"],
            "mlflow_tags": {"run_type": "walk_forward_window_winner"},
            "benchmark_paths": {"SPY": "data/SPY.csv"},
            "extra_mlflow_json_artifacts": {"window_summary.json": {"best_metric": 1.2}},
        }
    )
    captured = {}

    class FakeClient:
        enabled = True

        def __init__(self):
            self._run_id = None

        def start_run(self, run_name=None, description=None):
            captured["start"] = {"run_name": run_name, "description": description}
            self._run_id = "run-123"
            return self

        @property
        def run_id(self):
            return self._run_id

        def get_run_url(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def set_tags(self, tags):
            captured["tags"] = tags

        def log_metrics(self, metrics):
            captured.setdefault("metrics", []).append(metrics)

        def log_text(self, text, filename):
            captured.setdefault("texts", []).append((filename, text))

        def log_chart(self, figure, filename, format="png", dpi=150):
            captured.setdefault("charts", []).append(filename)

        def log_json(self, payload, filename):
            captured.setdefault("json", []).append((filename, payload))

    class FakeAnalysis:
        def log_to_mlflow(self, **kwargs):
            captured["analysis_log_kwargs"] = kwargs

        def plot_equity_curve(self, show=False):
            return "equity-figure"

    monkeypatch.setattr(engine, "_create_mlflow_client", lambda: FakeClient())
    monkeypatch.setattr(engine, "_build_optimization_events_rows", lambda plans: [{"period_idx": 0}])
    monkeypatch.setattr(engine, "_log_optimization_events_artifacts", lambda client, rows: captured.setdefault("events", rows))
    monkeypatch.setattr(engine, "_plot_equity_with_events", lambda analysis, plans: "wf-events-figure")

    run_info = engine._log_full_run_to_mlflow(
        analysis=FakeAnalysis(),
        analysis_results={"metrics": SimpleNamespace(total_return_pct=2.0), "report": "wf report"},
        plans=[],
        aggregate={"num_periods": 1, "wf_annualized_return": 3.0},
    )

    assert run_info == {"run_id": "run-123", "run_url": None}
    assert captured["tags"] == {"run_type": "walk_forward_window_winner"}
    assert captured["analysis_log_kwargs"]["start_new_run"] is False
    assert captured["analysis_log_kwargs"]["parameters"]["config.mode"] == "walk-forward"
    assert captured["analysis_log_kwargs"]["parameters"]["optimization_window_days"] == engine.optimization_window_days
    assert captured["analysis_log_kwargs"]["artifact_paths"] == ["cfg.yaml"]
    assert captured["analysis_log_kwargs"]["mlflow_client"].run_id == "run-123"
    assert captured["metrics"] == [{"num_periods": 1, "wf_annualized_return": 3.0}]
    assert ("window_summary.json", {"best_metric": 1.2}) in captured["json"]
    assert "walk_forward_equity_with_events" in captured["charts"]


def test_log_full_run_to_mlflow_keeps_run_when_artifact_logging_fails(monkeypatch):
    engine = _build_engine({"log_to_mlflow": True})
    captured = {"warnings": []}

    class FakeClient:
        enabled = True

        def __init__(self):
            self._run_id = None

        def start_run(self, run_name=None, description=None):
            self._run_id = "run-123"
            return self

        @property
        def run_id(self):
            return self._run_id

        def get_run_url(self):
            return "http://mlflow/run-123"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def log_metrics(self, metrics):
            captured["metrics"] = metrics

        def log_text(self, text, filename):
            raise RuntimeError("artifact failed")

        def log_chart(self, figure, filename, format="png", dpi=150):
            captured.setdefault("charts", []).append(filename)

    class FakeAnalysis:
        def log_to_mlflow(self, **kwargs):
            captured["analysis"] = True

        def plot_equity_curve(self, show=False):
            return "equity-figure"

    monkeypatch.setattr(engine, "_create_mlflow_client", lambda: FakeClient())
    monkeypatch.setattr(engine, "_build_optimization_events_rows", lambda plans: [])
    monkeypatch.setattr(engine, "_log_optimization_events_artifacts", lambda client, rows: captured.setdefault("events", True))
    monkeypatch.setattr(engine, "_plot_equity_with_events", lambda analysis, plans: "wf-events-figure")
    monkeypatch.setattr(wf_module.logger, "warning", lambda message: captured["warnings"].append(message))

    run_info = engine._log_full_run_to_mlflow(
        analysis=FakeAnalysis(),
        analysis_results={"metrics": SimpleNamespace(total_return_pct=2.0), "report": "wf report"},
        plans=[],
        aggregate={"num_periods": 1},
    )

    assert run_info == {"run_id": "run-123", "run_url": "http://mlflow/run-123"}
    assert captured["metrics"] == {"num_periods": 1}
    assert "equity_curve" in captured["charts"]
    assert "walk_forward_equity_with_events" in captured["charts"]
    assert any("report" in warning for warning in captured["warnings"])


def test_continuous_backtest_activates_all_plans_with_none_event_ids(monkeypatch):
    engine = _build_engine()
    ticks = [
        [SimpleNamespace(timestamp=datetime(2024, 1, 1), symbol="SPY", close=1.0)],
        [SimpleNamespace(timestamp=datetime(2024, 1, 2), symbol="SPY", close=1.0)],
        [SimpleNamespace(timestamp=datetime(2024, 1, 3), symbol="SPY", close=1.0)],
    ]
    engine.dp.iterate = lambda: iter(ticks)

    algo_patches = []
    pf_patches = []
    activated = []
    processed = []
    engine.al.reconfigure = lambda patch: algo_patches.append(dict(patch))
    engine.pf.reconfigure = lambda patch: pf_patches.append(dict(patch))
    monkeypatch.setattr(engine, "_update_optimization_event_activation", lambda event_id, timestamp: activated.append((event_id, timestamp)))
    monkeypatch.setattr(engine, "_process_tick", lambda tick, allow_trading: processed.append((tick[0].timestamp, allow_trading)))

    class FakeAnalysis:
        def __init__(self, pf, om):
            pass

        def run_full_analysis(self, **kwargs):
            return {
                "metrics": SimpleNamespace(
                    total_return_pct=1.0,
                    annualized_return=2.0,
                    sharpe_ratio=1.0,
                    max_drawdown_pct=-1.0,
                    total_trades=0,
                    final_equity=1000.0,
                )
            }

    monkeypatch.setattr(wf_module, "AnalysisEngine", FakeAnalysis)
    plans = [
        {
            "period_idx": 0,
            "event_id": None,
            "algo_patch": {"alpha": 1},
            "pf_patch": {"risk": 1},
            "adopted": True,
            "period": wf_module.WalkForwardPeriod(
                optimization_start=datetime(2023, 12, 1),
                optimization_end=datetime(2023, 12, 15),
                validation_start=datetime(2023, 12, 15),
                validation_end=datetime(2023, 12, 31),
                trading_start=datetime(2024, 1, 1),
                trading_end=datetime(2024, 1, 2),
            ),
        },
        {
            "period_idx": 1,
            "event_id": None,
            "algo_patch": {"alpha": 2},
            "pf_patch": {"risk": 2},
            "adopted": True,
            "period": wf_module.WalkForwardPeriod(
                optimization_start=datetime(2023, 12, 2),
                optimization_end=datetime(2023, 12, 16),
                validation_start=datetime(2023, 12, 16),
                validation_end=datetime(2024, 1, 1),
                trading_start=datetime(2024, 1, 2),
                trading_end=datetime(2024, 1, 3),
            ),
        },
    ]

    engine._run_continuous_backtest(plans)

    assert algo_patches == [{"alpha": 1}, {"alpha": 2}]
    assert pf_patches == [{"risk": 1}, {"risk": 2}]
    assert activated == [(None, datetime(2024, 1, 1)), (None, datetime(2024, 1, 2))]
    assert processed == [
        (datetime(2024, 1, 1), True),
        (datetime(2024, 1, 2), True),
        (datetime(2024, 1, 3), False),
    ]


def test_continuous_backtest_preapplies_first_config_for_warmup(monkeypatch):
    engine = _build_engine()
    ticks = [
        [SimpleNamespace(timestamp=datetime(2023, 12, 31), symbol="SPY", close=1.0)],
        [SimpleNamespace(timestamp=datetime(2024, 1, 1), symbol="SPY", close=1.0)],
    ]
    engine.dp.iterate = lambda: iter(ticks)
    active = {"alpha": 0}
    processed = []
    activation_updates = []

    def reconfigure_algo(patch):
        active.update(patch)

    engine.al.reconfigure = reconfigure_algo
    engine.pf.reconfigure = lambda patch: None
    monkeypatch.setattr(
        engine,
        "_update_optimization_event_activation",
        lambda event_id, timestamp: activation_updates.append((event_id, timestamp)),
    )
    monkeypatch.setattr(
        engine,
        "_process_tick",
        lambda tick, allow_trading: processed.append((tick[0].timestamp, allow_trading, active["alpha"])),
    )

    class FakeAnalysis:
        def __init__(self, pf, om):
            pass

        def run_full_analysis(self, **kwargs):
            return {
                "metrics": SimpleNamespace(
                    total_return_pct=1.0,
                    annualized_return=2.0,
                    sharpe_ratio=1.0,
                    max_drawdown_pct=-1.0,
                    total_trades=0,
                    final_equity=1000.0,
                )
            }

    monkeypatch.setattr(wf_module, "AnalysisEngine", FakeAnalysis)
    plans = [
        {
            "period_idx": 0,
            "event_id": "first-event",
            "algo_patch": {"alpha": 1},
            "pf_patch": {},
            "adopted": True,
            "period": wf_module.WalkForwardPeriod(
                optimization_start=datetime(2023, 12, 1),
                optimization_end=datetime(2023, 12, 15),
                validation_start=datetime(2023, 12, 15),
                validation_end=datetime(2023, 12, 31),
                trading_start=datetime(2024, 1, 1),
                trading_end=datetime(2024, 1, 2),
            ),
        }
    ]

    engine._run_continuous_backtest(plans)

    assert processed == [
        (datetime(2023, 12, 31), False, 1),
        (datetime(2024, 1, 1), True, 1),
    ]
    assert activation_updates == [("first-event", datetime(2024, 1, 1))]


def test_continuous_backtest_does_not_mark_rejected_event_activated(monkeypatch):
    engine = _build_engine()
    tick = [SimpleNamespace(timestamp=datetime(2024, 1, 1), symbol="SPY", close=1.0)]
    engine.dp.iterate = lambda: iter([tick])
    activation_updates = []

    monkeypatch.setattr(engine, "_update_optimization_event_activation", lambda event_id, timestamp: activation_updates.append((event_id, timestamp)))
    monkeypatch.setattr(engine, "_process_tick", lambda tick, allow_trading: None)

    class FakeAnalysis:
        def __init__(self, pf, om):
            pass

        def run_full_analysis(self, **kwargs):
            return {
                "metrics": SimpleNamespace(
                    total_return_pct=1.0,
                    annualized_return=2.0,
                    sharpe_ratio=1.0,
                    max_drawdown_pct=-1.0,
                    total_trades=0,
                    final_equity=1000.0,
                )
            }

    monkeypatch.setattr(wf_module, "AnalysisEngine", FakeAnalysis)
    plans = [
        {
            "period_idx": 0,
            "event_id": "rejected-event",
            "algo_patch": {},
            "pf_patch": {},
            "adopted": False,
            "period": wf_module.WalkForwardPeriod(
                optimization_start=datetime(2023, 12, 1),
                optimization_end=datetime(2023, 12, 15),
                validation_start=datetime(2023, 12, 15),
                validation_end=datetime(2023, 12, 31),
                trading_start=datetime(2024, 1, 1),
                trading_end=datetime(2024, 1, 2),
            ),
        }
    ]

    engine._run_continuous_backtest(plans)

    assert activation_updates == []


def test_metric_value_raises_for_unknown_metric():
    metrics = SimpleNamespace(annualized_return=1.2)

    with pytest.raises(ValueError, match="Unknown objective_metric 'annulized_return'"):
        metric_value(metrics, "annulized_return")


def test_run_backtest_core_preserves_history_length(monkeypatch):
    captured = {}

    class HistoryAlgorithm:
        def __init__(self, cfg, history_length: int = 0):
            captured["cfg"] = dict(cfg)
            captured["history_length"] = history_length

    class FakeDP:
        def __init__(self, cfg):
            self.cfg = cfg

    class FakeEngine:
        def __init__(self, cfg, dp, al, om, pf):
            self.pf = pf

        def run(self):
            return None

    class FakeAnalysisEngine:
        def __init__(self, pf, om):
            pass

        def run_full_analysis(self, **kwargs):
            return {"metrics": SimpleNamespace(annualized_return=1.0, total_return_pct=1.0)}

    monkeypatch.setattr(run_backtest_ray, "BacktestingEngine", FakeEngine)
    monkeypatch.setattr(run_backtest_ray, "AnalysisEngine", FakeAnalysisEngine)

    run_backtest_ray.run_backtest_core(
        backtest_cfg={"experiment_name": "exp", "starting_cash": 1000.0, "run_name": "run", "description": "desc"},
        alg_cfg={"history_length": 11, "alpha": 2},
        pf_cfg={"symbol": "SPY", "cash": 1000.0, "keep_history": True},
        dp_cfg={"path": "ignored"},
        algorithm_class=HistoryAlgorithm,
        portfolio_class=DummyPortfolio,
        data_provider_class=FakeDP,
        order_manager_class=DummyOrderManager,
    )

    assert captured["history_length"] == 11
    assert "history_length" not in captured["cfg"]
    assert captured["cfg"]["alpha"] == 2


def test_algorithm_base_reconfigure_syncs_matching_nested_attrs():
    algo = DummyNestedConfigAlgorithm({"alpha": 1, "nested": {"beta": 2}}, history_length=5)

    algo.reconfigure({"alpha": 7, "nested": {"beta": 9}})

    assert algo.cfg == {"history_length": 0, "full_history": False, "alpha": 7, "nested": {"beta": 9}}
    assert algo.alpha == 7
    assert algo.beta == 9
    assert algo.history_length == 5


def test_backtest_objective_fn_supports_nested_algorithm_keys(monkeypatch):
    captured = {}

    def fake_run_backtest_local(
        backtest_cfg,
        alg_cfg,
        pf_cfg,
        dp_cfg,
        warmup_dp_cfg=None,
        algorithm_class=None,
        portfolio_class=None,
        data_provider_class=None,
        order_manager_class=None,
        log_to_mlflow=True,
    ):
        captured["alg_cfg"] = alg_cfg
        captured["pf_cfg"] = pf_cfg
        captured["warmup_dp_cfg"] = warmup_dp_cfg
        captured["log_to_mlflow"] = log_to_mlflow
        return {"metrics": SimpleNamespace(annualized_return=1.23)}

    monkeypatch.setattr(run_backtest_ray, "run_backtest_local", fake_run_backtest_local)

    result = run_backtest_ray.backtest_objective_fn(
        config={"regime.alpha": 8, "stop_pct": 3.5},
        symbol="SPY",
        algorithm_class=DummyAlgorithm,
        portfolio_class=DummyPortfolio,
        data_provider_class=DummyDataProvider,
        order_manager_class=DummyOrderManager,
        base_algorithm_config={"regime": {"alpha": 1, "beta": 2}},
        base_portfolio_config={"stop_pct": 1.0, "profit_pct": 4.0},
        base_data_provider_config={"path": "ignored"},
        warmup_data_provider_config={"path": "warmup"},
        base_backtest_config={"run_name": "wf-run"},
        algorithm_param_keys=["regime.alpha"],
        portfolio_param_keys=["stop_pct"],
    )

    assert result == {"_metric": 1.23}
    assert captured["alg_cfg"] == {"regime": {"alpha": 8, "beta": 2}}
    assert captured["pf_cfg"] == {"stop_pct": 3.5, "profit_pct": 4.0}
    assert captured["warmup_dp_cfg"] == {"path": "warmup"}
    assert captured["log_to_mlflow"] is False


def test_tune_backtest_hyperparameters_disables_ray_sigint_handler_and_shuts_down(monkeypatch):
    env_before = "0"
    monkeypatch.setenv("TUNE_DISABLE_SIGINT_HANDLER", env_before)

    shutdown_calls = []
    captured_env = {}
    initial_sigint_handler = signal.getsignal(signal.SIGINT)
    sigbreak = getattr(signal, "SIGBREAK", None)
    initial_sigbreak_handler = signal.getsignal(sigbreak) if sigbreak is not None else None

    def fake_ray_init(**kwargs):
        captured_env["during_init"] = os.environ.get("TUNE_DISABLE_SIGINT_HANDLER")
        captured_env["sigint_during_init"] = signal.getsignal(signal.SIGINT)
        if sigbreak is not None:
            captured_env["sigbreak_during_init"] = signal.getsignal(sigbreak)

    monkeypatch.setattr(run_backtest_ray.ray, "init", fake_ray_init)
    monkeypatch.setattr(run_backtest_ray.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(run_backtest_ray.ray, "shutdown", lambda: shutdown_calls.append(True))
    monkeypatch.setattr(run_backtest_ray.tune, "with_parameters", lambda fn, **kwargs: fn)
    monkeypatch.setattr(run_backtest_ray, "OptunaSearch", lambda **kwargs: object())

    class FakeTuner:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self):
            captured_env["during_fit"] = os.environ.get("TUNE_DISABLE_SIGINT_HANDLER")
            captured_env["sigint_during_fit"] = signal.getsignal(signal.SIGINT)
            if sigbreak is not None:
                captured_env["sigbreak_during_fit"] = signal.getsignal(sigbreak)
            raise KeyboardInterrupt()

    monkeypatch.setattr(run_backtest_ray.tune, "Tuner", FakeTuner)

    with pytest.raises(KeyboardInterrupt):
        run_backtest_ray.tune_backtest_hyperparameters(
            symbol="SPY",
            algorithm_class=DummyAlgorithm,
            portfolio_class=DummyPortfolio,
            data_provider_class=DummyDataProvider,
            order_manager_class=DummyOrderManager,
            base_algorithm_config={},
            base_portfolio_config={},
            base_data_provider_config={},
            base_backtest_config={},
            search_space={},
            algorithm_param_keys=[],
            portfolio_param_keys=[],
            num_samples=1,
            max_concurrent_trials=1,
        )

    assert captured_env["during_init"] == "1"
    assert captured_env["during_fit"] == "1"
    assert captured_env["sigint_during_init"] == signal.default_int_handler
    assert captured_env["sigint_during_fit"] == signal.default_int_handler
    assert signal.getsignal(signal.SIGINT) == initial_sigint_handler
    if sigbreak is not None:
        assert captured_env["sigbreak_during_init"] == signal.default_int_handler
        assert captured_env["sigbreak_during_fit"] == signal.default_int_handler
        assert signal.getsignal(sigbreak) == initial_sigbreak_handler
    assert os.environ["TUNE_DISABLE_SIGINT_HANDLER"] == env_before
    assert shutdown_calls == [True]


def test_tune_backtest_hyperparameters_ignores_ray_shutdown_failure(monkeypatch):
    monkeypatch.setattr(run_backtest_ray.ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(run_backtest_ray.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(run_backtest_ray.tune, "with_parameters", lambda fn, **kwargs: fn)
    monkeypatch.setattr(run_backtest_ray, "OptunaSearch", lambda **kwargs: object())

    def bad_shutdown():
        raise RuntimeError("ray cleanup failed")

    monkeypatch.setattr(run_backtest_ray.ray, "shutdown", bad_shutdown)

    class FakeTuner:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self):
            return [SimpleNamespace(metrics={"_metric": 1.0}, config={"stop_pct": 3.5})]

    monkeypatch.setattr(run_backtest_ray.tune, "Tuner", FakeTuner)

    best = run_backtest_ray.tune_backtest_hyperparameters(
        symbol="SPY",
        algorithm_class=DummyAlgorithm,
        portfolio_class=DummyPortfolio,
        data_provider_class=DummyDataProvider,
        order_manager_class=DummyOrderManager,
        base_algorithm_config={},
        base_portfolio_config={},
        base_data_provider_config={},
        base_backtest_config={},
        search_space={"stop_pct": run_backtest_ray.tune.uniform(1.0, 5.0)},
        algorithm_param_keys=[],
        portfolio_param_keys=["stop_pct"],
        num_samples=1,
        max_concurrent_trials=1,
    )

    assert best == {"stop_pct": 3.5}


def test_tune_backtest_hyperparameters_seeds_first_trial_from_base_configs(monkeypatch):
    captured = {}

    monkeypatch.setattr(run_backtest_ray.ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(run_backtest_ray.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(run_backtest_ray.ray, "shutdown", lambda: None)
    monkeypatch.setattr(run_backtest_ray.tune, "with_parameters", lambda fn, **kwargs: fn)

    def fake_optuna_search(**kwargs):
        captured["optuna_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(run_backtest_ray, "OptunaSearch", fake_optuna_search)

    class FakeTuner:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self):
            return [SimpleNamespace(metrics={"_metric": 1.0}, config={"regime.alpha": 7, "stop_pct": 3.5})]

    monkeypatch.setattr(run_backtest_ray.tune, "Tuner", FakeTuner)

    best = run_backtest_ray.tune_backtest_hyperparameters(
        symbol="SPY",
        algorithm_class=DummyAlgorithm,
        portfolio_class=DummyPortfolio,
        data_provider_class=DummyDataProvider,
        order_manager_class=DummyOrderManager,
        base_algorithm_config={"regime": {"alpha": 7, "beta": 2}},
        base_portfolio_config={"stop_pct": 3.5, "profit_pct": 4.0},
        base_data_provider_config={},
        base_backtest_config={},
        search_space={
            "regime.alpha": run_backtest_ray.tune.randint(1, 10),
            "stop_pct": run_backtest_ray.tune.uniform(1.0, 5.0),
        },
        algorithm_param_keys=["regime.alpha"],
        portfolio_param_keys=["stop_pct"],
        num_samples=2,
        max_concurrent_trials=1,
    )

    assert best == {"regime.alpha": 7, "stop_pct": 3.5}
    assert captured["optuna_kwargs"]["points_to_evaluate"] == [{"regime.alpha": 7, "stop_pct": 3.5}]


def test_tune_backtest_hyperparameters_rejects_seed_outside_search_space():
    with pytest.raises(ValueError, match="outside randint range"):
        run_backtest_ray.tune_backtest_hyperparameters(
            symbol="SPY",
            algorithm_class=DummyAlgorithm,
            portfolio_class=DummyPortfolio,
            data_provider_class=DummyDataProvider,
            order_manager_class=DummyOrderManager,
            base_algorithm_config={"regime": {"alpha": 10}},
            base_portfolio_config={},
            base_data_provider_config={},
            base_backtest_config={},
            search_space={"regime.alpha": run_backtest_ray.tune.randint(1, 10)},
            algorithm_param_keys=["regime.alpha"],
            portfolio_param_keys=[],
            num_samples=1,
            max_concurrent_trials=1,
        )
