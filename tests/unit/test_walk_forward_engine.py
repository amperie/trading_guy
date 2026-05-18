from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from trading.engines import walk_forward_engine as wf_module
from trading.engines.walk_forward_engine import WalkForwardEngine
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
    for period in periods:
        assert period.validation_start > period.optimization_end
        assert period.trading_start > period.validation_end
    for previous, current in zip(periods, periods[1:]):
        assert current.optimization_start == previous.trading_start


def test_evaluate_period_compares_with_portfolio_params(monkeypatch):
    engine = _build_engine()
    captured = []

    monkeypatch.setattr(engine, "_create_dp_for_range", lambda start, end: f"{start.date()}->{end.date()}")
    monkeypatch.setattr(engine, "_run_optimization", lambda dp: {"alpha": 2, "risk": 9})

    def fake_run_backtest(al_cfg, pf_cfg, dp):
        captured.append((dict(al_cfg), dict(pf_cfg), dp))
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

    result = engine._evaluate_period(1, period, current_al_cfg, current_pf_cfg)

    assert captured[0][1]["risk"] == 1
    assert captured[1][1]["risk"] == 9
    assert result["adopted"] is True
    assert result["pf_cfg"]["risk"] == 9
    assert result["decision"]["challenger_metric"] == 25.0


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
        algorithm_class,
        portfolio_class,
        data_provider_class,
        order_manager_class,
        log_to_mlflow=True,
    ):
        captured["alg_cfg"] = alg_cfg
        captured["pf_cfg"] = pf_cfg
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
        base_backtest_config={"run_name": "wf-run"},
        algorithm_param_keys=["regime.alpha"],
        portfolio_param_keys=["stop_pct"],
    )

    assert result == {"_metric": 1.23}
    assert captured["alg_cfg"] == {"regime": {"alpha": 8, "beta": 2}}
    assert captured["pf_cfg"] == {"stop_pct": 3.5, "profit_pct": 4.0}
    assert captured["log_to_mlflow"] is False
