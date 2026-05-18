from __future__ import annotations

import copy
import threading
from datetime import datetime, timedelta
from typing import Any

from trading.engines.base_engine import AsyncEngine
from trading.engines.walk_forward_policy import decide_walk_forward_adoption
from utils.logger import Logger
from utils.utils import apply_tunable_config, build_tunable_patch

logger = Logger().get_logger(__name__)

_SCHEDULE_DAYS = {
    "daily": 1,
    "weekly": 7,
    "monthly": 30,
}


class LiveWalkForwardEngine:
    """AsyncEngine wrapper that applies walk-forward validation before reconfiguration."""

    def __init__(self, inner_engine: AsyncEngine, optimization_cfg: dict):
        self.inner = inner_engine
        self.cfg = optimization_cfg

        self._wf_lock = threading.Lock()
        self._is_optimizing = False

        self.schedule = optimization_cfg.get("schedule", "weekly")
        self.optimization_window_days = optimization_cfg.get("optimization_window_days", 90)
        self.validation_window_days = optimization_cfg.get("validation_window_days", 20)
        self.trading_window_days = optimization_cfg.get("trading_window_days", _SCHEDULE_DAYS.get(self.schedule, 7))
        self.improvement_threshold_pct = optimization_cfg.get("improvement_threshold_pct", 5.0)
        self.min_validation_trades = optimization_cfg.get("min_validation_trades", 0)
        self.objective_metric = optimization_cfg.get("objective_metric", "annualized_return")
        self.num_trials = optimization_cfg.get("num_trials", 50)
        self.max_concurrent_trials = optimization_cfg.get("max_concurrent_trials", 8)
        self.log_ray_worker_output = optimization_cfg.get("log_ray_worker_output", True)
        self.adoption_policy = optimization_cfg.get("adoption_policy", "next_optimization_boundary")

        self._search_space_cfg = optimization_cfg.get("search_space", {})
        self.algorithm_param_keys = optimization_cfg.get("algorithm_param_keys", [])
        self.portfolio_param_keys = optimization_cfg.get("portfolio_param_keys", [])
        self._hist_dp_cfg = optimization_cfg.get("historical_data_provider", {})

        self._current_al_params = copy.deepcopy(self.inner.al.cfg) if self.inner.al else {}
        self._current_pf_params = copy.deepcopy(self.inner.pf.cfg) if self.inner.pf else {}
        self._pending_algo_params: dict[str, Any] | None = None
        self._pending_pf_params: dict[str, Any] | None = None
        self._pending_event_id: str | None = None

        schedule_days = _SCHEDULE_DAYS.get(self.schedule, 7)
        self._schedule_interval = timedelta(days=schedule_days)
        self._next_optimization_time = datetime.now() + self._schedule_interval

        self._original_on_tick = self.inner.on_tick
        self.inner.on_tick = self._wrapped_on_tick

    def _wrapped_on_tick(self, tick):
        self._maybe_apply_pending_params()
        now = datetime.now()
        if not self._is_optimizing and now >= self._next_optimization_time:
            self._start_optimization()
        with self._wf_lock:
            return self._original_on_tick(tick)

    def _maybe_apply_pending_params(self) -> None:
        if self._pending_algo_params is None and self._pending_pf_params is None:
            return
        can_apply = self.adoption_policy == "immediate"
        if self.adoption_policy == "next_optimization_boundary" and datetime.now() >= self._next_optimization_time:
            can_apply = True
        if self.adoption_policy == "when_flat" and getattr(self.inner.pf, "positions", {}):
            can_apply = False
        elif self.adoption_policy == "when_flat":
            can_apply = True
        if not can_apply:
            return

        with self._wf_lock:
            if self._pending_algo_params:
                self.inner.al.reconfigure(self._pending_algo_params)
                self._current_al_params.update(self._pending_algo_params)
            if self._pending_pf_params:
                self.inner.pf.reconfigure(self._pending_pf_params)
                self._current_pf_params.update(self._pending_pf_params)

        if self.inner.state_store and self.inner.session_id and self._pending_event_id:
            self.inner.state_store.update_optimization_event(
                self.inner.session_id,
                self._pending_event_id,
                {
                    "status": "activated",
                    "activation_time": datetime.now(),
                },
            )
        self._pending_algo_params = None
        self._pending_pf_params = None
        self._pending_event_id = None

    def _start_optimization(self):
        self._is_optimizing = True
        thread = threading.Thread(target=self._run_optimization, name="walk-forward-optimization", daemon=True)
        thread.start()

    def _create_historical_dp(self, *, start_date: datetime, end_date: datetime):
        from trading.config.component_loader import instantiate_component
        from trading.config.models import ComponentConfig

        provider_path = self._hist_dp_cfg.get("provider")
        if not provider_path:
            raise ValueError("No historical_data_provider.provider configured")
        dp_cfg = copy.deepcopy(self._hist_dp_cfg)
        dp_cfg.pop("provider", None)
        dp_cfg["start_date"] = start_date.strftime("%Y-%m-%d")
        dp_cfg["end_date"] = end_date.strftime("%Y-%m-%d")
        component = ComponentConfig(implementation=provider_path, params=dp_cfg)
        return instantiate_component(component, cfg=dp_cfg)

    def _run_validation_backtest(self, al_cfg: dict, pf_cfg: dict, dp):
        from trading.analysis.analysis_engine import AnalysisEngine
        from trading.core.om.backtesting_om import BacktestingOrderManager
        from trading.engines.backtest_engine import BacktestingEngine

        al_class = type(self.inner.al)
        pf_class = type(self.inner.pf)
        om = BacktestingOrderManager()
        al = al_class(copy.deepcopy(al_cfg), history_length=getattr(self.inner.al, "history_length", 0))
        pf = pf_class(copy.deepcopy(pf_cfg), om, self.inner.pf.cash if self.inner.pf else 10000.0, {}, False)
        engine = BacktestingEngine({}, dp, al, om, pf)
        engine.run()
        analysis = AnalysisEngine(pf, om)
        return analysis.calculate_metrics()

    def _run_optimization(self):
        try:
            from trading.launchers.run_backtest_ray import tune_backtest_hyperparameters
            from utils.utils import parse_search_space

            end_date = datetime.now()
            validation_start = end_date - timedelta(days=self.validation_window_days)
            optimization_start = validation_start - timedelta(days=self.optimization_window_days)

            opt_dp = self._create_historical_dp(start_date=optimization_start, end_date=validation_start)

            symbol = self._current_pf_params.get("symbol", self._current_pf_params.get("upro_symbol", ""))
            best_config = tune_backtest_hyperparameters(
                symbol=symbol,
                algorithm_class=type(self.inner.al),
                portfolio_class=type(self.inner.pf),
                data_provider_class=type(opt_dp),
                order_manager_class=__import__("trading.core.om.backtesting_om", fromlist=["BacktestingOrderManager"]).BacktestingOrderManager,
                base_algorithm_config=copy.deepcopy(self._current_al_params),
                base_portfolio_config=copy.deepcopy(self._current_pf_params),
                base_data_provider_config=copy.deepcopy(opt_dp.cfg),
                base_backtest_config={
                    "symbol": symbol,
                    "run_name": "live_walk_forward",
                    "description": "live walk-forward optimization",
                    "starting_cash": self.inner.pf.cash if self.inner.pf else 10000.0,
                    "experiment_name": "Live Walk Forward",
                },
                search_space=parse_search_space(self._search_space_cfg),
                algorithm_param_keys=self.algorithm_param_keys,
                portfolio_param_keys=self.portfolio_param_keys,
                num_samples=self.num_trials,
                max_concurrent_trials=self.max_concurrent_trials,
                log_to_mlflow=False,
                log_ray_worker_output=self.log_ray_worker_output,
            )

            challenger_al_cfg = apply_tunable_config(
                self._current_al_params, best_config, self.algorithm_param_keys
            )
            challenger_pf_cfg = apply_tunable_config(
                self._current_pf_params, best_config, self.portfolio_param_keys
            )

            validation_dp_incumbent = self._create_historical_dp(start_date=validation_start, end_date=end_date)
            incumbent_metrics = self._run_validation_backtest(
                self._current_al_params, self._current_pf_params, validation_dp_incumbent
            )
            validation_dp_challenger = self._create_historical_dp(start_date=validation_start, end_date=end_date)
            challenger_metrics = self._run_validation_backtest(
                challenger_al_cfg, challenger_pf_cfg, validation_dp_challenger
            )

            decision = decide_walk_forward_adoption(
                incumbent_metrics,
                challenger_metrics,
                objective_metric=self.objective_metric,
                improvement_threshold_pct=self.improvement_threshold_pct,
                min_validation_trades=self.min_validation_trades,
            )
            event_id = self._persist_event(
                best_config=best_config,
                challenger_al_cfg=challenger_al_cfg,
                challenger_pf_cfg=challenger_pf_cfg,
                incumbent_metrics=incumbent_metrics,
                challenger_metrics=challenger_metrics,
                validation_start=validation_start,
                validation_end=end_date,
                optimization_start=optimization_start,
                optimization_end=validation_start,
                decision=decision,
            )

            if decision.adopted:
                self._pending_algo_params = build_tunable_patch(best_config, self.algorithm_param_keys)
                self._pending_pf_params = build_tunable_patch(best_config, self.portfolio_param_keys)
                self._pending_event_id = event_id
                if self.adoption_policy == "immediate":
                    self._maybe_apply_pending_params()
            elif self.inner.state_store and self.inner.session_id and event_id:
                self.inner.state_store.update_optimization_event(
                    self.inner.session_id,
                    event_id,
                    {"status": "rejected"},
                )
        except Exception:
            logger.exception("Live walk-forward optimization failed")
        finally:
            self._is_optimizing = False
            self._next_optimization_time = datetime.now() + self._schedule_interval

    def _persist_event(
        self,
        *,
        best_config: dict,
        challenger_al_cfg: dict,
        challenger_pf_cfg: dict,
        incumbent_metrics,
        challenger_metrics,
        optimization_start: datetime,
        optimization_end: datetime,
        validation_start: datetime,
        validation_end: datetime,
        decision,
    ) -> str | None:
        if not self.inner.state_store or not self.inner.session_id:
            return None
        event = {
            "event_type": "walk_forward_live",
            "optimization_start": optimization_start,
            "optimization_end": optimization_end,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "next_trading_start": datetime.now(),
            "next_trading_end": datetime.now() + timedelta(days=self.trading_window_days),
            "objective_metric": self.objective_metric,
            "improvement_threshold_pct": self.improvement_threshold_pct,
            "min_validation_trades": self.min_validation_trades,
            "adoption_policy": self.adoption_policy,
            "incumbent_algorithm_params": copy.deepcopy(self._current_al_params),
            "incumbent_portfolio_params": copy.deepcopy(self._current_pf_params),
            "challenger_algorithm_params": challenger_al_cfg,
            "challenger_portfolio_params": challenger_pf_cfg,
            "best_config_raw": best_config,
            "incumbent_metrics": incumbent_metrics.__dict__,
            "challenger_metrics": challenger_metrics.__dict__,
            "status": "pending_activation" if decision.adopted else "evaluated",
            **decision.as_dict(),
        }
        return self.inner.state_store.save_optimization_event(self.inner.session_id, event)

    def run(self):
        self.inner.run()

    async def start(self):
        await self.inner.start()

    async def stop(self):
        await self.inner.stop()
