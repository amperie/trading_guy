"""
Self-optimizing live trading engine.

Wraps any AsyncEngine (e.g. AlpacaRealTimeEngine) and adds periodic
background HPO. The inner engine continues trading while optimization
runs in a background thread. When HPO finds parameters that improve
on the current config by more than a configurable threshold, the
algorithm and portfolio are hot-swapped via reconfigure().

Configuration (in the engine config dict under 'optimization'):

    optimization:
      enabled: true
      schedule: daily           # daily | weekly | monthly
      rolling_window_days: 90
      improvement_threshold_pct: 5.0
      num_trials: 50
      max_concurrent_trials: 8
      search_space:
        macd_fast_period:  { type: randint, low: 5, high: 50 }
        stop_pct:          { type: uniform, low: 1.0, high: 20.0 }
      algorithm_param_keys: [macd_fast_period]
      portfolio_param_keys: [stop_pct]
      historical_data_provider:
        provider: "trading.data_providers.alpaca_data_provider.AlpacaDataProvider"
        symbols: ["SPY", "UPRO", "SPXU"]
        timeframe: "5Min"
"""
import copy
import threading
from datetime import datetime, timedelta

from trading.engines.base_engine import AsyncEngine
from trading.core.classes import PriceData, TickResults
from utils.logger import Logger
from utils.performance_tracker import PerformanceTracker
from utils.utils import apply_tunable_config, build_tunable_patch

logger = Logger().get_logger(__name__)

# Schedule intervals in days
_SCHEDULE_DAYS = {
    "daily": 1,
    "weekly": 7,
    "monthly": 30,
}


class SelfOptimizingLiveEngine:
    """Wraps an AsyncEngine and adds periodic background HPO.

    The inner engine keeps trading on its current parameters while
    optimization runs in a separate thread. When optimization
    completes and finds an improvement exceeding the threshold,
    the algorithm and portfolio are reconfigured in-place.

    Thread safety:
        A threading.Lock (_hpo_lock) is held briefly during
        reconfiguration. on_tick acquires the same lock to ensure
        no tick is processed mid-reconfigure.

    Args:
        inner_engine: A fully configured AsyncEngine instance.
        optimization_cfg: Dict with optimization settings (see module docstring).
    """

    def __init__(self, inner_engine: AsyncEngine, optimization_cfg: dict):
        self.inner = inner_engine
        self.cfg = optimization_cfg

        self._hpo_lock = threading.Lock()
        self._is_optimizing = False

        # Parse config
        self.schedule = optimization_cfg.get("schedule", "daily")
        self.rolling_window_days = optimization_cfg.get("rolling_window_days", 90)
        self.improvement_threshold_pct = optimization_cfg.get("improvement_threshold_pct", 5.0)
        self.num_trials = optimization_cfg.get("num_trials", 50)
        self.max_concurrent_trials = optimization_cfg.get("max_concurrent_trials", 8)
        self.log_ray_worker_output = optimization_cfg.get("log_ray_worker_output", True)

        self._search_space_cfg = optimization_cfg.get("search_space", {})
        self.algorithm_param_keys = optimization_cfg.get("algorithm_param_keys", [])
        self.portfolio_param_keys = optimization_cfg.get("portfolio_param_keys", [])
        self._hist_dp_cfg = optimization_cfg.get("historical_data_provider", {})

        # Performance tracking
        self._performance_tracker = PerformanceTracker(self.rolling_window_days)

        # Current active params (initialized from inner engine)
        self._current_al_params = copy.deepcopy(self.inner.al.cfg) if self.inner.al else {}
        self._current_pf_params = copy.deepcopy(self.inner.pf.cfg) if self.inner.pf else {}

        # Scheduling
        schedule_days = _SCHEDULE_DAYS.get(self.schedule, 1)
        self._next_optimization_time = datetime.now() + timedelta(days=schedule_days)
        self._schedule_interval = timedelta(days=schedule_days)

        # Monkey-patch inner engine's on_tick to go through our wrapper
        self._original_on_tick = self.inner.on_tick
        self.inner.on_tick = self._wrapped_on_tick

        logger.info(
            f"SelfOptimizingLiveEngine initialized: schedule={self.schedule}, "
            f"window={self.rolling_window_days}d, threshold={self.improvement_threshold_pct}%"
        )

    def _wrapped_on_tick(self, tick: list[PriceData]) -> TickResults:
        """Wrap on_tick with lock and optimization check."""
        # Track performance
        if tick and self.inner.pf:
            self._performance_tracker.update(
                tick[0].timestamp, self.inner.pf.total_value
            )

        # Check if optimization is due
        now = datetime.now()
        if not self._is_optimizing and now >= self._next_optimization_time:
            self._start_optimization()

        # Process tick with lock to prevent mid-reconfigure processing
        with self._hpo_lock:
            return self._original_on_tick(tick)

    def _start_optimization(self):
        """Spawn background optimization thread."""
        self._is_optimizing = True
        logger.info("Starting background HPO optimization...")
        thread = threading.Thread(
            target=self._run_optimization,
            name="hpo-optimization",
            daemon=True,
        )
        thread.start()

    def _run_optimization(self):
        """Run HPO in background thread and optionally reconfigure."""
        try:
            from utils.utils import parse_search_space
            from trading.launchers.run_backtest_ray import tune_backtest_hyperparameters
            from trading.engines.backtest_engine import BacktestingEngine
            from trading.analysis.analysis_engine import AnalysisEngine
            from trading.core.om.backtesting_om import BacktestingOrderManager

            # 1. Create historical data provider for rolling window
            hist_dp = self._create_historical_dp()
            if hist_dp is None:
                logger.error("Could not create historical data provider for HPO")
                return

            # 2. Run HPO
            search_space = parse_search_space(self._search_space_cfg)

            al_class = type(self.inner.al)
            pf_class = type(self.inner.pf)
            dp_class = type(hist_dp)
            om_class = BacktestingOrderManager

            symbol = self._current_pf_params.get(
                "symbol",
                self._current_pf_params.get("upro_symbol", ""),
            )
            base_backtest_cfg = {
                "symbol": symbol,
                "run_name": "HPO_optimization",
                "description": "Background HPO",
                "starting_cash": self.inner.pf.cash if self.inner.pf else 10000.0,
                "experiment_name": "Live Self-Optimization",
            }

            best_config = tune_backtest_hyperparameters(
                symbol=symbol,
                algorithm_class=al_class,
                portfolio_class=pf_class,
                data_provider_class=dp_class,
                order_manager_class=om_class,
                base_algorithm_config=copy.deepcopy(self._current_al_params),
                base_portfolio_config=copy.deepcopy(self._current_pf_params),
                base_data_provider_config=copy.deepcopy(hist_dp.cfg),
                base_backtest_config=base_backtest_cfg,
                search_space=search_space,
                algorithm_param_keys=self.algorithm_param_keys,
                portfolio_param_keys=self.portfolio_param_keys,
                num_samples=self.num_trials,
                max_concurrent_trials=self.max_concurrent_trials,
                log_to_mlflow=False,
                log_ray_worker_output=self.log_ray_worker_output,
            )

            logger.info(f"HPO complete. Best config: {best_config}")

            # 3. Run baseline with current params on same data
            hist_dp_baseline = self._create_historical_dp()
            om = BacktestingOrderManager()
            al = al_class(copy.deepcopy(self._current_al_params))
            starting_cash = self.inner.pf.cash if self.inner.pf else 10000.0
            pf = pf_class(copy.deepcopy(self._current_pf_params), om, starting_cash, {}, False)
            engine = BacktestingEngine({}, hist_dp_baseline, al, om, pf)
            engine.run()
            current_metric = 0.0
            if pf.total_value > 0 and starting_cash > 0:
                current_metric = (pf.total_value - starting_cash) / starting_cash

            # 4. Run best config on same data
            hist_dp_best = self._create_historical_dp()
            best_al_cfg = apply_tunable_config(
                self._current_al_params, best_config, self.algorithm_param_keys
            )
            best_pf_cfg = apply_tunable_config(
                self._current_pf_params, best_config, self.portfolio_param_keys
            )

            om2 = BacktestingOrderManager()
            al2 = al_class(best_al_cfg)
            pf2 = pf_class(best_pf_cfg, om2, starting_cash, {}, False)
            engine2 = BacktestingEngine({}, hist_dp_best, al2, om2, pf2)
            engine2.run()
            best_metric = 0.0
            if pf2.total_value > 0 and starting_cash > 0:
                best_metric = (pf2.total_value - starting_cash) / starting_cash

            # 5. Compare and decide
            if abs(current_metric) > 0:
                improvement_pct = ((best_metric - current_metric) / abs(current_metric)) * 100
            else:
                improvement_pct = 100.0 if best_metric > 0 else 0.0

            logger.info(
                f"HPO comparison: current={current_metric:.4f}, best={best_metric:.4f}, "
                f"improvement={improvement_pct:.2f}%"
            )

            if improvement_pct >= self.improvement_threshold_pct:
                logger.info("Reconfiguring algorithm and portfolio with HPO-best params")
                with self._hpo_lock:
                    # Extract algo params
                    algo_params = build_tunable_patch(best_config, self.algorithm_param_keys)
                    pf_params = build_tunable_patch(best_config, self.portfolio_param_keys)

                    if algo_params:
                        self.inner.al.reconfigure(algo_params)
                    if pf_params:
                        self.inner.pf.reconfigure(pf_params)

                    self._current_al_params.update(algo_params)
                    self._current_pf_params.update(pf_params)

                self._log_reconfiguration(best_config, current_metric, best_metric, True)
            else:
                logger.info("HPO did not exceed threshold — keeping current params")
                self._log_reconfiguration(best_config, current_metric, best_metric, False)

        except Exception:
            logger.exception("Background HPO optimization failed")
        finally:
            self._is_optimizing = False
            self._next_optimization_time = datetime.now() + self._schedule_interval
            logger.info(f"Next optimization scheduled for {self._next_optimization_time}")

    def _create_historical_dp(self):
        """Create a DataProvider for the rolling historical window."""
        try:
            provider_path = self._hist_dp_cfg.get("provider")
            if not provider_path:
                logger.error("No historical_data_provider.provider configured")
                return None

            dp_cfg = copy.deepcopy(self._hist_dp_cfg)
            dp_cfg.pop("provider", None)

            # Set date range for rolling window
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.rolling_window_days)
            dp_cfg["start_date"] = start_date.strftime("%Y-%m-%d")
            dp_cfg["end_date"] = end_date.strftime("%Y-%m-%d")

            import importlib
            module_path, class_name = provider_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            dp_class = getattr(module, class_name)
            return dp_class(dp_cfg)
        except Exception:
            logger.exception("Failed to create historical data provider")
            return None

    def _log_reconfiguration(self, best_config, current_metric, best_metric, reconfigured):
        """Log HPO results to MLflow."""
        try:
            from utils.mlflow_client import MLflowClient
            client = MLflowClient.from_config()
            if not client:
                return

            with client.start_run(
                run_name=f"HPO_{datetime.now().strftime('%Y%m%d_%H%M')}",
                description=f"Background HPO — {'reconfigured' if reconfigured else 'kept current'}",
            ):
                client.log_params(best_config)
                client.log_metrics({
                    "current_metric": current_metric,
                    "best_metric": best_metric,
                    "improvement_pct": ((best_metric - current_metric) / abs(current_metric) * 100)
                        if abs(current_metric) > 0 else 0,
                })
                client.set_tag("reconfigured", str(reconfigured))
        except Exception:
            logger.exception("Failed to log HPO results to MLflow")

    def run(self):
        """Start the inner engine (blocking)."""
        self.inner.run()

    async def start(self):
        """Start the inner engine (async)."""
        await self.inner.start()

    async def stop(self):
        """Stop the inner engine."""
        await self.inner.stop()
