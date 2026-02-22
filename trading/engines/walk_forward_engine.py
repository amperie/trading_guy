"""
Walk-forward backtesting engine.

Splits historical data into rolling optimization + out-of-sample trading
windows, re-optimizing parameters at each boundary using Ray Tune HPO.

Data timeline example (1 year, 90-day opt window, 30-day trading window):

    |--- opt 90d ---|-- trade 30d --|
                    |--- opt 90d ---|-- trade 30d --|
                                    |--- opt 90d ---|-- trade 30d --|
                                                    ...
"""
import copy
from datetime import datetime, timedelta
from dataclasses import asdict
from typing import Type

import pandas as pd

from trading.engines.base_engine import BaseEngine
from trading.engines.backtest_engine import BacktestingEngine
from trading.analysis.analysis_engine import AnalysisEngine
from trading.core.algorithm import Algorithm
from trading.core.portfolio import Portfolio
from trading.core.om.order_manager import OrderManager
from trading.data_providers.data_provider import DataProvider
from utils.logger import Logger

logger = Logger().get_logger(__name__)


class WalkForwardEngine(BaseEngine):
    """
    Walk-forward backtesting engine that re-optimizes at each period boundary.

    For each period:
      1. Optimize: Run HPO on the optimization window.
      2. Baseline: Run a backtest with current params on the same window.
      3. Decide: Adopt new params if improvement exceeds threshold.
      4. Trade: Run out-of-sample backtest on the trading window.
      5. Log results to MLflow (nested runs under a parent).

    Args:
        cfg: Engine configuration dict containing walk_forward settings.
        dp: DataProvider instance (used as a template for its class and config).
        al: Algorithm instance (used as a template for its class and config).
        om: OrderManager instance (used as a template for its class).
        pf: Portfolio instance (used as a template for its class and config).
    """

    def __init__(
        self,
        cfg: dict = None,
        dp: DataProvider = None,
        al: Algorithm = None,
        om: OrderManager = None,
        pf: Portfolio = None,
    ):
        super().__init__(cfg=cfg, dp=dp, al=al, om=om, pf=pf)

        wf = cfg.get("walk_forward", {})
        self.optimization_window_days = wf.get("optimization_window_days", 90)
        self.trading_window_days = wf.get("trading_window_days", 30)
        self.improvement_threshold_pct = wf.get("improvement_threshold_pct", 5.0)
        self.num_trials = wf.get("num_trials", 50)
        self.max_concurrent_trials = wf.get("max_concurrent_trials", 8)

        # Search space config (parsed lazily when needed)
        self._search_space_cfg = wf.get("search_space", {})
        self.algorithm_param_keys = wf.get("algorithm_param_keys", [])
        self.portfolio_param_keys = wf.get("portfolio_param_keys", [])

        # MLflow config
        self.experiment_name = cfg.get("experiment_name", "Walk Forward Backtest")
        self.run_name = cfg.get("run_name", "WalkForward")
        self.description = cfg.get("description", "")

        # Store original configs for re-initialization
        self.original_dp_cfg = copy.deepcopy(dp.cfg) if dp else {}
        self.original_al_cfg = copy.deepcopy(al.cfg) if (al and hasattr(al, "cfg")) else {}
        self.original_pf_cfg = copy.deepcopy(pf.cfg) if (pf and hasattr(pf, "cfg")) else {}
        self.starting_cash = pf.cash if pf else 0.0

        # Component classes
        self._dp_class = type(dp) if dp else None
        self._al_class = type(al) if al else None
        self._om_class = type(om) if om else None
        self._pf_class = type(pf) if pf else None

        # Backtest config for HPO objective fn
        symbol = self.original_pf_cfg.get(
            "symbol",
            self.original_pf_cfg.get("upro_symbol", ""),
        )
        self._base_backtest_cfg = {
            "symbol": symbol,
            "run_name": self.run_name,
            "description": self.description,
            "starting_cash": self.starting_cash,
            "experiment_name": self.experiment_name,
        }

    def _get_date_range(self) -> tuple[datetime, datetime]:
        """Get the min/max dates from the data provider."""
        self.dp.load_data()
        df = self.dp.data
        timestamps = pd.to_datetime(df["timestamp"])
        return timestamps.min().to_pydatetime(), timestamps.max().to_pydatetime()

    def _compute_periods(self) -> list[dict]:
        """Compute optimization/trading period boundaries."""
        data_start, data_end = self._get_date_range()

        periods = []
        opt_start = data_start

        while True:
            opt_end = opt_start + timedelta(days=self.optimization_window_days)
            trade_start = opt_end
            trade_end = trade_start + timedelta(days=self.trading_window_days)

            if trade_end > data_end:
                # If we can fit at least some trading days, use remaining data
                if trade_start < data_end:
                    trade_end = data_end
                else:
                    break

            periods.append({
                "opt_start": opt_start,
                "opt_end": opt_end,
                "trade_start": trade_start,
                "trade_end": trade_end,
            })

            # Slide forward by trading_window_days
            opt_start = trade_start

        return periods

    def _parse_search_space(self) -> dict:
        """Parse search space config into Ray Tune distributions."""
        from utils.utils import parse_search_space
        return parse_search_space(self._search_space_cfg)

    def _create_dp_for_range(self, start: datetime, end: datetime) -> DataProvider:
        """Create a DataProvider filtered to a date range."""
        dp_cfg = copy.deepcopy(self.original_dp_cfg)
        dp_cfg["start_date"] = start.strftime("%Y-%m-%d")
        dp_cfg["end_date"] = end.strftime("%Y-%m-%d")
        return self._dp_class(dp_cfg)

    def _run_backtest(
        self, al_cfg: dict, dp: DataProvider
    ) -> dict:
        """Run a single backtest with given config and return analysis results."""
        om = self._om_class()
        al = self._al_class(al_cfg)
        pf = self._pf_class(
            copy.deepcopy(self.original_pf_cfg),
            om,
            self.starting_cash,
            {},
            True,
        )

        engine = BacktestingEngine({}, dp, al, om, pf)
        engine.run()

        analysis = AnalysisEngine(pf, om)
        trades = analysis.extract_trades()
        metrics = analysis.calculate_metrics()

        return {
            "metrics": metrics,
            "trades": trades,
            "portfolio": pf,
            "order_manager": om,
            "analysis_engine": analysis,
        }

    def _run_optimization(self, opt_dp: DataProvider) -> dict:
        """Run HPO on the optimization window and return best config."""
        from trading.launchers.run_backtest_ray import tune_backtest_hyperparameters

        search_space = self._parse_search_space()
        symbol = self._base_backtest_cfg["symbol"]

        best_config = tune_backtest_hyperparameters(
            symbol=symbol,
            algorithm_class=self._al_class,
            portfolio_class=self._pf_class,
            data_provider_class=self._dp_class,
            order_manager_class=self._om_class,
            base_algorithm_config=copy.deepcopy(self.original_al_cfg),
            base_portfolio_config=copy.deepcopy(self.original_pf_cfg),
            base_data_provider_config=copy.deepcopy(opt_dp.cfg),
            base_backtest_config=self._base_backtest_cfg,
            search_space=search_space,
            algorithm_param_keys=self.algorithm_param_keys,
            portfolio_param_keys=self.portfolio_param_keys,
            num_samples=self.num_trials,
            max_concurrent_trials=self.max_concurrent_trials,
        )

        return best_config

    def _apply_config(self, best_config: dict) -> dict:
        """Merge best HPO config into algorithm config.

        Returns the merged algorithm config dict.
        """
        al_cfg = copy.deepcopy(self.original_al_cfg)
        pf_cfg = copy.deepcopy(self.original_pf_cfg)

        for key in self.algorithm_param_keys:
            if key in best_config:
                al_cfg[key] = best_config[key]

        for key in self.portfolio_param_keys:
            if key in best_config:
                pf_cfg[key] = best_config[key]

        return al_cfg, pf_cfg

    def run(self):
        """Execute the walk-forward backtest across all periods.

        Returns:
            Dictionary with:
            - periods: List of per-period result dicts.
            - aggregate: Aggregated metrics across all periods.
        """
        periods = self._compute_periods()
        if not periods:
            logger.error("No valid periods could be computed from data range")
            return {"periods": [], "aggregate": {}}

        logger.info("=" * 80)
        logger.info("WALK-FORWARD BACKTEST")
        logger.info("=" * 80)
        logger.info(f"Optimization window: {self.optimization_window_days} days")
        logger.info(f"Trading window: {self.trading_window_days} days")
        logger.info(f"Number of periods: {len(periods)}")
        logger.info(f"HPO trials per period: {self.num_trials}")
        logger.info(f"Improvement threshold: {self.improvement_threshold_pct}%")
        logger.info("=" * 80)

        # Try to create MLflow parent run
        mlflow_client = self._create_mlflow_client()
        parent_run_ctx = None
        if mlflow_client:
            parent_run_ctx = mlflow_client.start_run(
                run_name=f"{self.run_name}_aggregate",
                description=f"Walk-forward aggregate: {len(periods)} periods",
            )

        period_results = []
        current_al_cfg = copy.deepcopy(self.original_al_cfg)
        current_pf_cfg = copy.deepcopy(self.original_pf_cfg)
        parent_run_id = None

        try:
            if mlflow_client and mlflow_client._run:
                parent_run_id = mlflow_client._run.info.run_id

            for i, period in enumerate(periods):
                logger.info("")
                logger.info(f"--- Period {i + 1}/{len(periods)} ---")
                logger.info(
                    f"Optimization: {period['opt_start'].date()} to {period['opt_end'].date()}"
                )
                logger.info(
                    f"Trading:      {period['trade_start'].date()} to {period['trade_end'].date()}"
                )

                result = self._run_period(
                    i, period, current_al_cfg, current_pf_cfg, mlflow_client, parent_run_id
                )
                period_results.append(result)

                # Update current config if new params were adopted
                if result.get("adopted"):
                    current_al_cfg = result["al_cfg"]
                    current_pf_cfg = result["pf_cfg"]

            # Aggregate results
            aggregate = self._aggregate_results(period_results)

            # Log aggregate to parent MLflow run
            if mlflow_client:
                mlflow_client.log_metrics(aggregate)
                mlflow_client.log_params({
                    "optimization_window_days": self.optimization_window_days,
                    "trading_window_days": self.trading_window_days,
                    "num_periods": len(periods),
                    "num_trials": self.num_trials,
                    "improvement_threshold_pct": self.improvement_threshold_pct,
                })

            logger.info("")
            logger.info("=" * 80)
            logger.info("WALK-FORWARD BACKTEST COMPLETE")
            logger.info("=" * 80)
            for key, val in aggregate.items():
                logger.info(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
            logger.info("=" * 80)

            return {"periods": period_results, "aggregate": aggregate}

        finally:
            if mlflow_client:
                mlflow_client.end_run()

    def _run_period(
        self,
        period_idx: int,
        period: dict,
        current_al_cfg: dict,
        current_pf_cfg: dict,
        mlflow_client,
        parent_run_id: str | None,
    ) -> dict:
        """Run a single walk-forward period: optimize, decide, trade.

        Returns:
            Dict with period results including metrics, config, adopted flag.
        """
        opt_start = period["opt_start"]
        opt_end = period["opt_end"]
        trade_start = period["trade_start"]
        trade_end = period["trade_end"]

        # 1. Create optimization-window data provider
        opt_dp = self._create_dp_for_range(opt_start, opt_end)

        # 2. Run HPO
        logger.info("Running HPO optimization...")
        best_config = self._run_optimization(opt_dp)
        best_al_cfg, best_pf_cfg = self._apply_config(best_config)
        logger.info(f"HPO best config: {best_config}")

        # 3. Run baseline with current params on opt window
        adopted = True
        if period_idx > 0:
            logger.info("Running baseline backtest with current params...")
            opt_dp_baseline = self._create_dp_for_range(opt_start, opt_end)
            baseline_result = self._run_backtest(current_al_cfg, opt_dp_baseline)
            current_metric = baseline_result["metrics"].annualized_return or 0.0

            logger.info("Running HPO-best backtest on opt window...")
            opt_dp_best = self._create_dp_for_range(opt_start, opt_end)
            best_result = self._run_backtest(best_al_cfg, opt_dp_best)
            best_metric = best_result["metrics"].annualized_return or 0.0

            # 4. Decide whether to adopt
            if abs(current_metric) > 0:
                improvement_pct = ((best_metric - current_metric) / abs(current_metric)) * 100
            else:
                improvement_pct = 100.0 if best_metric > 0 else 0.0

            logger.info(
                f"Current metric: {current_metric:.4f}, Best metric: {best_metric:.4f}, "
                f"Improvement: {improvement_pct:.2f}%"
            )

            if improvement_pct >= self.improvement_threshold_pct:
                logger.info("Adopting new parameters (improvement exceeds threshold)")
                adopted = True
            else:
                logger.info("Keeping current parameters (improvement below threshold)")
                adopted = False
        else:
            logger.info("First period — adopting HPO params unconditionally")

        # 5. Choose config for trading
        if adopted:
            trade_al_cfg = best_al_cfg
            trade_pf_cfg = best_pf_cfg
        else:
            trade_al_cfg = current_al_cfg
            trade_pf_cfg = current_pf_cfg

        # 6. Trade: run out-of-sample backtest
        logger.info("Running out-of-sample trading backtest...")
        trade_dp = self._create_dp_for_range(trade_start, trade_end)

        # Update pf_cfg with the chosen portfolio params
        pf_cfg_for_trade = copy.deepcopy(self.original_pf_cfg)
        for key in self.portfolio_param_keys:
            if key in trade_pf_cfg:
                pf_cfg_for_trade[key] = trade_pf_cfg[key]

        om = self._om_class()
        al = self._al_class(trade_al_cfg)
        pf = self._pf_class(pf_cfg_for_trade, om, self.starting_cash, {}, True)

        trade_engine = BacktestingEngine({}, trade_dp, al, om, pf)
        trade_engine.run()

        analysis = AnalysisEngine(pf, om)
        trades = analysis.extract_trades()
        metrics = analysis.calculate_metrics()

        logger.info(f"Period {period_idx + 1} results:")
        logger.info(f"  Total Return: {metrics.total_return_pct:.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"  Total Trades: {metrics.total_trades}")

        # 7. Log to MLflow as nested run
        if mlflow_client:
            self._log_period_to_mlflow(
                analysis, metrics, period_idx, period, trade_al_cfg,
                trade_pf_cfg, adopted, best_config, mlflow_client,
                parent_run_id,
            )

        return {
            "period_idx": period_idx,
            "opt_start": opt_start.isoformat(),
            "opt_end": opt_end.isoformat(),
            "trade_start": trade_start.isoformat(),
            "trade_end": trade_end.isoformat(),
            "metrics": metrics,
            "trades": trades,
            "adopted": adopted,
            "best_config": best_config,
            "al_cfg": trade_al_cfg,
            "pf_cfg": trade_pf_cfg,
        }

    def _log_period_to_mlflow(
        self, analysis, metrics, period_idx, period, al_cfg,
        pf_cfg, adopted, best_config, mlflow_client, parent_run_id,
    ):
        """Log a single period's results to MLflow as a nested run."""
        try:
            import mlflow

            trade_start = period["trade_start"].strftime("%Y-%m-%d")
            trade_end = period["trade_end"].strftime("%Y-%m-%d")

            with mlflow.start_run(
                run_name=f"Period_{period_idx + 1}_{trade_start}_{trade_end}",
                nested=True,
                parent_run_id=parent_run_id,
            ):
                metrics_dict = asdict(metrics)
                numeric_metrics = {
                    k: v for k, v in metrics_dict.items()
                    if isinstance(v, (int, float)) and v is not None
                }
                mlflow.log_metrics(numeric_metrics)

                mlflow.log_params({
                    "period_idx": period_idx,
                    "opt_start": period["opt_start"].strftime("%Y-%m-%d"),
                    "opt_end": period["opt_end"].strftime("%Y-%m-%d"),
                    "trade_start": trade_start,
                    "trade_end": trade_end,
                    "adopted": adopted,
                })

                # Log best config
                for k, v in best_config.items():
                    mlflow.log_param(f"hpo_{k}", v)

                # Log used config
                for k, v in al_cfg.items():
                    mlflow.log_param(f"alg_{k}", v)
                for k, v in pf_cfg.items():
                    mlflow.log_param(f"pf_{k}", v)

        except Exception:
            logger.exception(f"Failed to log period {period_idx + 1} to MLflow")

    def _aggregate_results(self, period_results: list[dict]) -> dict:
        """Aggregate metrics across all trading periods."""
        if not period_results:
            return {}

        total_trades = 0
        returns = []
        sharpes = []
        win_rates = []

        for pr in period_results:
            m = pr["metrics"]
            total_trades += m.total_trades
            returns.append(m.total_return_pct)
            sharpes.append(m.sharpe_ratio)
            if m.win_rate is not None:
                win_rates.append(m.win_rate)

        n = len(period_results)
        return {
            "num_periods": n,
            "total_trades": total_trades,
            "mean_return_pct": sum(returns) / n if n else 0,
            "mean_sharpe_ratio": sum(sharpes) / n if n else 0,
            "mean_win_rate": sum(win_rates) / len(win_rates) if win_rates else 0,
            "min_return_pct": min(returns) if returns else 0,
            "max_return_pct": max(returns) if returns else 0,
            "periods_adopted_new_params": sum(1 for pr in period_results if pr["adopted"]),
        }

    def _create_mlflow_client(self):
        """Try to create an MLflow client from config."""
        try:
            from utils.mlflow_client import MLflowClient
            return MLflowClient.from_config()
        except Exception:
            logger.warning("MLflow client not available — results will not be logged")
            return None

    def on_tick(self, tick):
        """Not used — individual BacktestingEngines handle ticks."""
        pass

    def finalize(self):
        """Not used — individual BacktestingEngines handle finalization."""
        pass
