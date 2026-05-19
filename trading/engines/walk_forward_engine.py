"""
Walk-forward backtesting engine.

Computes walk-forward optimization/adoption decisions, then runs a single
continuous backtest over the full data stream while only allowing new trades
during the approved trade windows.
"""
from __future__ import annotations

import copy
import os
from pprint import pformat
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from trading.analysis.analysis_engine import AnalysisEngine
from trading.core.algorithm import Algorithm
from trading.core.classes import PriceData
from trading.core.om.order_manager import OrderManager
from trading.core.portfolio import Portfolio
from trading.data_providers.data_provider import DataProvider
from trading.engines.backtest_engine import BacktestingEngine
from trading.engines.base_engine import BaseEngine
from trading.engines.walk_forward_policy import (
    WalkForwardDecision,
    WalkForwardPeriod,
    compute_walk_forward_periods,
    decide_walk_forward_adoption,
)
from utils.logger import Logger
from utils.utils import apply_tunable_config, build_tunable_patch

logger = Logger().get_logger(__name__)


class WalkForwardEngine(BaseEngine):
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
        self.validation_window_days = wf.get("validation_window_days", 30)
        self.trading_window_days = wf.get("trading_window_days", 30)
        self.improvement_threshold_pct = wf.get("improvement_threshold_pct", 5.0)
        self.min_validation_trades = wf.get("min_validation_trades", 0)
        self.objective_metric = wf.get("objective_metric", "annualized_return")
        self.num_trials = wf.get("num_trials", 50)
        self.max_concurrent_trials = wf.get("max_concurrent_trials", 8)
        self.log_ray_worker_output = wf.get("log_ray_worker_output", True)

        self._search_space_cfg = wf.get("search_space", {})
        self.algorithm_param_keys = wf.get("algorithm_param_keys", [])
        self.portfolio_param_keys = wf.get("portfolio_param_keys", [])

        self.experiment_name = cfg.get("experiment_name", "Walk Forward Backtest")
        self.run_name = cfg.get("run_name", "WalkForward")
        self.description = cfg.get("description", "")
        self.log_to_mlflow = cfg.get("log_to_mlflow", True)
        self.tracking_uri = cfg.get("tracking_uri")

        self.original_dp_cfg = copy.deepcopy(dp.cfg) if dp else {}
        self.original_al_cfg = copy.deepcopy(al.cfg) if (al and hasattr(al, "cfg")) else {}
        self.original_pf_cfg = copy.deepcopy(pf.cfg) if (pf and hasattr(pf, "cfg")) else {}
        self.original_al_history_length = getattr(al, "history_length", 0) if al else 0
        self.original_pf_keep_history = getattr(pf, "keep_history", True) if pf else True
        self.starting_cash = pf.cash if pf else 0.0

        self._dp_class = type(dp) if dp else None
        self._al_class = type(al) if al else None
        self._om_class = type(om) if om else None
        self._pf_class = type(pf) if pf else None

        symbol = self.original_pf_cfg.get("symbol", self.original_pf_cfg.get("upro_symbol", ""))
        self._base_backtest_cfg = {
            "symbol": symbol,
            "run_name": self.run_name,
            "description": self.description,
            "starting_cash": self.starting_cash,
            "experiment_name": self.experiment_name,
        }

        self._planned_events: list[dict[str, Any]] = []

    def _log_period_decision_event(
        self,
        *,
        period_idx: int,
        period: WalkForwardPeriod,
        current_al_cfg: dict,
        current_pf_cfg: dict,
        challenger_al_cfg: dict,
        challenger_pf_cfg: dict,
        decision: WalkForwardDecision | None,
        adopted: bool,
        event_id: str | None,
        incumbent_metrics: Any | None,
        challenger_metrics: Any | None,
    ) -> None:
        def _log_metrics_block(title: str, metrics_obj: Any | None) -> None:
            logger.info(title, color="magenta")
            if metrics_obj is None:
                logger.info("  n/a", color="magenta")
                return
            if is_dataclass(metrics_obj):
                metrics_dict = asdict(metrics_obj)
            else:
                metrics_dict = dict(vars(metrics_obj))
            important_keys = [
                "annualized_return",
                "total_return_pct",
                "sharpe_ratio",
                "sortino_ratio",
                "max_drawdown_pct",
                "win_rate",
                "profit_factor",
                "total_trades",
                "winning_trades",
                "losing_trades",
                "avg_trade_pnl",
                "avg_trade_duration",
                "volatility",
                "calmar_ratio",
                "final_equity",
            ]
            for key in important_keys:
                value = metrics_dict.get(key)
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.6f}", color="magenta")
                else:
                    logger.info(f"  {key}: {value}", color="magenta")

        logger.info("")
        logger.info("=" * 80, color="magenta")
        logger.info(f"WALK-FORWARD OPTIMIZATION RESULT - PERIOD {period_idx + 1}", color="magenta")
        logger.info("=" * 80, color="magenta")
        logger.info(f"Event ID:            {event_id or 'n/a'}", color="magenta")
        logger.info(
            f"Optimization Window: {period.optimization_start.date()} -> {period.optimization_end.date()}",
            color="magenta",
        )
        logger.info(
            f"Validation Window:   {period.validation_start.date()} -> {period.validation_end.date()}",
            color="magenta",
        )
        logger.info(
            f"Trading Window:      {period.trading_start.date()} -> {period.trading_end.date()}",
            color="magenta",
        )

        if decision is None:
            logger.info("Decision:            FIRST PERIOD -> CHALLENGER SCHEDULED", color="magenta")
            logger.info("Reason:              first_period_adopted", color="magenta")
            logger.info("Comparison:          No incumbent comparison for first period", color="magenta")
        else:
            logger.info(
                f"Decision:            {'NEW CONFIG WILL BE APPLIED' if adopted else 'KEEPING INCUMBENT CONFIG'}",
                color="magenta",
            )
            logger.info(f"Reason:              {decision.reason}", color="magenta")
            logger.info(f"Objective Metric:    {decision.objective_metric}", color="magenta")
            logger.info(f"Incumbent Metric:    {decision.incumbent_metric:.6f}", color="magenta")
            logger.info(f"Challenger Metric:   {decision.challenger_metric:.6f}", color="magenta")
            logger.info(f"Incumbent Trades:    {decision.incumbent_trades}", color="magenta")
            logger.info(f"Challenger Trades:   {decision.challenger_trades}", color="magenta")
            logger.info(f"Improvement Pct:     {decision.improvement_pct:.2f}", color="magenta")
            logger.info(f"Threshold Pct:       {decision.threshold_pct:.2f}", color="magenta")
            logger.info(f"Min Validation:      {decision.min_validation_trades}", color="magenta")

        logger.info("-" * 80, color="magenta")
        _log_metrics_block("Incumbent Validation Metrics:", incumbent_metrics)
        _log_metrics_block("Challenger Validation Metrics:", challenger_metrics)
        logger.info("-" * 80, color="magenta")
        logger.info("Incumbent Algorithm Params:", color="magenta")
        for line in pformat(current_al_cfg, sort_dicts=True).splitlines():
            logger.info(line, color="magenta")
        logger.info("Incumbent Portfolio Params:", color="magenta")
        for line in pformat(current_pf_cfg, sort_dicts=True).splitlines():
            logger.info(line, color="magenta")
        logger.info("Challenger Algorithm Params:", color="magenta")
        for line in pformat(challenger_al_cfg, sort_dicts=True).splitlines():
            logger.info(line, color="magenta")
        logger.info("Challenger Portfolio Params:", color="magenta")
        for line in pformat(challenger_pf_cfg, sort_dicts=True).splitlines():
            logger.info(line, color="magenta")
        logger.info("=" * 80, color="magenta")

    def _get_date_range(self) -> tuple[datetime, datetime]:
        self.dp.load_data()
        df = self.dp.data
        timestamps = pd.to_datetime(df["timestamp"])
        return timestamps.min().to_pydatetime(), timestamps.max().to_pydatetime()

    def _compute_periods(self) -> list[WalkForwardPeriod]:
        data_start, data_end = self._get_date_range()
        return compute_walk_forward_periods(
            data_start=data_start,
            data_end=data_end,
            optimization_window_days=self.optimization_window_days,
            validation_window_days=self.validation_window_days,
            trading_window_days=self.trading_window_days,
        )

    def _parse_search_space(self) -> dict:
        from utils.utils import parse_search_space

        return parse_search_space(self._search_space_cfg)

    def _create_dp_for_range(self, start: datetime, end: datetime) -> DataProvider:
        dp_cfg = copy.deepcopy(self.original_dp_cfg)
        dp_cfg["start_date"] = start.isoformat(sep=" ")
        dp_cfg["end_date"] = (end - pd.Timedelta(microseconds=1)).isoformat(sep=" ")
        return self._dp_class(dp_cfg)

    def _build_algorithm(self, al_cfg: dict) -> Algorithm:
        return self._al_class(copy.deepcopy(al_cfg), history_length=self.original_al_history_length)

    def _build_portfolio(self, pf_cfg: dict, om: OrderManager) -> Portfolio:
        return self._pf_class(
            copy.deepcopy(pf_cfg), om, self.starting_cash, {}, self.original_pf_keep_history
        )

    def _run_backtest(self, al_cfg: dict, pf_cfg: dict, dp: DataProvider) -> dict:
        om = self._om_class()
        al = self._build_algorithm(al_cfg)
        pf = self._build_portfolio(pf_cfg, om)
        engine = BacktestingEngine({}, dp, al, om, pf)
        engine.run()
        analysis = AnalysisEngine(pf, om)
        return {
            "metrics": analysis.calculate_metrics(),
            "trades": analysis.extract_trades(),
            "portfolio": pf,
            "order_manager": om,
            "analysis_engine": analysis,
        }

    def _run_optimization(self, opt_dp: DataProvider) -> dict:
        from trading.launchers.run_backtest_ray import tune_backtest_hyperparameters

        return tune_backtest_hyperparameters(
            symbol=self._base_backtest_cfg["symbol"],
            algorithm_class=self._al_class,
            portfolio_class=self._pf_class,
            data_provider_class=self._dp_class,
            order_manager_class=self._om_class,
            base_algorithm_config=copy.deepcopy(self.original_al_cfg),
            base_portfolio_config=copy.deepcopy(self.original_pf_cfg),
            base_data_provider_config=copy.deepcopy(opt_dp.cfg),
            base_backtest_config=self._base_backtest_cfg,
            search_space=self._parse_search_space(),
            algorithm_param_keys=self.algorithm_param_keys,
            portfolio_param_keys=self.portfolio_param_keys,
            num_samples=self.num_trials,
            max_concurrent_trials=self.max_concurrent_trials,
            log_to_mlflow=False,
            log_ray_worker_output=self.log_ray_worker_output,
        )

    def _apply_config(self, best_config: dict) -> tuple[dict, dict]:
        al_cfg = apply_tunable_config(self.original_al_cfg, best_config, self.algorithm_param_keys)
        pf_cfg = apply_tunable_config(self.original_pf_cfg, best_config, self.portfolio_param_keys)
        return al_cfg, pf_cfg

    def _evaluate_period(
        self,
        period_idx: int,
        period: WalkForwardPeriod,
        current_al_cfg: dict,
        current_pf_cfg: dict,
    ) -> dict[str, Any]:
        opt_dp = self._create_dp_for_range(period.optimization_start, period.optimization_end)
        logger.info("Running HPO optimization...")
        best_config = self._run_optimization(opt_dp)
        challenger_al_cfg, challenger_pf_cfg = self._apply_config(best_config)
        logger.info(f"HPO best config: {best_config}")

        decision: WalkForwardDecision | None = None
        incumbent_metrics = None
        challenger_metrics = None
        adopted = True
        if period_idx > 0:
            incumbent_result = self._run_backtest(
                current_al_cfg,
                current_pf_cfg,
                self._create_dp_for_range(period.validation_start, period.validation_end),
            )
            challenger_result = self._run_backtest(
                challenger_al_cfg,
                challenger_pf_cfg,
                self._create_dp_for_range(period.validation_start, period.validation_end),
            )
            incumbent_metrics = incumbent_result["metrics"]
            challenger_metrics = challenger_result["metrics"]
            decision = decide_walk_forward_adoption(
                incumbent_metrics,
                challenger_metrics,
                objective_metric=self.objective_metric,
                improvement_threshold_pct=self.improvement_threshold_pct,
                min_validation_trades=self.min_validation_trades,
            )
            adopted = decision.adopted
            logger.info(
                f"Validation decision: incumbent={decision.incumbent_metric:.4f}, "
                f"challenger={decision.challenger_metric:.4f}, "
                f"improvement={decision.improvement_pct:.2f}% reason={decision.reason}"
            )
        else:
            logger.info("First period - adopting HPO params unconditionally")

        trade_al_cfg = challenger_al_cfg if adopted else current_al_cfg
        trade_pf_cfg = challenger_pf_cfg if adopted else current_pf_cfg
        algo_patch = build_tunable_patch(best_config, self.algorithm_param_keys) if adopted else {}
        pf_patch = build_tunable_patch(best_config, self.portfolio_param_keys) if adopted else {}
        event_id = self._persist_optimization_event(
            period_idx=period_idx,
            period=period,
            best_config=best_config,
            current_al_cfg=current_al_cfg,
            current_pf_cfg=current_pf_cfg,
            challenger_al_cfg=challenger_al_cfg,
            challenger_pf_cfg=challenger_pf_cfg,
            trade_al_cfg=trade_al_cfg,
            trade_pf_cfg=trade_pf_cfg,
            decision=decision,
            adopted=adopted,
        )
        self._log_period_decision_event(
            period_idx=period_idx,
            period=period,
            current_al_cfg=current_al_cfg,
            current_pf_cfg=current_pf_cfg,
            challenger_al_cfg=challenger_al_cfg,
            challenger_pf_cfg=challenger_pf_cfg,
            decision=decision,
            adopted=adopted,
            event_id=event_id,
            incumbent_metrics=incumbent_metrics,
            challenger_metrics=challenger_metrics,
        )
        plan = {
            "period_idx": period_idx,
            "period": period,
            "best_config": best_config,
            "decision": decision.as_dict() if decision else None,
            "adopted": adopted,
            "al_cfg": trade_al_cfg,
            "pf_cfg": trade_pf_cfg,
            "algo_patch": algo_patch,
            "pf_patch": pf_patch,
            "event_id": event_id,
        }
        return plan

    def _persist_optimization_event(
        self,
        *,
        period_idx: int,
        period: WalkForwardPeriod,
        best_config: dict,
        current_al_cfg: dict,
        current_pf_cfg: dict,
        challenger_al_cfg: dict,
        challenger_pf_cfg: dict,
        trade_al_cfg: dict,
        trade_pf_cfg: dict,
        decision: WalkForwardDecision | None,
        adopted: bool,
    ) -> str | None:
        if self.state_store is None or self.session_id is None:
            return None
        event = {
            "event_type": "walk_forward_backtest",
            "period_idx": period_idx,
            "optimization_start": period.optimization_start,
            "optimization_end": period.optimization_end,
            "validation_start": period.validation_start,
            "validation_end": period.validation_end,
            "trading_start": period.trading_start,
            "trading_end": period.trading_end,
            "objective_metric": self.objective_metric,
            "improvement_threshold_pct": self.improvement_threshold_pct,
            "min_validation_trades": self.min_validation_trades,
            "incumbent_algorithm_params": current_al_cfg,
            "incumbent_portfolio_params": current_pf_cfg,
            "challenger_algorithm_params": challenger_al_cfg,
            "challenger_portfolio_params": challenger_pf_cfg,
            "best_config_raw": best_config,
            "adopted": adopted,
            "active_algorithm_params": trade_al_cfg,
            "active_portfolio_params": trade_pf_cfg,
            "status": "scheduled_activation" if adopted else "rejected",
            "activation_time": None,
        }
        if decision is not None:
            event.update(decision.as_dict())
        return self.state_store.save_optimization_event(self.session_id, event)

    def _update_optimization_event_activation(self, event_id: str | None, timestamp: datetime) -> None:
        if self.state_store is None or self.session_id is None or event_id is None:
            return
        self.state_store.update_optimization_event(
            self.session_id,
            event_id,
            {
                "status": "activated",
                "activation_time": timestamp,
            },
        )

    def _process_tick(self, tick: list[PriceData], allow_trading: bool):
        market_signals = self.al.on_data(tick)
        effective_signals = market_signals if allow_trading else []
        tick_results = self.pf.process_market_signals_for_tick(effective_signals, tick)
        self._persist_tick(tick, effective_signals, tick_results)
        return tick_results

    def _run_continuous_backtest(self, plans: list[dict[str, Any]]) -> dict[str, Any]:
        trade_idx = 0
        activation_idx = 0
        activated_event_ids: set[str] = set()

        logger.info("Running continuous walk-forward simulation...")
        for tick in self.dp.iterate():
            timestamp = pd.to_datetime(tick[0].timestamp).to_pydatetime() if tick else None
            if timestamp is None:
                continue

            while activation_idx < len(plans) and timestamp >= plans[activation_idx]["period"].trading_start:
                plan = plans[activation_idx]
                if plan["event_id"] not in activated_event_ids:
                    logger.info(
                        (
                            f"[WF EVENT] Activating config for period {plan['period_idx'] + 1} | "
                            f"event_id={(plan['event_id'] or 'n/a')} | "
                            f"timestamp={timestamp}\n"
                            f"Algorithm patch:\n{pformat(plan['algo_patch'], sort_dicts=True)}\n"
                            f"Portfolio patch:\n{pformat(plan['pf_patch'], sort_dicts=True)}"
                        ),
                        color="magenta",
                    )
                    if plan["algo_patch"]:
                        self.al.reconfigure(plan["algo_patch"])
                    if plan["pf_patch"]:
                        self.pf.reconfigure(plan["pf_patch"])
                    self._update_optimization_event_activation(plan["event_id"], timestamp)
                    activated_event_ids.add(plan["event_id"])
                activation_idx += 1

            while trade_idx < len(plans) and timestamp >= plans[trade_idx]["period"].trading_end:
                trade_idx += 1

            allow_trading = (
                trade_idx < len(plans)
                and plans[trade_idx]["period"].trading_start <= timestamp < plans[trade_idx]["period"].trading_end
            )
            self._process_tick(tick, allow_trading)
            self._check_debug()

        analysis = AnalysisEngine(self.pf, self.om)
        results = analysis.run_full_analysis(log_to_mlflow=False, show_summary=False)
        return {
            "analysis": analysis,
            "results": results,
        }

    def _build_optimization_events_rows(self, plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for plan in plans:
            period: WalkForwardPeriod = plan["period"]
            decision = plan.get("decision") or {}
            rows.append(
                {
                    "event_id": plan.get("event_id"),
                    "period_idx": plan["period_idx"],
                    "optimization_start": period.optimization_start.isoformat(),
                    "optimization_end": period.optimization_end.isoformat(),
                    "validation_start": period.validation_start.isoformat(),
                    "validation_end": period.validation_end.isoformat(),
                    "trading_start": period.trading_start.isoformat(),
                    "trading_end": period.trading_end.isoformat(),
                    "incumbent_metric": decision.get("incumbent_metric"),
                    "challenger_metric": decision.get("challenger_metric"),
                    "incumbent_trades": decision.get("incumbent_trades"),
                    "challenger_trades": decision.get("challenger_trades"),
                    "improvement_pct": decision.get("improvement_pct"),
                    "adopted": plan["adopted"],
                    "reason": decision.get("reason", "first_period_adopted"),
                    "best_config": plan["best_config"],
                }
            )
        return rows

    def _build_aggregate_summary(self, plans: list[dict[str, Any]], metrics) -> dict[str, Any]:
        return {
            "num_periods": len(plans),
            "periods_adopted_new_params": sum(1 for plan in plans if plan["adopted"]),
            "wf_total_return_pct": metrics.total_return_pct,
            "wf_annualized_return": metrics.annualized_return,
            "wf_sharpe_ratio": metrics.sharpe_ratio,
            "wf_max_drawdown_pct": metrics.max_drawdown_pct,
            "wf_total_trades": metrics.total_trades,
            "wf_final_equity": metrics.final_equity,
        }

    def _log_optimization_events_artifacts(self, mlflow_client, event_rows: list[dict[str, Any]]) -> None:
        mlflow_client.log_json(event_rows, "optimization_events.json")
        if not event_rows:
            return

        table_df = pd.DataFrame(event_rows)
        csv_df = table_df.copy()
        if "best_config" in csv_df.columns:
            csv_df["best_config"] = csv_df["best_config"].apply(lambda x: str(x))

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "optimization_events.csv")
            csv_df.to_csv(csv_path, index=False)
            mlflow_client.log_artifact(csv_path)

        markdown_lines = [
            "# Walk-Forward Optimization Events",
            "",
            "| Period | Event ID | Challenger Metric | Improvement % | Adopted | Reason |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in event_rows:
            metric = row["challenger_metric"]
            improvement = row["improvement_pct"]
            markdown_lines.append(
                "| "
                f"{row['period_idx'] + 1} | "
                f"{(row['event_id'] or '')[:8]} | "
                f"{'' if metric is None else f'{metric:.4f}'} | "
                f"{'' if improvement is None else f'{improvement:.2f}'} | "
                f"{row['adopted']} | "
                f"{row['reason']} |"
            )
        mlflow_client.log_markdown("\n".join(markdown_lines), "optimization_events.md")

    def _plot_equity_with_events(self, analysis: AnalysisEngine, plans: list[dict[str, Any]]):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        equity_curve = pd.Series(self.pf.value_history)
        equity_curve.index = pd.to_datetime(list(self.pf.tick_history.keys()), utc=True)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(equity_curve.index, equity_curve.values, linewidth=2, color="navy")
        ax.set_title("Walk-Forward Equity Curve with Optimization Events", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        for plan in plans:
            period: WalkForwardPeriod = plan["period"]
            activation_ts = pd.Timestamp(period.trading_start, tz="UTC")
            idx = equity_curve.index.searchsorted(activation_ts, side="left")
            if idx >= len(equity_curve):
                continue
            curve_ts = equity_curve.index[idx]
            curve_val = equity_curve.iloc[idx]
            label = (plan.get("event_id") or "no-event")[:8]

            decision = plan.get("decision") or {}
            improvement = decision.get("improvement_pct")
            improvement_text = "" if improvement is None else f"\n{improvement:.1f}%"

            if plan["adopted"]:
                marker = "^"
                color = "green"
                text = f"A:{label}{improvement_text}"
            else:
                marker = "x"
                color = "gray"
                text = f"R:{label}{improvement_text}"

            ax.scatter(curve_ts, curve_val, marker=marker, color=color, s=70, zorder=5)
            ax.annotate(
                text,
                (curve_ts, curve_val),
                textcoords="offset points",
                xytext=(0, 10 if plan["adopted"] else -18),
                ha="center",
                fontsize=8,
                color=color,
            )

        plt.tight_layout()
        return fig

    def _log_full_run_to_mlflow(
        self,
        analysis: AnalysisEngine,
        analysis_results: dict[str, Any],
        plans: list[dict[str, Any]],
        aggregate: dict[str, Any],
    ) -> None:
        mlflow_client = self._create_mlflow_client()
        if mlflow_client is None:
            return

        metrics = analysis_results["metrics"]
        event_rows = self._build_optimization_events_rows(plans)
        try:
            with mlflow_client.start_run(
                run_name=self.run_name,
                description=self.description or f"Walk-forward backtest with {len(plans)} periods",
            ):
                mlflow_client.log_params(
                    {
                        "optimization_window_days": self.optimization_window_days,
                        "validation_window_days": self.validation_window_days,
                        "trading_window_days": self.trading_window_days,
                        "num_periods": len(plans),
                        "num_trials": self.num_trials,
                        "improvement_threshold_pct": self.improvement_threshold_pct,
                        "min_validation_trades": self.min_validation_trades,
                        "objective_metric": self.objective_metric,
                    }
                )
                mlflow_client.log_metrics(
                    {
                        **{
                            k: v
                            for k, v in asdict(metrics).items()
                            if isinstance(v, (int, float)) and v is not None
                        },
                        **{
                            k: v
                            for k, v in aggregate.items()
                            if isinstance(v, (int, float)) and v is not None
                        },
                    }
                )
                mlflow_client.log_text(analysis_results["report"], "walk_forward_report.txt")
                mlflow_client.log_chart(analysis.plot_equity_curve(show=False), "equity_curve", format="png", dpi=150)
                mlflow_client.log_chart(
                    self._plot_equity_with_events(analysis, plans),
                    "walk_forward_equity_with_events",
                    format="png",
                    dpi=150,
                )
                self._log_optimization_events_artifacts(mlflow_client, event_rows)
        except Exception as exc:
            logger.warning(f"Failed to log walk-forward run to MLflow: {exc}")

    def run(self):
        periods = self._compute_periods()
        if not periods:
            logger.error("No valid periods could be computed from data range")
            return {"periods": [], "aggregate": {}}

        logger.info("=" * 80)
        logger.info("WALK-FORWARD BACKTEST")
        logger.info("=" * 80)
        logger.info(f"Optimization window: {self.optimization_window_days} days")
        logger.info(f"Validation window:   {self.validation_window_days} days")
        logger.info(f"Trading window:      {self.trading_window_days} days")
        logger.info(f"Number of periods: {len(periods)}")
        logger.info(f"HPO trials per period: {self.num_trials}")
        logger.info(f"Improvement threshold: {self.improvement_threshold_pct}%")
        logger.info("=" * 80)

        plans: list[dict[str, Any]] = []
        current_al_cfg = copy.deepcopy(self.original_al_cfg)
        current_pf_cfg = copy.deepcopy(self.original_pf_cfg)

        for i, period in enumerate(periods):
            logger.info("")
            logger.info(f"--- Period {i + 1}/{len(periods)} ---")
            logger.info(
                f"Optimization: {period.optimization_start.date()} to {period.optimization_end.date()}"
            )
            logger.info(
                f"Validation:   {period.validation_start.date()} to {period.validation_end.date()}"
            )
            logger.info(f"Trading:      {period.trading_start.date()} to {period.trading_end.date()}")

            plan = self._evaluate_period(i, period, current_al_cfg, current_pf_cfg)
            plans.append(plan)
            if plan["adopted"]:
                current_al_cfg = plan["al_cfg"]
                current_pf_cfg = plan["pf_cfg"]

        self._planned_events = plans
        continuous = self._run_continuous_backtest(plans)
        analysis = continuous["analysis"]
        analysis_results = continuous["results"]
        aggregate = self._build_aggregate_summary(plans, analysis_results["metrics"])
        self._log_full_run_to_mlflow(analysis, analysis_results, plans, aggregate)

        return {
            "periods": plans,
            "aggregate": aggregate,
            "metrics": analysis_results["metrics"],
            "optimization_events": self._build_optimization_events_rows(plans),
        }

    def _create_mlflow_client(self):
        try:
            from utils.mlflow_client import MLflowClient

            if not self.log_to_mlflow:
                return None
            return MLflowClient(
                experiment_name=self.experiment_name,
                tracking_uri=self.tracking_uri,
                enabled=True,
            )
        except Exception:
            logger.warning("MLflow client not available - results will not be logged")
            return None

    def on_tick(self, tick: list[PriceData]):
        return self._process_tick(tick, allow_trading=True)

    def finalize(self):
        pass
