from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile

import pandas as pd

from trading.commands.analysis import get_git_info
from trading.commands.common import (
    apply_cli_overrides,
    apply_session_log_file,
    load_account_creds,
    load_raw_config,
    resolve_alpaca_credentials,
)
from trading.reporting import CombinedArtifactSpec, AnalyzerReportTarget, ExperimentReport, ExperimentReporter
from utils.logger import Logger
from utils.utils import instantiate_from_string

logger = Logger().get_logger(__name__)


def _resolve_state_store_params(ss_cfg: dict) -> tuple[str, str]:
    from trading.data_providers.session_replay_data_provider import SessionReplayDataProvider

    return SessionReplayDataProvider._resolve_mongo_params(
        ss_cfg.get("connection_uri"),
        ss_cfg.get("database"),
    )


def _build_algorithm_from_config(raw_cfg: dict, fallback_class_path: str, fallback_cfg: dict, fallback_history_length: int):
    if not raw_cfg.get("_use_config_components_for_replay"):
        return instantiate_from_string(
            fallback_class_path,
            cfg=dict(fallback_cfg),
            history_length=fallback_history_length,
        )

    from trading.config.component_loader import instantiate_component
    from trading.config.service import normalize_config_dict

    component = normalize_config_dict({k: v for k, v in raw_cfg.items() if not k.startswith("_")}).algorithm
    kwargs = dict(component.params)
    history_length = kwargs.pop("history_length", 0)
    return instantiate_component(component, cfg=kwargs, history_length=history_length)


def _build_portfolio_from_config(raw_cfg: dict, fallback_class_path: str, fallback_cfg: dict, om, *, keep_history: bool = True):
    if not raw_cfg.get("_use_config_components_for_replay"):
        return instantiate_from_string(
            fallback_class_path,
            cfg={**fallback_cfg, "keep_history": keep_history},
            order_manager=om,
        )

    from trading.config.component_loader import instantiate_component
    from trading.config.service import normalize_config_dict

    component = normalize_config_dict({k: v for k, v in raw_cfg.items() if not k.startswith("_")}).portfolio
    kwargs = {**component.params, "keep_history": keep_history}
    return instantiate_component(component, cfg=kwargs, order_manager=om)


def _run_clean_mongo_backtest(
    *,
    args: argparse.Namespace,
    raw_cfg: dict,
    meta: dict,
    al_class_path: str | None,
    al_cfg_raw: dict,
    pf_class_path: str | None,
    pf_cfg_raw: dict,
    history_length: int,
    connection_uri: str,
    database: str,
):
    from trading.analysis.portfolio_analyzer import PortfolioAnalyzer
    from trading.commands.common import flatten_config
    from trading.core.om.backtesting_om import BacktestingOrderManager
    from trading.data_providers.mongodb_data_provider import MongoDBDataProvider
    from trading.engines.backtest_engine import BacktestingEngine

    om = BacktestingOrderManager(cfg={"market_hours_only": True})
    al = _build_algorithm_from_config(raw_cfg, al_class_path, al_cfg_raw, history_length)
    pf = _build_portfolio_from_config(raw_cfg, pf_class_path, pf_cfg_raw, om)

    mongo_dp = MongoDBDataProvider(cfg={
        "session_id": args.session_id,
        "connection_uri": connection_uri,
        "database": database,
    })
    BacktestingEngine(cfg={}, dp=mongo_dp, al=al, om=om, pf=pf).run()
    logger.info(
        f"Clean MongoDB backtest complete - Value: ${pf.total_value:,.2f}, "
        f"Cash: ${pf.cash:,.2f}, Positions: {list(pf.positions.keys())}"
    )

    analysis_cfg = raw_cfg.get("analysis", {})
    log_mlflow = analysis_cfg.get("log_to_mlflow", True) and not getattr(args, "no_mlflow", False)
    experiment_name = analysis_cfg.get("experiment_name", "Session Replay")
    run_name = getattr(args, "run_name", None) or analysis_cfg.get("run_name") or f"mongo-backtest-{args.session_id[:28]}"

    analyzer = PortfolioAnalyzer(pf)
    report, summary = ExperimentReporter.build_single_report(
        analyzer=analyzer,
        experiment_name=experiment_name,
        run_name=run_name,
        description=analysis_cfg.get("description"),
        tags=get_git_info() or None,
        parameters={
            k[:250]: v
            for k, v in {
                "session_id": args.session_id,
                "replay_mode": "clean_mongo_backtest",
                **flatten_config({"meta": meta}),
                **({"source_run_url": args.source_run_url} if getattr(args, "source_run_url", None) else {}),
                **({"source_run_id": args.source_run_id} if getattr(args, "source_run_id", None) else {}),
            }.items()
            if isinstance(v, (str, int, float, bool)) or v is None
        },
        config_artifact_paths=[args.config] if getattr(args, "config", None) else None,
    )

    mlflow_info: dict[str, str] = {}
    if log_mlflow:
        mlflow_info = ExperimentReporter.log_to_mlflow(report) or {}
        logger.info("MLflow run complete")
    else:
        ExperimentReporter.show_summary(summary)

    metrics = analyzer.get_metrics()
    return {
        "session_id": args.session_id,
        "mode": "clean_mongo_backtest",
        "mlflow_run_id": mlflow_info.get("run_id"),
        "mlflow_run_url": mlflow_info.get("run_url"),
        "final_equity": metrics.final_equity,
    }


def cmd_session_replay(args: argparse.Namespace):
    raw_cfg = load_raw_config(args.config)
    raw_cfg = apply_cli_overrides(raw_cfg, args)
    if getattr(args, "use_config_components", False):
        raw_cfg["_use_config_components_for_replay"] = True
    apply_session_log_file(raw_cfg, args)
    experiment_name_override = getattr(args, "mlflow_experiment_name_override", None)
    if experiment_name_override:
        raw_cfg.setdefault("analysis", {})["experiment_name"] = experiment_name_override
    creds = load_account_creds(args.account)
    raw_cfg = resolve_alpaca_credentials(raw_cfg, creds)

    alpaca_cfg = raw_cfg.get("alpaca", {})
    if not alpaca_cfg.get("api_key") or not alpaca_cfg.get("secret_key"):
        logger.error("Alpaca API credentials required. Set in config or in accounts.yaml.")
        sys.exit(1)

    if not getattr(args, "session_id", None):
        logger.error("session-replay requires --session-id <id>.")
        sys.exit(1)

    from trading.core.classes import BracketOrder, OrderStatus, Position
    from trading.data_providers.session_replay_data_provider import SessionReplayDataProvider
    from utils.trading_state_store import TradingStateStore

    ss_cfg = raw_cfg.get("state_store", {})
    connection_uri, database = _resolve_state_store_params(ss_cfg)
    meta_loader = SessionReplayDataProvider(cfg={
        "session_id": args.session_id,
        "api_key": alpaca_cfg["api_key"],
        "secret_key": alpaca_cfg["secret_key"],
        "connection_uri": connection_uri,
        "database": database,
        "timeframe": getattr(args, "timeframe", None),
    })
    meta_loader.load_data()
    meta = meta_loader._session_metadata

    logger.info(
        f"Session replay: symbols={meta.get('symbols')} "
        f"timeframe={meta.get('timeframe')} warmup_bars={meta.get('warmup_bars')}"
    )

    def _build_opening_state(store: TradingStateStore, session_id: str, start_ts: pd.Timestamp):
        start_dt = start_ts.to_pydatetime()
        first_snapshot = store._snapshots.find_one({"session_id": session_id}, sort=[("timestamp", 1)])

        opening_cash = None
        opening_positions: dict[str, Position] = {}
        if first_snapshot is not None:
            opening_cash = first_snapshot.get("cash")
            for symbol, pos in (first_snapshot.get("positions") or {}).items():
                qty = int(pos.get("quantity", 0) or 0)
                if qty > 0:
                    opening_positions[symbol] = Position(symbol=symbol, quantity=qty)

        def _reset_order_for_replay(order):
            order.processed_by_portfolio = False
            if isinstance(order, BracketOrder):
                order.MANUAL_SALE = False
                order.SOLD_ORDER = None
                order.status = OrderStatus.PENDING_SALE if (
                    order.executed_datetime is not None and order.executed_datetime <= start_dt
                ) else OrderStatus.PENDING
                for child_name in list(order.get_child_order_names()):
                    child = order.get_child_order(child_name)
                    if child_name == "MANUAL_ORDER":
                        order.add_child_order("MANUAL_ORDER", None)
                        continue
                    if child is None:
                        continue
                    child.status = OrderStatus.PENDING
                    child.executed_datetime = None
                    child.cash = 0.0
                    child.processed_by_portfolio = False
            else:
                order.status = OrderStatus.PENDING
            return order

        def _bracket_sold_ts(order: BracketOrder):
            sold_order = order.SOLD_ORDER
            if sold_order is None:
                return None
            return sold_order.executed_datetime or sold_order.placed_datetime

        opening_orders = []
        order_data = store.load_orders(session_id)
        for order in order_data["all_orders"].values():
            placed_ts = order.placed_datetime
            executed_ts = order.executed_datetime
            if placed_ts is None or placed_ts > start_dt:
                continue

            if isinstance(order, BracketOrder):
                if executed_ts is None or executed_ts > start_dt:
                    opening_orders.append(_reset_order_for_replay(copy.deepcopy(order)))
                    continue

                sold_ts = _bracket_sold_ts(order)
                if order.status == OrderStatus.FILLED:
                    if sold_ts is not None and sold_ts > start_dt:
                        opening_orders.append(_reset_order_for_replay(copy.deepcopy(order)))
                    continue
                if order.status in {OrderStatus.PENDING, OrderStatus.PENDING_SALE}:
                    opening_orders.append(_reset_order_for_replay(copy.deepcopy(order)))
            elif order.status == OrderStatus.PENDING:
                opening_orders.append(_reset_order_for_replay(copy.deepcopy(order)))

        return opening_cash, opening_positions, opening_orders

    al_class_path = meta.get("algorithm_class")
    al_cfg_raw = dict(meta.get("algorithm_config") or {})
    pf_class_path = meta.get("portfolio_class")
    pf_cfg_raw = dict(meta.get("portfolio_config") or {})
    if getattr(args, "cash", None) is not None:
        pf_cfg_raw["cash"] = args.cash

    if not raw_cfg.get("_use_config_components_for_replay") and (not al_class_path or not pf_class_path):
        logger.error(
            "Session metadata is missing algorithm_class / portfolio_class. "
            "This session was created before replay metadata was stored."
        )
        sys.exit(1)

    warmup_bars = meta.get("warmup_bars", 0)
    history_length = al_cfg_raw.pop("history_length", 0)

    if getattr(args, "clean_mongo_backtest", False):
        return _run_clean_mongo_backtest(
            args=args,
            raw_cfg=raw_cfg,
            meta=meta,
            al_class_path=al_class_path,
            al_cfg_raw=al_cfg_raw,
            pf_class_path=pf_class_path,
            pf_cfg_raw=pf_cfg_raw,
            history_length=history_length,
            connection_uri=connection_uri,
            database=database,
        )

    al = _build_algorithm_from_config(raw_cfg, al_class_path, al_cfg_raw, history_length)

    from trading.core.om.backtesting_om import BacktestingOrderManager

    om = BacktestingOrderManager(cfg={"market_hours_only": True})
    pf = _build_portfolio_from_config(raw_cfg, pf_class_path, pf_cfg_raw, om)

    session_start = pd.to_datetime(meta["session_start"])
    session_end = pd.to_datetime(meta["session_end"])
    store = TradingStateStore(
        connection_uri=connection_uri,
        database=database,
    )
    opening_cash, opening_positions, opening_orders = _build_opening_state(store, args.session_id, session_start)
    opening_positions_template = copy.deepcopy(opening_positions)
    opening_orders_template = copy.deepcopy(opening_orders)
    if getattr(args, "cash", None) is None and opening_cash is not None:
        pf.cash = float(opening_cash)
    if opening_positions_template:
        pf.positions = copy.deepcopy(opening_positions_template)
    for order in copy.deepcopy(opening_orders_template):
        om.all_orders[order.order_id] = order
        om.pending_orders_by_id[order.order_id] = order
    if hasattr(pf, "_symbol_entry_time"):
        pf._symbol_entry_time = {}
        for order in om.pending_orders_by_id.values():
            if isinstance(order, BracketOrder) and order.status == OrderStatus.PENDING_SALE and order.executed_datetime is not None:
                pf._symbol_entry_time[order.symbol] = order.executed_datetime
    logger.info(
        f"Restored opening replay state: cash={pf.cash:.2f}, "
        f"positions={{{', '.join(f'{k}:{v.quantity}' for k, v in pf.positions.items())}}}, "
        f"active_orders={len(om.pending_orders_by_id)}"
    )

    from trading.data_providers.alpaca_data_provider import AlpacaDataProvider
    from trading.engines.backtest_engine import BacktestingEngine

    if warmup_bars > 0:
        from utils.utils import compute_warmup_start_date

        warmup_start = compute_warmup_start_date(warmup_bars, meta["timeframe"], session_start.to_pydatetime())
        warmup_dp = AlpacaDataProvider(cfg={
            "api_key": alpaca_cfg["api_key"],
            "secret_key": alpaca_cfg["secret_key"],
            "symbols": meta["symbols"],
            "timeframe": meta["timeframe"],
            "start_date": warmup_start.strftime("%Y-%m-%dT%H:%M:%S"),
            "end_date": session_start.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        warmup_dp.load_data()
        al.warm_up(list(warmup_dp.iterate()))
        logger.info(f"Algorithm warmed up (is_warmed_up={al.is_warmed_up})")

    live_dp = AlpacaDataProvider(cfg={
        "api_key": alpaca_cfg["api_key"],
        "secret_key": alpaca_cfg["secret_key"],
        "symbols": meta["symbols"],
        "timeframe": meta["timeframe"],
        "start_date": session_start.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_date": session_end.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    live_dp.load_data()

    engine = BacktestingEngine(cfg={}, dp=live_dp, al=al, om=om, pf=pf)
    engine.run()

    logger.info(
        f"Alpaca replay complete — Value: ${pf.total_value:,.2f}, "
        f"Cash: ${pf.cash:,.2f}, Positions: {list(pf.positions.keys())}"
    )

    pf_extended = None
    if getattr(args, "start_date", None):
        start_dt = pd.to_datetime(args.start_date).to_pydatetime()
        al_ext = _build_algorithm_from_config(raw_cfg, al_class_path, al_cfg_raw, history_length)
        om_ext = BacktestingOrderManager(cfg={"market_hours_only": True})
        pf_ext = _build_portfolio_from_config(raw_cfg, pf_class_path, pf_cfg_raw, om_ext)
        if getattr(args, "cash", None) is not None:
            pf_ext.cash = args.cash
        ext_dp = AlpacaDataProvider(cfg={
            "api_key": alpaca_cfg["api_key"],
            "secret_key": alpaca_cfg["secret_key"],
            "symbols": meta["symbols"],
            "timeframe": meta["timeframe"],
            "start_date": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "end_date": session_end.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        ext_dp.load_data()
        BacktestingEngine(cfg={}, dp=ext_dp, al=al_ext, om=om_ext, pf=pf_ext).run()
        logger.info(
            f"Extended Alpaca replay complete — Value: ${pf_ext.total_value:,.2f}, "
            f"Cash: ${pf_ext.cash:,.2f}, Positions: {list(pf_ext.positions.keys())}"
        )
        pf_extended = pf_ext

    from trading.data_providers.mongodb_data_provider import MongoDBDataProvider

    al_mongo = _build_algorithm_from_config(raw_cfg, al_class_path, al_cfg_raw, history_length)
    om_mongo = BacktestingOrderManager(cfg={"market_hours_only": True})
    pf_mongo = _build_portfolio_from_config(raw_cfg, pf_class_path, pf_cfg_raw, om_mongo)
    if getattr(args, "cash", None) is None and opening_cash is not None:
        pf_mongo.cash = float(opening_cash)
    if opening_positions_template:
        pf_mongo.positions = copy.deepcopy(opening_positions_template)
    for restored in copy.deepcopy(opening_orders_template):
        om_mongo.all_orders[restored.order_id] = restored
        om_mongo.pending_orders_by_id[restored.order_id] = restored
    if hasattr(pf_mongo, "_symbol_entry_time"):
        pf_mongo._symbol_entry_time = {}
        for order in om_mongo.pending_orders_by_id.values():
            if isinstance(order, BracketOrder) and order.status == OrderStatus.PENDING_SALE and order.executed_datetime is not None:
                pf_mongo._symbol_entry_time[order.symbol] = order.executed_datetime

    if warmup_bars > 0:
        al_mongo.warm_up(list(warmup_dp.iterate()))
        logger.info(f"MongoDB-replay algorithm warmed up (is_warmed_up={al_mongo.is_warmed_up})")

    mongo_dp = MongoDBDataProvider(cfg={
        "session_id": args.session_id,
        "connection_uri": connection_uri,
        "database": database,
    })
    BacktestingEngine(cfg={}, dp=mongo_dp, al=al_mongo, om=om_mongo, pf=pf_mongo).run()

    logger.info(
        f"MongoDB replay complete — Value: ${pf_mongo.total_value:,.2f}, "
        f"Cash: ${pf_mongo.cash:,.2f}, Positions: {list(pf_mongo.positions.keys())}"
    )

    analysis_cfg = raw_cfg.get("analysis", {})
    log_mlflow = analysis_cfg.get("log_to_mlflow", True) and not getattr(args, "no_mlflow", False)
    experiment_name = analysis_cfg.get("experiment_name", "Session Replay")
    run_name = getattr(args, "run_name", None) or analysis_cfg.get("run_name") or f"session-replay-{args.session_id[:28]}"

    from trading.analysis.portfolio_analyzer import PortfolioAnalyzer

    replay_analyzer = PortfolioAnalyzer(pf)
    mongo_analyzer = PortfolioAnalyzer(pf_mongo)
    live_analyzer = PortfolioAnalyzer.from_mongodb(
        args.session_id,
        connection_uri=connection_uri,
        database=database,
    )
    replay_metrics = replay_analyzer.get_metrics()
    mongo_metrics = mongo_analyzer.get_metrics()
    live_metrics = live_analyzer.get_metrics()

    def _equity_drift_pct(lhs: float, rhs: float) -> float:
        baseline = abs(rhs) if abs(rhs) > 1e-9 else 1.0
        return abs(lhs - rhs) / baseline * 100.0

    mlflow_info: dict[str, str] = {}
    if log_mlflow:
        from trading.commands.common import flatten_config

        params = {"session_id": args.session_id, **flatten_config({"meta": meta})}
        if getattr(args, "source_run_url", None):
            params["source_run_url"] = args.source_run_url
        if getattr(args, "source_run_id", None):
            params["source_run_id"] = args.source_run_id
        if pf_extended is not None:
            params["extended_start_date"] = str(args.start_date)

        session_start_str = meta.get("session_start")
        align_start = pd.to_datetime(session_start_str, utc=True) if session_start_str else None

        targets = [
            AnalyzerReportTarget(
                name="alpaca_replay",
                analyzer=replay_analyzer,
                summary=ExperimentReporter.build_single_report(
                    analyzer=replay_analyzer,
                    experiment_name=experiment_name,
                    run_name=run_name,
                    description=None,
                    tags=None,
                    parameters=None,
                )[1],
            ),
            AnalyzerReportTarget(
                name="mongo_replay",
                analyzer=mongo_analyzer,
                summary=ExperimentReporter.build_single_report(
                    analyzer=mongo_analyzer,
                    experiment_name=experiment_name,
                    run_name=run_name,
                    description=None,
                    tags=None,
                    parameters=None,
                )[1],
                metric_prefix="mongo_",
                artifact_prefix="mongo_",
            ),
            AnalyzerReportTarget(
                name="live",
                analyzer=live_analyzer,
                summary=ExperimentReporter.build_single_report(
                    analyzer=live_analyzer,
                    experiment_name=experiment_name,
                    run_name=run_name,
                    description=None,
                    tags=None,
                    parameters=None,
                )[1],
                metric_prefix="live_",
                artifact_prefix="live_",
            ),
        ]

        combined_artifacts = [
            CombinedArtifactSpec(
                filename="combined_equity_curve.png",
                builder=lambda path: replay_analyzer.save_combined_equity_curve(
                    live_analyzer, path, self_label="Alpaca Replay", other_label="Live", align_start=align_start
                ),
            ),
            CombinedArtifactSpec(
                filename="combined_lifecycle.html",
                builder=lambda path: replay_analyzer.save_combined_lifecycle_chart_interactive(
                    live_analyzer, path, self_label="Alpaca Replay", other_label="Live", align_start=align_start
                ),
            ),
            CombinedArtifactSpec(
                filename="combined_equity_curve_mongo.png",
                builder=lambda path: mongo_analyzer.save_combined_equity_curve(
                    live_analyzer, path, self_label="MongoDB Replay", other_label="Live", align_start=align_start
                ),
            ),
            CombinedArtifactSpec(
                filename="combined_lifecycle_mongo.html",
                builder=lambda path: mongo_analyzer.save_combined_lifecycle_chart_interactive(
                    live_analyzer, path, self_label="MongoDB Replay", other_label="Live", align_start=align_start
                ),
            ),
        ]

        if pf_extended is not None:
            extended_analyzer = PortfolioAnalyzer(pf_extended)
            targets.append(
                AnalyzerReportTarget(
                    name="extended",
                    analyzer=extended_analyzer,
                    summary=ExperimentReporter.build_single_report(
                        analyzer=extended_analyzer,
                        experiment_name=experiment_name,
                        run_name=run_name,
                        description=None,
                        tags=None,
                        parameters=None,
                    )[1],
                    metric_prefix="extended_",
                    artifact_prefix="extended_",
                )
            )
            combined_artifacts.extend([
                CombinedArtifactSpec(
                    filename="combined_equity_curve_extended.png",
                    builder=lambda path: extended_analyzer.save_combined_equity_curve(
                        live_analyzer, path, self_label="Extended Alpaca", other_label="Live", align_start=align_start
                    ),
                ),
                CombinedArtifactSpec(
                    filename="combined_lifecycle_extended.html",
                    builder=lambda path: extended_analyzer.save_combined_lifecycle_chart_interactive(
                        live_analyzer, path, self_label="Extended Alpaca", other_label="Live", align_start=align_start
                    ),
                ),
            ])

        report = ExperimentReport(
            experiment_name=experiment_name,
            run_name=run_name,
            tags=get_git_info() or None,
            parameters={k[:250]: v for k, v in params.items() if isinstance(v, (str, int, float, bool)) or v is None},
            analyzers=targets,
            combined_artifacts=combined_artifacts,
        )

        mlflow_info = ExperimentReporter.log_to_mlflow(report) or {}
        logger.info("MLflow run complete")
    else:
        summary = ExperimentReporter.build_single_report(
            analyzer=replay_analyzer,
            experiment_name=experiment_name,
            run_name=run_name,
            description=None,
            tags=None,
            parameters=None,
        )[1]
        ExperimentReporter.show_summary(summary)
    return {
        "session_id": args.session_id,
        "mlflow_run_id": mlflow_info.get("run_id"),
        "mlflow_run_url": mlflow_info.get("run_url"),
        "alpaca_live_equity_drift_pct": _equity_drift_pct(replay_metrics.final_equity, live_metrics.final_equity),
        "mongo_live_equity_drift_pct": _equity_drift_pct(mongo_metrics.final_equity, live_metrics.final_equity),
        "alpaca_final_equity": replay_metrics.final_equity,
        "mongo_final_equity": mongo_metrics.final_equity,
        "live_final_equity": live_metrics.final_equity,
    }
