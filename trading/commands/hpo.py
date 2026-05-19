from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import pandas as pd

from trading.analysis.analysis_engine import AnalysisEngine
from trading.commands.analysis import get_git_info
from trading.commands.analysis import _collect_config_artifact_paths
from trading.commands.common import (
    apply_cli_overrides,
    apply_session_log_file,
    build_experiment_config,
    flatten_config,
    load_account_creds,
    load_raw_config,
)
from trading.config.component_loader import import_component_class
from trading.engines.backtest_engine import BacktestingEngine
from trading.engines.walk_forward_policy import metric_value
from utils.mlflow_client import MLflowClient
from utils.logger import Logger
from utils.utils import apply_tunable_config, parse_search_space

logger = Logger().get_logger(__name__)


def _fill_hpo_data_provider_creds(raw_cfg: dict[str, Any], creds: dict[str, str]) -> None:
    dp_section = raw_cfg.get("data_provider", {})
    provider_name = dp_section.get("provider") or dp_section.get("implementation", "")
    if "alpaca" in provider_name.lower():
        target = dp_section.setdefault("params", {}) if "implementation" in dp_section else dp_section
        target["api_key"] = creds["api_key"]
        target["secret_key"] = creds["secret_key"]


def run_hpo_from_raw_config(
    raw_cfg: dict[str, Any],
    config_artifact_path: str | None = None,
    num_samples_override: int | None = None,
    max_concurrent_override: int | None = None,
) -> dict[str, Any]:
    hpo_cfg = raw_cfg.setdefault("hpo", {})
    if num_samples_override is not None:
        hpo_cfg["num_samples"] = num_samples_override
    if max_concurrent_override is not None:
        hpo_cfg["max_concurrent_trials"] = max_concurrent_override

    experiment = build_experiment_config(raw_cfg)
    config_dict = experiment.model_dump(exclude_none=True)

    logger.info("Starting HPO")

    if experiment.data_provider is None:
        raise ValueError("HPO requires a data_provider section.")

    al_class = import_component_class(experiment.algorithm)
    pf_class = import_component_class(experiment.portfolio)
    dp_class = import_component_class(experiment.data_provider)
    om_class = import_component_class(experiment.order_manager)

    base_al_cfg = dict(experiment.algorithm.params)
    history_length = base_al_cfg.pop("history_length", 0)
    if history_length:
        base_al_cfg["history_length"] = history_length
    base_pf_cfg = dict(experiment.portfolio.params)
    starting_cash = float(base_pf_cfg.get("cash", 10000.0))
    base_pf_cfg = {k: v for k, v in base_pf_cfg.items() if k not in ("cash", "keep_history")}
    base_dp_cfg = dict(experiment.data_provider.params)

    analysis_cfg = config_dict.get("analysis", {})
    base_backtest_cfg = {
        "starting_cash": starting_cash,
        "experiment_name": analysis_cfg.get("experiment_name", "HPO"),
        "run_name": analysis_cfg.get("run_name", "HPO_Run"),
        "description": analysis_cfg.get("description", ""),
        "symbol": base_pf_cfg.get("symbol") or base_pf_cfg.get("upro_symbol", ""),
        "config_artifact_path": config_artifact_path,
        "tracking_uri": config_dict.get("mlflow", {}).get("tracking_uri"),
        "git_tags": get_git_info(),
        "benchmark_paths": analysis_cfg.get("benchmarks") or {},
    }
    base_backtest_cfg.update(flatten_config(config_dict))

    from trading.launchers.run_backtest_ray import tune_backtest_hyperparameters

    best_config = tune_backtest_hyperparameters(
        symbol=base_backtest_cfg["symbol"],
        algorithm_class=al_class,
        portfolio_class=pf_class,
        data_provider_class=dp_class,
        order_manager_class=om_class,
        base_algorithm_config=base_al_cfg,
        base_portfolio_config=base_pf_cfg,
        base_data_provider_config=base_dp_cfg,
        base_backtest_config=base_backtest_cfg,
        search_space=parse_search_space(hpo_cfg.get("search_space", {})),
        algorithm_param_keys=hpo_cfg.get("algorithm_param_keys", []),
        portfolio_param_keys=hpo_cfg.get("portfolio_param_keys", []),
        num_samples=hpo_cfg.get("num_samples", 50),
        max_concurrent_trials=hpo_cfg.get("max_concurrent_trials", 8),
        log_to_mlflow=hpo_cfg.get("log_trials_to_mlflow", analysis_cfg.get("log_to_mlflow", True)),
        log_ray_worker_output=hpo_cfg.get("log_ray_worker_output", True),
    )
    return best_config


def _resolve_data_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    provider_root = Path(__file__).resolve().parent.parent / "data_providers"
    return (provider_root / path).resolve()


def _resolve_hpo_split_dates(base_dp_cfg: dict[str, Any], validation_period_days: int) -> tuple[str, str, str, str]:
    if validation_period_days <= 0:
        raise ValueError("hpo.validation_period_days must be a positive integer")

    start_raw = base_dp_cfg.get("start_date")
    end_raw = base_dp_cfg.get("end_date")
    start_ts = pd.to_datetime(start_raw).normalize() if start_raw else None
    end_ts = pd.to_datetime(end_raw).normalize() if end_raw else None

    if "path" in base_dp_cfg and (start_ts is None or end_ts is None):
        timestamp_df = pd.read_csv(_resolve_data_path(base_dp_cfg["path"]), usecols=["timestamp"])
        timestamps = pd.to_datetime(timestamp_df["timestamp"], utc=True, errors="coerce").dropna().dt.tz_convert(None)
        if timestamps.empty:
            raise ValueError("Could not infer data range from data_provider.path because no valid timestamps were found")
        if start_ts is None:
            start_ts = timestamps.min().normalize()
        if end_ts is None:
            end_ts = timestamps.max().normalize()

    if start_ts is None or end_ts is None:
        raise ValueError(
            "Split HPO requires either data_provider.start_date and data_provider.end_date, "
            "or a CSV data_provider.path with a timestamp column"
        )

    validation_start = end_ts - pd.Timedelta(days=validation_period_days - 1)
    training_end = validation_start - pd.Timedelta(days=1)
    if training_end < start_ts:
        raise ValueError(
            "validation_period_days leaves no training window. "
            f"Range={start_ts.date()}..{end_ts.date()} validation_period_days={validation_period_days}"
        )

    return (
        start_ts.strftime("%Y-%m-%d"),
        training_end.strftime("%Y-%m-%d"),
        validation_start.strftime("%Y-%m-%d"),
        end_ts.strftime("%Y-%m-%d"),
    )


def _run_backtest_analysis(
    *,
    backtest_cfg: dict[str, Any],
    alg_cfg: dict[str, Any],
    pf_cfg: dict[str, Any],
    dp_cfg: dict[str, Any],
    algorithm_class,
    portfolio_class,
    data_provider_class,
    order_manager_class,
    mlflow_client: MLflowClient | None = None,
    log_to_mlflow: bool = False,
    metric_prefix: str = "",
    artifact_prefix: str = "",
    parameters: dict[str, Any] | None = None,
    artifact_paths: list[str] | None = None,
) -> dict[str, Any]:
    history_length = alg_cfg.get("history_length", 0)
    al = algorithm_class(
        {k: v for k, v in alg_cfg.items() if k != "history_length"},
        history_length=history_length,
    )
    om = order_manager_class()
    dp = data_provider_class(dp_cfg)
    pf = portfolio_class(pf_cfg, om, backtest_cfg["starting_cash"], {}, True)

    sim = BacktestingEngine({"state_store": {"enabled": False}}, dp, al, om, pf)
    sim.run()

    analysis = AnalysisEngine(sim.pf, pf.om)
    return analysis.run_full_analysis(
        experiment_name=backtest_cfg["experiment_name"],
        run_name=backtest_cfg["run_name"],
        description=backtest_cfg["description"],
        parameters=parameters,
        tracking_uri=backtest_cfg.get("tracking_uri"),
        log_to_mlflow=log_to_mlflow,
        save_charts_locally=False,
        save_report_locally=False,
        tags=backtest_cfg.get("git_tags") or None,
        artifact_paths=artifact_paths,
        benchmark_paths=backtest_cfg.get("benchmark_paths") or None,
        metric_prefix=metric_prefix,
        artifact_prefix=artifact_prefix,
        mlflow_client=mlflow_client,
        start_new_run=False,
        log_parameters=bool(parameters),
    )


def _build_trial_configs(
    trial_config: dict[str, Any],
    *,
    base_al_cfg: dict[str, Any],
    base_pf_cfg: dict[str, Any],
    algorithm_param_keys: list[str],
    portfolio_param_keys: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    al_cfg = apply_tunable_config(base_al_cfg, trial_config, algorithm_param_keys)
    pf_cfg = apply_tunable_config(base_pf_cfg, trial_config, portfolio_param_keys)
    return al_cfg, pf_cfg


def _select_best_split_config(
    *,
    trial_summaries: list[dict[str, Any]],
    objective_metric: str,
    base_backtest_cfg: dict[str, Any],
    base_al_cfg: dict[str, Any],
    base_pf_cfg: dict[str, Any],
    val_dp_cfg: dict[str, Any],
    algorithm_class,
    portfolio_class,
    data_provider_class,
    order_manager_class,
    algorithm_param_keys: list[str],
    portfolio_param_keys: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float]:
    if objective_metric == "trn_annualized_return":
        best_trial = max(trial_summaries, key=lambda trial: trial["metric"])
        best_al_cfg, best_pf_cfg = _build_trial_configs(
            best_trial["config"],
            base_al_cfg=base_al_cfg,
            base_pf_cfg=base_pf_cfg,
            algorithm_param_keys=algorithm_param_keys,
            portfolio_param_keys=portfolio_param_keys,
        )
        return best_trial["config"], best_al_cfg, best_pf_cfg, float(best_trial["metric"])

    if objective_metric != "val_annualized_return":
        raise ValueError(
            "Split HPO objective_metric must be one of "
            "'val_annualized_return' or 'trn_annualized_return'"
        )

    scored_trials: list[tuple[float, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for trial in trial_summaries:
        al_cfg, pf_cfg = _build_trial_configs(
            trial["config"],
            base_al_cfg=base_al_cfg,
            base_pf_cfg=base_pf_cfg,
            algorithm_param_keys=algorithm_param_keys,
            portfolio_param_keys=portfolio_param_keys,
        )
        val_results = _run_backtest_analysis(
            backtest_cfg=base_backtest_cfg,
            alg_cfg=al_cfg,
            pf_cfg=pf_cfg,
            dp_cfg=val_dp_cfg,
            algorithm_class=algorithm_class,
            portfolio_class=portfolio_class,
            data_provider_class=data_provider_class,
            order_manager_class=order_manager_class,
            log_to_mlflow=False,
        )
        score = float(metric_value(val_results["metrics"], "annualized_return"))
        scored_trials.append((score, trial["config"], al_cfg, pf_cfg))

    best_score, best_config, best_al_cfg, best_pf_cfg = max(scored_trials, key=lambda item: item[0])
    return best_config, best_al_cfg, best_pf_cfg, best_score


def cmd_hpo_split(args: argparse.Namespace):
    raw_cfg = load_raw_config(args.config)
    raw_cfg = apply_cli_overrides(raw_cfg, args)
    apply_session_log_file(raw_cfg, args)
    creds = load_account_creds(args.account)
    _fill_hpo_data_provider_creds(raw_cfg, creds)

    hpo_cfg = raw_cfg.setdefault("hpo", {})
    if getattr(args, "num_samples", None) is not None:
        hpo_cfg["num_samples"] = args.num_samples
    if getattr(args, "max_concurrent_trials", None) is not None:
        hpo_cfg["max_concurrent_trials"] = args.max_concurrent_trials
    if getattr(args, "validation_period_days", None) is not None:
        hpo_cfg["validation_period_days"] = args.validation_period_days

    validation_period_days = int(hpo_cfg.get("validation_period_days", 0))
    if validation_period_days <= 0:
        raise ValueError("Split HPO requires hpo.validation_period_days > 0")
    objective_metric = str(hpo_cfg.get("objective_metric", "val_annualized_return"))

    experiment = build_experiment_config(raw_cfg)
    config_dict = experiment.model_dump(exclude_none=True)
    if experiment.data_provider is None:
        raise ValueError("Split HPO requires a data_provider section.")

    al_class = import_component_class(experiment.algorithm)
    pf_class = import_component_class(experiment.portfolio)
    dp_class = import_component_class(experiment.data_provider)
    om_class = import_component_class(experiment.order_manager)

    base_al_cfg = dict(experiment.algorithm.params)
    history_length = base_al_cfg.pop("history_length", 0)
    if history_length:
        base_al_cfg["history_length"] = history_length
    base_pf_cfg = dict(experiment.portfolio.params)
    starting_cash = float(base_pf_cfg.get("cash", 10000.0))
    base_pf_cfg = {k: v for k, v in base_pf_cfg.items() if k not in ("cash", "keep_history")}
    base_dp_cfg = dict(experiment.data_provider.params)

    train_start, train_end, val_start, val_end = _resolve_hpo_split_dates(base_dp_cfg, validation_period_days)
    train_dp_cfg = copy.deepcopy(base_dp_cfg)
    train_dp_cfg["start_date"] = train_start
    train_dp_cfg["end_date"] = train_end
    val_dp_cfg = copy.deepcopy(base_dp_cfg)
    val_dp_cfg["start_date"] = val_start
    val_dp_cfg["end_date"] = val_end

    analysis_cfg = config_dict.get("analysis", {})
    base_backtest_cfg = {
        "starting_cash": starting_cash,
        "experiment_name": analysis_cfg.get("experiment_name", "HPO"),
        "run_name": analysis_cfg.get("run_name", "HPO_Run"),
        "description": analysis_cfg.get("description", ""),
        "symbol": base_pf_cfg.get("symbol") or base_pf_cfg.get("upro_symbol", ""),
        "tracking_uri": config_dict.get("mlflow", {}).get("tracking_uri"),
        "git_tags": get_git_info(),
        "benchmark_paths": analysis_cfg.get("benchmarks") or {},
    }
    base_backtest_cfg.update(flatten_config(config_dict))

    logger.info(
        "Starting split HPO: training=%s..%s validation=%s..%s validation_period_days=%s",
        train_start,
        train_end,
        val_start,
        val_end,
        validation_period_days,
    )

    from trading.launchers.run_backtest_ray import tune_backtest_hyperparameters

    _, trial_summaries = tune_backtest_hyperparameters(
        symbol=base_backtest_cfg["symbol"],
        algorithm_class=al_class,
        portfolio_class=pf_class,
        data_provider_class=dp_class,
        order_manager_class=om_class,
        base_algorithm_config=base_al_cfg,
        base_portfolio_config=base_pf_cfg,
        base_data_provider_config=train_dp_cfg,
        base_backtest_config=base_backtest_cfg,
        search_space=parse_search_space(hpo_cfg.get("search_space", {})),
        algorithm_param_keys=hpo_cfg.get("algorithm_param_keys", []),
        portfolio_param_keys=hpo_cfg.get("portfolio_param_keys", []),
        num_samples=hpo_cfg.get("num_samples", 50),
        max_concurrent_trials=hpo_cfg.get("max_concurrent_trials", 8),
        log_to_mlflow=hpo_cfg.get("log_trials_to_mlflow", analysis_cfg.get("log_to_mlflow", True)),
        log_ray_worker_output=hpo_cfg.get("log_ray_worker_output", True),
        return_trial_summaries=True,
    )
    best_config, best_al_cfg, best_pf_cfg, selected_objective_value = _select_best_split_config(
        trial_summaries=trial_summaries,
        objective_metric=objective_metric,
        base_backtest_cfg=base_backtest_cfg,
        base_al_cfg=base_al_cfg,
        base_pf_cfg=base_pf_cfg,
        val_dp_cfg=val_dp_cfg,
        algorithm_class=al_class,
        portfolio_class=pf_class,
        data_provider_class=dp_class,
        order_manager_class=om_class,
        algorithm_param_keys=hpo_cfg.get("algorithm_param_keys", []),
        portfolio_param_keys=hpo_cfg.get("portfolio_param_keys", []),
    )
    run_parameters = base_backtest_cfg | best_al_cfg | best_pf_cfg | base_dp_cfg | {
        "hpo.validation_period_days": validation_period_days,
        "hpo.objective_metric": objective_metric,
        "hpo.train_start_date": train_start,
        "hpo.train_end_date": train_end,
        "hpo.val_start_date": val_start,
        "hpo.val_end_date": val_end,
    }
    artifact_paths = _collect_config_artifact_paths(raw_cfg, config_path=args.config)

    if analysis_cfg.get("log_to_mlflow", True):
        mlflow_client = MLflowClient(
            experiment_name=base_backtest_cfg["experiment_name"],
            tracking_uri=base_backtest_cfg.get("tracking_uri"),
        )
        with mlflow_client.start_run(
            run_name=base_backtest_cfg["run_name"],
            description=base_backtest_cfg["description"],
            tags=base_backtest_cfg["git_tags"] or None,
        ):
            mlflow_client.log_json(best_config, "hpo_best_config.json")
            train_results = _run_backtest_analysis(
                backtest_cfg=base_backtest_cfg,
                alg_cfg=best_al_cfg,
                pf_cfg=best_pf_cfg,
                dp_cfg=train_dp_cfg,
                algorithm_class=al_class,
                portfolio_class=pf_class,
                data_provider_class=dp_class,
                order_manager_class=om_class,
                mlflow_client=mlflow_client,
                log_to_mlflow=True,
                metric_prefix="trn_",
                parameters=run_parameters,
                artifact_paths=artifact_paths,
            )
            val_results = _run_backtest_analysis(
                backtest_cfg=base_backtest_cfg,
                alg_cfg=best_al_cfg,
                pf_cfg=best_pf_cfg,
                dp_cfg=val_dp_cfg,
                algorithm_class=al_class,
                portfolio_class=pf_class,
                data_provider_class=dp_class,
                order_manager_class=om_class,
                mlflow_client=mlflow_client,
                log_to_mlflow=True,
                metric_prefix="val_",
                artifact_prefix="val_",
                artifact_paths=artifact_paths,
            )
    else:
        train_results = _run_backtest_analysis(
            backtest_cfg=base_backtest_cfg,
            alg_cfg=best_al_cfg,
            pf_cfg=best_pf_cfg,
            dp_cfg=train_dp_cfg,
            algorithm_class=al_class,
            portfolio_class=pf_class,
            data_provider_class=dp_class,
            order_manager_class=om_class,
        )
        val_results = _run_backtest_analysis(
            backtest_cfg=base_backtest_cfg,
            alg_cfg=best_al_cfg,
            pf_cfg=best_pf_cfg,
            dp_cfg=val_dp_cfg,
            algorithm_class=al_class,
            portfolio_class=pf_class,
            data_provider_class=dp_class,
            order_manager_class=om_class,
        )

    logger.info("Split HPO complete. Best config:")
    for key, val in best_config.items():
        logger.info(f"  {key}: {val}")
    logger.info("Selected objective metric: %s=%.4f", objective_metric, selected_objective_value)
    logger.info(
        "Final metrics: trn_annualized_return=%.4f val_annualized_return=%.4f",
        train_results["metrics"].annualized_return,
        val_results["metrics"].annualized_return,
    )


def cmd_hpo(args: argparse.Namespace):
    raw_cfg = load_raw_config(args.config)
    raw_cfg = apply_cli_overrides(raw_cfg, args)
    apply_session_log_file(raw_cfg, args)
    creds = load_account_creds(args.account)
    _fill_hpo_data_provider_creds(raw_cfg, creds)

    logger.info(f"Starting HPO with profile: {args.config}")
    best_config = run_hpo_from_raw_config(
        raw_cfg,
        config_artifact_path=args.config,
        num_samples_override=getattr(args, "num_samples", None),
        max_concurrent_override=getattr(args, "max_concurrent_trials", None),
    )

    logger.info("HPO complete. Best config:")
    for key, val in best_config.items():
        logger.info(f"  {key}: {val}")
