from __future__ import annotations

import copy
import os
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from trading.core.algorithm import Algorithm
from trading.core.om.order_manager import OrderManager
from trading.core.portfolio import Portfolio
from trading.data_providers.data_provider import DataProvider
from trading.engines.walk_forward_engine import WalkForwardEngine
from utils.logger import Logger

logger = Logger().get_logger(__name__)


WINDOW_KEYS = (
    "optimization_window_days",
    "validation_window_days",
    "trading_window_days",
)


@dataclass(slots=True)
class WindowHPOCandidate:
    trial_number: int
    metric: float
    windows: dict[str, int]
    mlflow_run_id: str | None
    num_periods: int


class WalkForwardWindowHPO:
    """Outer HPO over walk-forward window sizes.

    The existing walk-forward engine already runs inner strategy HPO inside
    each optimization window. This optimizer samples only the three schedule
    sizes and lets each candidate execute a normal walk-forward run.
    """

    def __init__(
        self,
        *,
        engine_cfg: dict[str, Any],
        dp: DataProvider,
        al: Algorithm,
        om: OrderManager,
        pf: Portfolio,
    ) -> None:
        self.engine_cfg = copy.deepcopy(engine_cfg)
        self.window_hpo_cfg = copy.deepcopy(engine_cfg.get("walk_forward_window_hpo", {}))

        self._dp_class = type(dp)
        self._al_class = type(al)
        self._om_class = type(om)
        self._pf_class = type(pf)

        self._base_dp_cfg = copy.deepcopy(dp.cfg)
        self._base_al_cfg = copy.deepcopy(getattr(al, "cfg", {}))
        self._base_pf_cfg = copy.deepcopy(getattr(pf, "cfg", {}))
        self._history_length = getattr(al, "history_length", 0)
        self._keep_history = getattr(pf, "keep_history", True)
        self._starting_cash = getattr(pf, "cash", 0.0)

        self.group_id = self.window_hpo_cfg.get("group_id") or uuid.uuid4().hex[:12]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_staging_name = f"wf_window_hpo_tmp_{timestamp}_{self.group_id}"
        self.staging_experiment_name = self.window_hpo_cfg.get(
            "staging_experiment_name",
            default_staging_name,
        )
        self.final_experiment_name = self.window_hpo_cfg.get(
            "final_experiment_name",
            self.engine_cfg.get("experiment_name", "Walk Forward Backtest"),
        )
        self.objective_metric = self.window_hpo_cfg.get("objective_metric", "wf_annualized_return")
        self.num_samples = int(self.window_hpo_cfg.get("num_samples", 10))
        self.max_concurrent_trials = int(self.window_hpo_cfg.get("max_concurrent_trials", 1))
        self.min_periods = int(self.window_hpo_cfg.get("min_periods", 1))
        self.cleanup_staging_experiment = bool(self.window_hpo_cfg.get("cleanup_staging_experiment", False))
        self.run_mlflow_gc = bool(self.window_hpo_cfg.get("run_mlflow_gc", False))
        self.cleanup_s3_prefix = bool(self.window_hpo_cfg.get("cleanup_s3_prefix", False))
        self.staging_artifact_location = self.window_hpo_cfg.get("staging_artifact_location")
        self.final_artifact_location = self.window_hpo_cfg.get("final_artifact_location")

        self.candidates: list[WindowHPOCandidate] = []

    def _build_components(self):
        dp = self._dp_class(copy.deepcopy(self._base_dp_cfg))
        al = self._al_class(copy.deepcopy(self._base_al_cfg), history_length=self._history_length)
        om = self._om_class()
        pf = self._pf_class(copy.deepcopy(self._base_pf_cfg), om, self._starting_cash, {}, self._keep_history)
        return dp, al, om, pf

    def _sample_windows(self, trial) -> dict[str, int]:
        search_space = self.window_hpo_cfg.get("search_space", {})
        missing = [key for key in WINDOW_KEYS if key not in search_space]
        if missing:
            raise ValueError(f"walk_forward_window_hpo.search_space missing keys: {missing}")

        return {key: int(self._sample_value(trial, key, search_space[key])) for key in WINDOW_KEYS}

    def _sample_value(self, trial, key: str, spec: dict[str, Any]) -> Any:
        kind = spec["type"]
        if kind == "choice":
            return trial.suggest_categorical(key, spec["values"])
        if kind == "randint":
            return trial.suggest_int(key, int(spec["low"]), int(spec["high"]) - 1)
        if kind == "uniform":
            return trial.suggest_float(key, float(spec["low"]), float(spec["high"]))
        if kind == "loguniform":
            return trial.suggest_float(key, float(spec["low"]), float(spec["high"]), log=True)
        raise ValueError(f"Unknown window HPO search space type '{kind}' for key '{key}'")

    def _metric_from_result(self, result: dict[str, Any]) -> float:
        aggregate = result.get("aggregate", {})
        if self.objective_metric in aggregate:
            return float(aggregate.get(self.objective_metric) or 0.0)

        metrics = result.get("metrics")
        return float(getattr(metrics, self.objective_metric, 0.0) or 0.0)

    def _engine_cfg_for_candidate(self, windows: dict[str, int], trial_number: int) -> dict[str, Any]:
        cfg = copy.deepcopy(self.engine_cfg)
        cfg["experiment_name"] = self.staging_experiment_name
        cfg["run_name"] = (
            f"wf_window_candidate_{trial_number}_"
            f"opt{windows['optimization_window_days']}_"
            f"val{windows['validation_window_days']}_"
            f"trade{windows['trading_window_days']}"
        )
        cfg["log_to_mlflow"] = bool(self.engine_cfg.get("log_to_mlflow", True))
        if self.staging_artifact_location:
            cfg["artifact_location"] = self.staging_artifact_location
        cfg.setdefault("walk_forward", {}).update(windows)
        cfg["mlflow_tags"] = {
            **cfg.get("mlflow_tags", {}),
            "run_type": "walk_forward_window_candidate",
            "window_hpo_group_id": self.group_id,
            **{key: str(value) for key, value in windows.items()},
        }
        return cfg

    def _engine_cfg_for_winner(self, windows: dict[str, int]) -> dict[str, Any]:
        cfg = copy.deepcopy(self.engine_cfg)
        cfg["experiment_name"] = self.final_experiment_name
        cfg["run_name"] = (
            "wf_window_winner_"
            f"opt{windows['optimization_window_days']}_"
            f"val{windows['validation_window_days']}_"
            f"trade{windows['trading_window_days']}"
        )
        cfg["log_to_mlflow"] = bool(self.engine_cfg.get("log_to_mlflow", True))
        if self.final_artifact_location:
            cfg["artifact_location"] = self.final_artifact_location
        cfg.setdefault("walk_forward", {}).update(windows)
        cfg["mlflow_tags"] = {
            **cfg.get("mlflow_tags", {}),
            "run_type": "walk_forward_window_winner",
            "window_hpo_group_id": self.group_id,
            **{key: str(value) for key, value in windows.items()},
        }
        return cfg

    def _run_walk_forward(self, cfg: dict[str, Any]) -> dict[str, Any]:
        dp, al, om, pf = self._build_components()
        return WalkForwardEngine(cfg=cfg, dp=dp, al=al, om=om, pf=pf).run()

    def _objective(self, trial) -> float:
        windows = self._sample_windows(trial)
        logger.info(
            "Running walk-forward window candidate "
            f"{trial.number + 1}/{self.num_samples}: {windows}",
            color="cyan",
        )

        result = self._run_walk_forward(self._engine_cfg_for_candidate(windows, trial.number))
        metric = self._metric_from_result(result)
        num_periods = len(result.get("periods", []))
        if num_periods < self.min_periods:
            logger.warning(
                f"Candidate has only {num_periods} periods; min_periods={self.min_periods}. Penalizing metric."
            )
            metric = -1e18

        candidate = WindowHPOCandidate(
            trial_number=trial.number,
            metric=metric,
            windows=windows,
            mlflow_run_id=result.get("mlflow_run_id"),
            num_periods=num_periods,
        )
        self.candidates.append(candidate)
        trial.set_user_attr("mlflow_run_id", candidate.mlflow_run_id)
        trial.set_user_attr("num_periods", num_periods)
        for key, value in windows.items():
            trial.set_user_attr(key, value)
        return metric

    def run(self) -> dict[str, Any]:
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("walk-forward-hpo requires optuna. Install it with: pip install optuna") from exc

        if self.max_concurrent_trials > 1:
            logger.warning(
                "walk-forward-hpo currently runs outer candidates sequentially so nested per-period Ray HPO remains stable. "
                f"Ignoring max_concurrent_trials={self.max_concurrent_trials}."
            )

        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.num_samples, n_jobs=1)

        best_windows = {key: int(study.best_trial.params[key]) for key in WINDOW_KEYS}
        logger.info(f"Best walk-forward windows: {best_windows} metric={study.best_value:.6f}", color="green")

        final_result = self._run_walk_forward(self._engine_cfg_for_winner(best_windows))
        cleanup = self._cleanup_staging()

        return {
            "best_windows": best_windows,
            "best_metric": study.best_value,
            "best_trial_number": study.best_trial.number,
            "final_result": final_result,
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "cleanup": cleanup,
        }

    def _cleanup_staging(self) -> dict[str, Any]:
        result = {
            "deleted_experiment": False,
            "ran_mlflow_gc": False,
            "deleted_s3_prefix": False,
        }
        if not self.cleanup_staging_experiment:
            return result

        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=self.engine_cfg.get("tracking_uri"))
            experiment = client.get_experiment_by_name(self.staging_experiment_name)
            if experiment is not None:
                client.delete_experiment(experiment.experiment_id)
                result["deleted_experiment"] = True
                if self.run_mlflow_gc:
                    result["ran_mlflow_gc"] = self._run_mlflow_gc(experiment.experiment_id)
        except Exception as exc:
            logger.warning(f"Failed to delete staging MLflow experiment: {exc}")

        if self.cleanup_s3_prefix and self.staging_artifact_location:
            result["deleted_s3_prefix"] = self._delete_s3_prefix(self.staging_artifact_location)
        return result

    def _run_mlflow_gc(self, experiment_id: str) -> bool:
        mlflow_bin = shutil.which("mlflow")
        if not mlflow_bin:
            logger.warning("Cannot run mlflow gc because the mlflow executable was not found")
            return False

        cmd = [mlflow_bin, "gc", "--experiment-ids", str(experiment_id)]
        tracking_uri = self.engine_cfg.get("tracking_uri")
        env = os.environ.copy()
        if tracking_uri:
            env["MLFLOW_TRACKING_URI"] = tracking_uri
        if tracking_uri and not str(tracking_uri).startswith(("http://", "https://")):
            cmd.extend(["--backend-store-uri", tracking_uri])
        try:
            subprocess.run(cmd, check=True, env=env)
            return True
        except Exception as exc:
            logger.warning(f"mlflow gc failed for staging experiment {experiment_id}: {exc}")
            return False

    def _delete_s3_prefix(self, artifact_location: str) -> bool:
        if not artifact_location.startswith("s3://"):
            logger.warning(f"cleanup_s3_prefix ignored for non-S3 artifact location: {artifact_location}")
            return False

        aws_bin = shutil.which("aws")
        if not aws_bin:
            logger.warning("Cannot delete S3 prefix because the aws executable was not found")
            return False

        try:
            subprocess.run([aws_bin, "s3", "rm", artifact_location, "--recursive"], check=True)
            return True
        except Exception as exc:
            logger.warning(f"Failed to delete staging S3 prefix {artifact_location}: {exc}")
            return False
