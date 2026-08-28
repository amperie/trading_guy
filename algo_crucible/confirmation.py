from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError
import yaml

from algo_crucible.builders import build_candidate_from_params
from algo_crucible.ids import hash16
from algo_crucible.models import Candidate


def load_confirmation_candidates(run_dir: str | Path, resolved_cfg) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    perturbation_path = _existing_path(run_dir, "stages/07_perturbation/summaries/perturbation_summary.csv", "summaries/perturbation_summary.csv")
    hpo_path = _existing_path(run_dir, "stages/04_hpo/summaries/hpo_trial_summary.csv", "summaries/hpo_trial_summary.csv")
    if not perturbation_path.exists():
        raise FileNotFoundError("run_perturbation_stage must produce perturbation_summary.csv before confirmation can run")
    if not hpo_path.exists():
        raise FileNotFoundError("run_hpo_stage must produce hpo_trial_summary.csv before confirmation can run")

    accepted = {
        str(row["candidate_id"]): row
        for row in _read_csv_rows(perturbation_path)
        if str(row.get("accepted")).lower() in {"true", "1"}
    }
    candidates = []
    for row in _read_csv_rows(hpo_path):
        candidate_id = str(row.get("candidate_id"))
        if candidate_id not in accepted:
            continue
        candidate = build_candidate_from_params(
            resolved_cfg,
            _decode(row.get("algorithm_params")) or {},
            _decode(row.get("portfolio_params")) or {},
        )
        candidates.append({
            "candidate": candidate,
            "seed_id": accepted[candidate_id].get("seed_id"),
            "candidate_type": accepted[candidate_id].get("candidate_type", "generalist"),
            "specialist_regimes": accepted[candidate_id].get("specialist_regimes", ""),
        })
    return candidates


def confirmation_window(platform: dict[str, Any]) -> dict[str, Any]:
    cfg = platform.get("confirmation", {})
    start = cfg.get("start_date")
    end = cfg.get("end_date")
    if not start or not end:
        raise ValueError("confirmation.start_date and confirmation.end_date are required")
    return {
        "window_id": "confirmation_000",
        "train_start": "",
        "train_end": "",
        "embargo_days": 0,
        "validation_start": str(start),
        "validation_end": str(end),
        "confirmation": True,
    }


def confirmation_workload(resolved_cfg, window: dict[str, Any]) -> dict[str, Any]:
    workload = copy.deepcopy(resolved_cfg.workload)
    workload["data_provider"]["start_date"] = window["validation_start"]
    workload["data_provider"]["end_date"] = window["validation_end"]
    return workload


def freeze_candidate(candidate: Candidate, resolved_cfg) -> dict[str, Any]:
    payload = {
        "candidate": candidate.to_dict(),
        "crucible_run_id": resolved_cfg.crucible_run_id,
        "resolved_config_hash": resolved_cfg.resolved_config_hash,
        "frozen_config_hash": hash16(candidate.to_dict()),
        "tuning_locked": True,
    }
    return payload


def summarize_confirmation(candidates: list[dict[str, Any]], job_results: list[dict[str, Any]], platform: dict[str, Any]) -> list[dict[str, Any]]:
    by_candidate = {}
    for result in job_results:
        if result.get("status") == "complete":
            payload = result["result"]
            by_candidate[payload["candidate_id"]] = payload
    rows = []
    for item in candidates:
        candidate = item["candidate"]
        payload = by_candidate.get(candidate.candidate_id)
        scorecard = payload.get("overall_scorecard", {}) if payload else {}
        confirmed, reason = _passes_confirmation(scorecard, item, platform) if payload else (False, "confirmation_job_failed")
        rows.append({
            "candidate_id": candidate.candidate_id,
            "seed_id": item.get("seed_id"),
            "candidate_type": item.get("candidate_type", "generalist"),
            "specialist_regimes": item.get("specialist_regimes", ""),
            "confirmed": confirmed,
            "failure_reason": "" if confirmed else reason,
            **{key: scorecard.get(key) for key in (
                "total_return_pct",
                "annualized_return",
                "max_drawdown_pct",
                "volatility",
                "sharpe_ratio",
                "sortino_ratio",
                "total_trades",
            )},
        })
    return rows


def confirmation_metrics(rows: list[dict[str, Any]], jobs_total: int, jobs_complete: int, jobs_failed: int) -> dict[str, float]:
    returns = [_num(row.get("total_return_pct")) for row in rows]
    returns = [value for value in returns if value is not None]
    return {
        "confirmation.jobs_total": float(jobs_total),
        "confirmation.jobs_complete": float(jobs_complete),
        "confirmation.jobs_failed": float(jobs_failed),
        "confirmation.candidates_total": float(len(rows)),
        "confirmation.promoted_candidates": float(sum(1 for row in rows if row.get("confirmed") is True)),
        "confirmation.rejected_candidates": float(sum(1 for row in rows if row.get("confirmed") is not True)),
        "confirmation.best_return_pct": max(returns, default=0.0),
    }


def build_promotion_packet(
    *,
    resolved_cfg,
    confirmation_rows: list[dict[str, Any]],
    frozen_candidates: list[dict[str, Any]],
    artifact_paths: dict[str, str],
    mlflow_run_url: str | None = None,
) -> dict[str, Any]:
    confirmed = [row for row in confirmation_rows if row.get("confirmed") is True]
    return {
        "schema_version": 1,
        "decision": "promote_to_paper" if confirmed else "reject",
        "paper_trading_started": False,
        "approval_required": True,
        "crucible": {
            "run_name": resolved_cfg.run_name,
            "crucible_run_id": resolved_cfg.crucible_run_id,
            "resolved_config_hash": resolved_cfg.resolved_config_hash,
            "mlflow_run_url": mlflow_run_url,
        },
        "confirmed_candidate_ids": [row["candidate_id"] for row in confirmed],
        "confirmation": {
            "window": confirmation_window(resolved_cfg.platform),
            "rows": confirmation_rows,
        },
        "frozen_candidates": frozen_candidates,
        "artifacts": artifact_paths,
        "next_action": "manual_approval_required_before_paper_trading",
    }


def packet_markdown(packet: dict[str, Any]) -> str:
    rows = packet["confirmation"]["rows"]
    lines = [
        "# Algo Crucible Promotion Packet",
        "",
        f"- Decision: `{packet['decision']}`",
        f"- Crucible run: `{packet['crucible']['crucible_run_id']}`",
        f"- Resolved config hash: `{packet['crucible']['resolved_config_hash']}`",
        f"- Paper trading started: `{packet['paper_trading_started']}`",
        f"- Approval required: `{packet['approval_required']}`",
        "",
        "## Confirmed Candidates",
        "",
    ]
    if packet["confirmed_candidate_ids"]:
        lines.extend(f"- `{candidate_id}`" for candidate_id in packet["confirmed_candidate_ids"])
    else:
        lines.append("- None")
    lines.extend(["", "## Confirmation Results", ""])
    for row in rows:
        lines.append(
            f"- `{row['candidate_id']}` confirmed={row['confirmed']} "
            f"return={row.get('total_return_pct')} drawdown={row.get('max_drawdown_pct')} "
            f"reason=`{row.get('failure_reason', '')}`"
        )
    return "\n".join(lines) + "\n"


def write_promoted_packet(packet: dict[str, Any], platform: dict[str, Any]) -> dict[str, str]:
    promotion_cfg = platform.get("promotion", {})
    root = Path(promotion_cfg.get("output_dir", "trading/promoted"))
    name = _slug(promotion_cfg.get("name") or packet["crucible"]["crucible_run_id"])
    promoted_dir = root / name
    promoted_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "promotion_packet_json": promoted_dir / "promotion_packet.json",
        "promotion_packet_yaml": promoted_dir / "promotion_packet.yaml",
        "promotion_packet_md": promoted_dir / "promotion_packet.md",
    }
    paths["promotion_packet_json"].write_text(json.dumps(packet, indent=2, sort_keys=True, default=str), encoding="utf-8")
    paths["promotion_packet_yaml"].write_text(yaml.safe_dump(packet, sort_keys=True), encoding="utf-8")
    paths["promotion_packet_md"].write_text(packet_markdown(packet), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def _passes_confirmation(scorecard: dict[str, Any], candidate: dict[str, Any], platform: dict[str, Any]) -> tuple[bool, str]:
    cfg = platform.get("confirmation", {})
    gates = platform.get("gates", {}).get("specialist" if candidate.get("candidate_type") == "specialist" else "generalist", {})
    min_return = _pct_threshold(cfg.get("min_return_pct", gates.get("min_median_oos_return", 0.0)))
    max_drawdown = abs(_pct_threshold(cfg.get("max_drawdown_pct", gates.get("max_drawdown", 100.0))))
    min_trades = float(cfg.get("min_trades", gates.get("min_trades", 0)))
    total_return = _num(scorecard.get("total_return_pct"))
    if total_return is None:
        return False, "confirmation_missing_return"
    if total_return < min_return:
        return False, "confirmation_return_below_gate"
    drawdown = _num(scorecard.get("max_drawdown_pct"))
    if drawdown is not None and abs(drawdown) > max_drawdown:
        return False, "confirmation_drawdown_above_gate"
    total_trades = _num(scorecard.get("total_trades"))
    if total_trades is None:
        return False, "confirmation_missing_trade_count"
    if total_trades < min_trades:
        return False, "confirmation_trade_count_below_gate"
    return True, ""


def _decode(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return ast.literal_eval(str(value))


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return pd.read_csv(path).to_dict(orient="records")
    except EmptyDataError:
        return []


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pct_threshold(value: Any) -> float:
    value = float(value)
    return value * 100.0 if abs(value) <= 1.0 else value


def _slug(value: Any) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value).lower()).strip("_") or "promotion"


def _existing_path(run_dir: Path, *relative_paths: str) -> Path:
    for relative_path in relative_paths:
        path = run_dir / relative_path
        if path.exists():
            return path
    return run_dir / relative_paths[0]
