from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


def return_stream_rows(job_results: list[dict[str, Any]], accepted_candidate_ids: set[str]) -> list[dict[str, Any]]:
    rows = []
    for result in job_results:
        if result.get("status") != "complete":
            continue
        payload = result.get("result") or {}
        candidate_id = str(payload.get("source_candidate_id") or payload.get("candidate_id") or "")
        if candidate_id not in accepted_candidate_ids:
            continue
        for item in payload.get("return_stream") or []:
            rows.append({
                "candidate_id": candidate_id,
                "scenario_id": payload.get("scenario_id", ""),
                "window_id": payload.get("window_id", ""),
                "step": item.get("step"),
                "timestamp": item.get("timestamp"),
                "return_pct": item.get("return_pct"),
                "equity": item.get("equity"),
            })
    return rows


def load_monte_carlo_inputs(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "stages/07_perturbation/summaries/perturbation_return_stream.csv"
    return _read_csv_rows(path)


def simulate_monte_carlo(rows: list[dict[str, Any]], platform: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    cfg = platform.get("monte_carlo", {})
    num_paths = int(cfg.get("num_paths", 1000))
    max_path_points = int(cfg.get("max_path_points", 252))
    min_observations = int(cfg.get("min_observations", 20))
    seed = int(cfg.get("seed", 42))
    ruin_threshold_pct = float(cfg.get("ruin_threshold_pct", -50.0))
    max_ruin_probability = _probability_threshold(cfg.get("max_ruin_probability", 0.05))
    max_loss_probability = _probability_threshold(cfg.get("max_loss_probability", 0.40))
    min_p5_terminal_return_pct = float(cfg.get("min_p5_terminal_return_pct", -25.0))
    max_p95_drawdown_pct = abs(float(cfg.get("max_p95_drawdown_pct", 35.0)))

    summary_rows = []
    band_rows = []
    terminal_rows = []
    grouped = _group_returns(rows)
    for offset, (candidate_id, returns) in enumerate(sorted(grouped.items())):
        path_length = min(int(cfg.get("path_length", len(returns))), len(returns), max_path_points)
        if len(returns) < min_observations or path_length <= 0:
            summary_rows.append(_insufficient_row(candidate_id, len(returns), min_observations))
            continue
        paths = _bootstrap_paths(
            returns,
            num_paths=num_paths,
            path_length=path_length,
            seed=seed + offset,
        )
        terminals = [path[-1] - 100.0 for path in paths]
        drawdowns = [_max_drawdown_pct(path) for path in paths]
        ruin_probability = _pct([min(path) - 100.0 <= ruin_threshold_pct for path in paths])
        loss_probability = _pct([terminal < 0.0 for terminal in terminals])
        terminal_p5 = _quantile(terminals, 0.05)
        drawdown_p95 = abs(_quantile(drawdowns, 0.95))
        accepted = (
            ruin_probability <= max_ruin_probability
            and loss_probability <= max_loss_probability
            and terminal_p5 >= min_p5_terminal_return_pct
            and drawdown_p95 <= max_p95_drawdown_pct
        )
        summary_rows.append({
            "candidate_id": candidate_id,
            "accepted": accepted,
            "observation_count": len(returns),
            "path_count": num_paths,
            "path_length": path_length,
            "terminal_return_p5": terminal_p5,
            "terminal_return_p50": _quantile(terminals, 0.50),
            "terminal_return_p95": _quantile(terminals, 0.95),
            "max_drawdown_p50": abs(_quantile(drawdowns, 0.50)),
            "max_drawdown_p95": drawdown_p95,
            "loss_probability": loss_probability,
            "ruin_probability": ruin_probability,
            "failure_reason": "" if accepted else _failure_reason(
                ruin_probability=ruin_probability,
                loss_probability=loss_probability,
                terminal_p5=terminal_p5,
                drawdown_p95=drawdown_p95,
                cfg=cfg,
            ),
        })
        terminal_rows.extend(
            {"candidate_id": candidate_id, "path_id": idx, "terminal_return_pct": terminal, "max_drawdown_pct": abs(drawdowns[idx])}
            for idx, terminal in enumerate(terminals)
        )
        band_rows.extend(_band_rows(candidate_id, paths))
    return {"summary_rows": summary_rows, "band_rows": band_rows, "terminal_rows": terminal_rows}


def monte_carlo_metrics(summary_rows: list[dict[str, Any]]) -> dict[str, float]:
    accepted = [row for row in summary_rows if row.get("accepted") is True]
    p5_values = [_num(row.get("terminal_return_p5")) for row in summary_rows]
    ruin_values = [_num(row.get("ruin_probability")) for row in summary_rows]
    drawdown_values = [_num(row.get("max_drawdown_p95")) for row in summary_rows]
    return {
        "monte_carlo.candidate_count": float(len(summary_rows)),
        "monte_carlo.accepted_candidates": float(len(accepted)),
        "monte_carlo.rejected_candidates": float(len(summary_rows) - len(accepted)),
        "monte_carlo.best_terminal_return_p5": max([value for value in p5_values if value is not None], default=0.0),
        "monte_carlo.worst_ruin_probability": max([value for value in ruin_values if value is not None], default=0.0),
        "monte_carlo.worst_drawdown_p95": max([value for value in drawdown_values if value is not None], default=0.0),
    }


def _group_returns(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        value = _num(row.get("return_pct"))
        if candidate_id and value is not None:
            grouped[candidate_id].append(value)
    return grouped


def _bootstrap_paths(returns: list[float], *, num_paths: int, path_length: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    paths = []
    for _ in range(num_paths):
        equity = 100.0
        path = []
        for _ in range(path_length):
            equity *= 1.0 + rng.choice(returns) / 100.0
            path.append(equity)
        paths.append(path)
    return paths


def _band_rows(candidate_id: str, paths: list[list[float]]) -> list[dict[str, Any]]:
    if not paths:
        return []
    rows = []
    for idx in range(len(paths[0])):
        values = [path[idx] for path in paths]
        rows.append({
            "candidate_id": candidate_id,
            "step": idx + 1,
            "p5": _quantile(values, 0.05),
            "p50": _quantile(values, 0.50),
            "p95": _quantile(values, 0.95),
        })
    return rows


def _max_drawdown_pct(path: list[float]) -> float:
    peak = path[0] if path else 100.0
    worst = 0.0
    for value in path:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, (value / peak - 1.0) * 100.0)
    return worst


def _failure_reason(
    *,
    ruin_probability: float,
    loss_probability: float,
    terminal_p5: float,
    drawdown_p95: float,
    cfg: dict[str, Any],
) -> str:
    reasons = []
    if ruin_probability > _probability_threshold(cfg.get("max_ruin_probability", 0.05)):
        reasons.append("ruin_probability_too_high")
    if loss_probability > _probability_threshold(cfg.get("max_loss_probability", 0.40)):
        reasons.append("loss_probability_too_high")
    if terminal_p5 < float(cfg.get("min_p5_terminal_return_pct", -25.0)):
        reasons.append("terminal_p5_too_low")
    if drawdown_p95 > abs(float(cfg.get("max_p95_drawdown_pct", 35.0))):
        reasons.append("drawdown_p95_too_high")
    return ",".join(reasons)


def _insufficient_row(candidate_id: str, count: int, minimum: int) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "accepted": False,
        "observation_count": count,
        "path_count": 0,
        "path_length": 0,
        "terminal_return_p5": None,
        "terminal_return_p50": None,
        "terminal_return_p95": None,
        "max_drawdown_p50": None,
        "max_drawdown_p95": None,
        "loss_probability": None,
        "ruin_probability": None,
        "failure_reason": f"insufficient_return_observations:{count}<min:{minimum}",
    }


def _pct(flags: list[bool]) -> float:
    return 100.0 * sum(1 for flag in flags if flag) / len(flags) if flags else 0.0


def _probability_threshold(value: Any) -> float:
    value = float(value)
    return value * 100.0 if abs(value) <= 1.0 else value


def _quantile(values: list[float], q: float) -> float:
    values = sorted(value for value in values if value is not None and math.isfinite(value))
    if not values:
        return 0.0
    idx = min(max(int((len(values) - 1) * q), 0), len(values) - 1)
    return float(values[idx])


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return pd.read_csv(path).to_dict(orient="records")
    except (EmptyDataError, FileNotFoundError):
        return []
