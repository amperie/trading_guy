from __future__ import annotations

import ast
import copy
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from algo_crucible.builders import build_candidate_from_params
from utils.utils import apply_tunable_config


def load_plateau_seeds(run_dir: str | Path, resolved_cfg, platform: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    hpo_path = _existing_path(run_dir, "stages/04_hpo/summaries/hpo_trial_summary.csv", "summaries/hpo_trial_summary.csv")
    if not hpo_path.exists():
        raise FileNotFoundError("run_hpo_stage must produce hpo_trial_summary.csv before plateau analysis can run")
    rows = pd.read_csv(hpo_path).to_dict(orient="records")
    accepted = _accepted_candidates(run_dir)
    hpo_candidate_ids = {str(row.get("candidate_id")) for row in rows}
    if accepted and hpo_candidate_ids & accepted:
        rows = [row for row in rows if str(row.get("candidate_id")) in accepted]
    rows.sort(key=lambda row: _num(row.get("metric")) or float("-inf"), reverse=True)

    cfg = platform.get("plateau", {})
    max_seeds = int(cfg.get("max_seeds", 5))
    min_distance = float(cfg.get("min_seed_distance", 0.15))
    space = _search_space(resolved_cfg)
    selected = []
    for row in rows:
        trial_config = _decode(row.get("config")) or {}
        if any(normalized_distance(trial_config, seed["center_config"], space) < min_distance for seed in selected):
            continue
        seed_id = f"seed_{len(selected) + 1:03d}"
        selected.append({
            "seed_id": seed_id,
            "candidate_id": row["candidate_id"],
            "candidate_type": _gate_meta(run_dir, row["candidate_id"]).get("candidate_type", "unknown"),
            "specialist_regimes": _gate_meta(run_dir, row["candidate_id"]).get("specialist_regimes", ""),
            "center_config": trial_config,
            "center_score": _num(row.get("metric")),
            "algorithm_params": _decode(row.get("algorithm_params")) or {},
            "portfolio_params": _decode(row.get("portfolio_params")) or {},
        })
        if len(selected) >= max_seeds:
            break
    return selected


def seed_summary_rows(seeds: list[dict[str, Any]], space: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        nearest = min(
            [normalized_distance(seed["center_config"], other["center_config"], space) for other in seeds if other is not seed],
            default=None,
        )
        rows.append({
            "seed_id": seed["seed_id"],
            "candidate_id": seed["candidate_id"],
            "candidate_type": seed["candidate_type"],
            "specialist_regimes": seed["specialist_regimes"],
            "center_config_json": json.dumps(seed["center_config"], sort_keys=True),
            "center_score": seed["center_score"],
            "center_oos_return": "",
            "center_max_drawdown": "",
            "center_trade_count": "",
            "normalized_param_distance_to_nearest_seed": nearest,
            "selected_reason": "diverse_high_score_peak",
        })
    return rows


def build_plateau_neighbors(seeds: list[dict[str, Any]], resolved_cfg, platform: dict[str, Any]) -> list[dict[str, Any]]:
    space = _search_space(resolved_cfg)
    keys = _param_keys(resolved_cfg)
    radius = float(platform.get("plateau", {}).get("neighborhood_radius_pct", 0.10))
    max_neighbors = int(platform.get("plateau", {}).get("max_neighbors_per_seed", 25))
    neighbors = []
    for seed in seeds:
        seen = set()
        for config in _neighbor_configs(seed["center_config"], space, radius):
            key = json.dumps(config, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            algorithm_params = apply_tunable_config(copy.deepcopy(resolved_cfg.workload["algorithm"].get("params", {})), config, keys["algorithm"])
            portfolio_params = apply_tunable_config(copy.deepcopy(resolved_cfg.workload["portfolio"].get("params", {})), config, keys["portfolio"])
            candidate = build_candidate_from_params(resolved_cfg, algorithm_params, portfolio_params)
            neighbors.append({
                "seed_id": seed["seed_id"],
                "neighbor_id": f"{seed['seed_id']}_neighbor_{len(seen):03d}",
                "seed_candidate_id": seed["candidate_id"],
                "candidate": candidate,
                "candidate_type": seed["candidate_type"],
                "specialist_regimes": seed["specialist_regimes"],
                "config_patch": config,
                "param_distance_from_seed": normalized_distance(seed["center_config"], config, space),
            })
            if len(seen) >= max_neighbors:
                break
    return neighbors


def summarize_plateaus(
    *,
    seeds: list[dict[str, Any]],
    neighbors: list[dict[str, Any]],
    job_results: list[dict[str, Any]],
    platform: dict[str, Any],
) -> dict[str, Any]:
    by_neighbor = {neighbor["neighbor_id"]: {**neighbor, "overall": [], "regimes": []} for neighbor in neighbors}
    for result in job_results:
        if result.get("status") != "complete":
            continue
        payload = result["result"]
        neighbor_id = payload["neighbor_id"]
        if neighbor_id not in by_neighbor:
            continue
        by_neighbor[neighbor_id]["overall"].append(payload["overall_scorecard"])
        by_neighbor[neighbor_id]["regimes"].extend(payload["regime_scorecard"])

    neighbor_rows = []
    for neighbor in by_neighbor.values():
        metrics = _score_neighbor(neighbor, platform)
        neighbor_rows.append({
            "seed_id": neighbor["seed_id"],
            "job_id": neighbor["neighbor_id"],
            "candidate_id": neighbor["candidate"].candidate_id,
            "candidate_type": neighbor["candidate_type"],
            "specialist_regime": neighbor["specialist_regimes"],
            "param_distance_from_seed": neighbor["param_distance_from_seed"],
            "config_patch_json": json.dumps(neighbor["config_patch"], sort_keys=True),
            **metrics,
            "detail_run_id": "",
        })

    summary_rows = []
    for seed in seeds:
        rows = [row for row in neighbor_rows if row["seed_id"] == seed["seed_id"]]
        accepted, metrics, reason = _score_seed(seed, rows, platform)
        summary_rows.append({
            "seed_id": seed["seed_id"],
            "candidate_id": seed["candidate_id"],
            "candidate_type": seed["candidate_type"],
            "specialist_regimes": seed["specialist_regimes"],
            "accepted": accepted,
            **metrics,
            "failure_reason": reason,
        })
    return {"neighbor_rows": neighbor_rows, "summary_rows": summary_rows}


def plateau_metrics(summary_rows: list[dict[str, Any]], jobs_total: int, jobs_complete: int, jobs_failed: int) -> dict[str, float]:
    pass_rates = [_num(row.get("neighbor_pass_rate")) for row in summary_rows]
    pass_rates = [value for value in pass_rates if value is not None]
    scores = [_num(row.get("plateau_score")) for row in summary_rows]
    scores = [value for value in scores if value is not None]
    degradations = [_num(row.get("peak_to_median_degradation")) for row in summary_rows]
    degradations = [value for value in degradations if value is not None]
    worst_quartiles = [_num(row.get("worst_quartile_oos_return")) for row in summary_rows]
    worst_quartiles = [value for value in worst_quartiles if value is not None]
    return {
        "plateau.seed_count": float(len(summary_rows)),
        "plateau.neighborhood_jobs_total": float(jobs_total),
        "plateau.neighborhood_jobs_complete": float(jobs_complete),
        "plateau.neighborhood_jobs_failed": float(jobs_failed),
        "plateau.accepted_plateaus": float(sum(1 for row in summary_rows if row.get("accepted") is True)),
        "plateau.rejected_peaks": float(sum(1 for row in summary_rows if row.get("accepted") is not True)),
        "plateau.best_score": max(scores, default=0.0),
        "plateau.best_pass_rate": max(pass_rates, default=0.0),
        "plateau.median_pass_rate": _median(pass_rates) or 0.0,
        "plateau.best_peak_to_median_degradation": min(degradations, default=0.0),
        "plateau.best_worst_quartile_return": max(worst_quartiles, default=0.0),
    }


def distance_decay_svg(seed_id: str, rows: list[dict[str, Any]], gate_value: float) -> str:
    points = sorted(
        [(float(row["param_distance_from_seed"]), _num(row.get("median_oos_return")), bool(row.get("passed_gate"))) for row in rows],
        key=lambda item: item[0],
    )
    width, height = 760, 360
    if not points:
        body = '<text x="30" y="70" font-size="14" fill="#444">No plateau neighborhood results.</text>'
    else:
        ys = [point[1] for point in points if point[1] is not None] + [gate_value]
        low, high = min(ys), max(ys)
        if low == high:
            low -= 1.0
            high += 1.0
        max_x = max([point[0] for point in points] + [0.01])

        def sx(x): return 70 + (x / max_x) * 600
        def sy(y): return 290 - ((y - low) / (high - low)) * 220

        gate_y = sy(gate_value)
        marks = []
        first_fail = None
        for x, y, passed in points:
            if y is None:
                continue
            if first_fail is None and y < gate_value:
                first_fail = x
            color = "#16a34a" if passed else "#dc2626"
            marks.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{color}"/>')
        fail_line = ""
        if first_fail is not None:
            fx = sx(first_fail)
            fail_line = f'<line x1="{fx:.1f}" y1="70" x2="{fx:.1f}" y2="290" stroke="#dc2626" stroke-dasharray="4 4"/>'
        body = (
            '<line x1="70" y1="290" x2="670" y2="290" stroke="#333"/>'
            '<line x1="70" y1="70" x2="70" y2="290" stroke="#333"/>'
            f'<line x1="70" y1="{gate_y:.1f}" x2="670" y2="{gate_y:.1f}" stroke="#f59e0b" stroke-width="2"/>'
            f'<text x="680" y="{gate_y + 4:.1f}" font-size="12" font-family="Arial" fill="#92400e">gate {gate_value:.4g}</text>'
            f"{fail_line}{''.join(marks)}"
            f'<text x="70" y="325" font-size="12" font-family="Arial" fill="#444">normalized parameter distance</text>'
            f'<text x="15" y="80" font-size="12" font-family="Arial" fill="#444">median OOS return</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#fff"/>'
        f'<text x="30" y="35" font-size="20" font-family="Arial" font-weight="700" fill="#111">{html.escape(seed_id)} distance decay</text>'
        f"{body}</svg>"
    )


def normalized_distance(left: dict[str, Any], right: dict[str, Any], space: dict[str, Any]) -> float:
    distances = []
    for key, spec in space.items():
        if key not in left or key not in right:
            continue
        distances.append(abs(_norm(left[key], spec) - _norm(right[key], spec)))
    if not distances:
        return 0.0
    return float(math.sqrt(sum(value * value for value in distances)) / math.sqrt(len(distances)))


def _score_neighbor(neighbor: dict[str, Any], platform: dict[str, Any]) -> dict[str, Any]:
    rows = neighbor["overall"]
    thresholds = _thresholds(platform, neighbor["candidate_type"])
    if neighbor["candidate_type"] == "specialist" and neighbor["specialist_regimes"]:
        targets = set(str(neighbor["specialist_regimes"]).split(","))
        rows = [row for row in neighbor["regimes"] if row.get("regime") in targets]
        thresholds = _thresholds(platform, "specialist")
    returns = [_num(row.get("total_return_pct")) for row in rows]
    drawdowns = [_num(row.get("max_drawdown_pct")) for row in rows]
    trades = [_num(row.get("total_trades")) for row in rows]
    median_return = _median([value for value in returns if value is not None])
    profitable = _pct([value > 0 for value in returns if value is not None])
    worst_drawdown = min([value for value in drawdowns if value is not None], default=None)
    trade_count = sum(value for value in trades if value is not None)
    passed = (
        len(rows) >= thresholds["min_windows"]
        and median_return is not None
        and median_return >= thresholds["min_return"]
        and (profitable or 0.0) >= thresholds["min_profitable"]
        and (worst_drawdown is None or abs(worst_drawdown) <= thresholds["max_drawdown"])
        and trade_count >= thresholds["min_trades"]
    )
    return {
        "passed_gate": passed,
        "oos_return": median_return,
        "median_oos_return": median_return,
        "max_drawdown": worst_drawdown,
        "trade_count": trade_count,
        "profitable_windows_pct": profitable,
        "worst_quartile_oos_return": _quantile([value for value in returns if value is not None], 0.25),
    }


def _score_seed(seed: dict[str, Any], rows: list[dict[str, Any]], platform: dict[str, Any]) -> tuple[bool, dict[str, Any], str]:
    cfg = platform.get("plateau", {})
    min_neighbors = int(cfg.get("min_neighbor_trials", 3))
    min_pass_rate = _pct_threshold(cfg.get("min_neighbor_pass_rate", 0.60))
    max_degradation = abs(_pct_threshold(cfg.get("max_peak_to_median_degradation", 0.50)))
    returns = [_num(row.get("median_oos_return")) for row in rows]
    returns = [value for value in returns if value is not None]
    pass_rate = _pct([bool(row.get("passed_gate")) for row in rows]) or 0.0
    median_return = _median(returns)
    worst_quartile = _quantile(returns, 0.25)
    center = _num(seed.get("center_score")) or median_return or 0.0
    degradation = center - (median_return or 0.0)
    trade_counts = [_num(row.get("trade_count")) for row in rows]
    trade_counts = [value for value in trade_counts if value is not None]
    trade_cv = statistics.stdev(trade_counts) / statistics.mean(trade_counts) if len(trade_counts) > 1 and statistics.mean(trade_counts) else 0.0
    metrics = {
        "plateau_score": (pass_rate / 100.0) * (median_return or 0.0) - max(degradation, 0.0),
        "neighbor_trials": len(rows),
        "neighbor_pass_rate": pass_rate,
        "median_oos_return": median_return,
        "worst_quartile_oos_return": worst_quartile,
        "max_drawdown_p95": _quantile([abs(_num(row.get("max_drawdown")) or 0.0) for row in rows], 0.95),
        "trade_count_cv": trade_cv,
        "peak_to_median_degradation": degradation,
        "regime_consistency_score": pass_rate,
    }
    reasons = []
    if len(rows) < min_neighbors:
        reasons.append("insufficient_neighbors")
    if pass_rate < min_pass_rate:
        reasons.append("plateau_pass_rate_too_low")
    if degradation > max_degradation:
        reasons.append("peak_to_median_degradation_too_high")
    return not reasons, metrics, ",".join(reasons)


def _neighbor_configs(center: dict[str, Any], space: dict[str, Any], radius: float) -> list[dict[str, Any]]:
    configs = [copy.deepcopy(center)]
    for key, spec in space.items():
        if key not in center:
            continue
        for value in _neighbor_values(center[key], spec, radius):
            config = copy.deepcopy(center)
            config[key] = value
            configs.append(config)
    return configs


def _neighbor_values(value: Any, spec: dict[str, Any], radius: float) -> list[Any]:
    t = spec.get("type")
    if t == "choice":
        values = list(spec.get("values", []))
        if value not in values:
            return []
        idx = values.index(value)
        return [values[i] for i in (idx - 1, idx + 1) if 0 <= i < len(values)]
    if t in {"uniform", "randint", "loguniform"}:
        low, high = float(spec["low"]), float(spec["high"])
        if t == "loguniform":
            lv, step = math.log(float(value)), (math.log(high) - math.log(low)) * radius
            values = [math.exp(lv - step), math.exp(lv + step)]
        else:
            step = (high - low) * radius
            values = [float(value) - step, float(value) + step]
        values = [min(max(v, low), high) for v in values]
        return [int(round(v)) for v in values] if t == "randint" else values
    return []


def _search_space(resolved_cfg) -> dict[str, Any]:
    return resolved_cfg.workload.get("search_space", {}).get("space") or resolved_cfg.platform.get("hpo", {}).get("space", {})


def _param_keys(resolved_cfg) -> dict[str, list[str]]:
    space_cfg = resolved_cfg.workload.get("search_space", {})
    platform_hpo = resolved_cfg.platform.get("hpo", {})
    return {
        "algorithm": space_cfg.get("algorithm_param_keys", platform_hpo.get("algorithm_param_keys", [])),
        "portfolio": space_cfg.get("portfolio_param_keys", platform_hpo.get("portfolio_param_keys", [])),
    }


def _accepted_candidates(run_dir: str | Path) -> set[str]:
    path = _existing_path(Path(run_dir), "stages/05_regime_gate/summaries/regime_gate_summary.csv", "summaries/regime_gate_summary.csv")
    if not path.exists():
        return set()
    rows = pd.read_csv(path).to_dict(orient="records")
    return {str(row["candidate_id"]) for row in rows if str(row.get("passed")).lower() in {"true", "1"}}


def _gate_meta(run_dir: str | Path, candidate_id: str) -> dict[str, Any]:
    path = _existing_path(Path(run_dir), "stages/05_regime_gate/summaries/regime_gate_summary.csv", "summaries/regime_gate_summary.csv")
    if not path.exists():
        return {}
    rows = pd.read_csv(path).to_dict(orient="records")
    for row in rows:
        if str(row.get("candidate_id")) == str(candidate_id):
            return row
    return {}


def _decode(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value)
    try:
        return json.loads(text)
    except Exception:
        return ast.literal_eval(text)


def _thresholds(platform: dict[str, Any], candidate_type: str) -> dict[str, float]:
    cfg = platform.get("gates", {}).get("specialist" if candidate_type == "specialist" else "generalist", {})
    return {
        "min_windows": int(cfg.get("min_regime_windows" if candidate_type == "specialist" else "min_windows", 1)),
        "min_trades": float(cfg.get("min_trades", 0)),
        "min_return": _pct_threshold(cfg.get("min_regime_median_oos_return" if candidate_type == "specialist" else "min_median_oos_return", 0.0)),
        "min_profitable": _pct_threshold(cfg.get("min_regime_profitable_windows_pct" if candidate_type == "specialist" else "min_profitable_windows_pct", 0.0)),
        "max_drawdown": abs(_pct_threshold(cfg.get("max_regime_drawdown" if candidate_type == "specialist" else "max_drawdown", 100.0))),
    }


def _norm(value: Any, spec: dict[str, Any]) -> float:
    t = spec.get("type")
    if t == "choice":
        values = list(spec.get("values", []))
        if not values:
            return 0.0
        return values.index(value) / max(len(values) - 1, 1) if value in values else 0.0
    low, high = float(spec.get("low", 0.0)), float(spec.get("high", 1.0))
    if high == low:
        return 0.0
    if t == "loguniform":
        low, high, value = math.log(low), math.log(high), math.log(float(value))
    return (float(value) - low) / (high - low)


def _median(values: list[float]) -> float | None:
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    mid = len(values) // 2
    return float(values[mid]) if len(values) % 2 else float((values[mid - 1] + values[mid]) / 2.0)


def _quantile(values: list[float], q: float) -> float | None:
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    idx = min(max(int((len(values) - 1) * q), 0), len(values) - 1)
    return float(values[idx])


def _pct(flags: list[bool]) -> float | None:
    if not flags:
        return None
    return 100.0 * sum(1 for flag in flags if flag) / len(flags)


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


def _existing_path(run_dir: Path, *relative_paths: str) -> Path:
    for relative_path in relative_paths:
        path = run_dir / relative_path
        if path.exists():
            return path
    return run_dir / relative_paths[0]
