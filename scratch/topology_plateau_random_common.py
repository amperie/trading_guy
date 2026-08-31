from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    "trading/promoted/"
    "Topology-Based_UPRO_Predictor_Dual_Feature_B0_B1_Consensus_backtest_2a924236/"
    "TopologyConsensusAlgorithm.py"
)

PLATEAU = {
    "filtration_radius": (0.10, 0.30),
    "consensus_weight_b1": (0.55, 0.80),
    "b0_stability_threshold": (0.8, 2.3),
    "stability_lookback": (3200, 5000),
    "momentum_lookback": (700, 2500),
    "b1_momentum_threshold": (-0.5, 0.4),
    "position_sizing_pct": (0.5, 4.0),
}

EDGE = {
    "filtration_radius": [(0.30, 0.45)],
    "consensus_weight_b1": [(0.35, 0.55), (0.80, 0.90)],
    "b0_stability_threshold": [(2.3, 3.2)],
    "stability_lookback": [(2000, 3200)],
    "momentum_lookback": [(2500, 4000)],
}


def _rand_range(rng: random.Random, lo: float, hi: float, *, integer: bool = False):
    if integer:
        return rng.randint(int(lo), int(hi))
    return rng.uniform(lo, hi)


def sample_plateau(rng: random.Random) -> dict:
    params = {
        key: _rand_range(rng, *bounds, integer=key.endswith("lookback"))
        for key, bounds in PLATEAU.items()
    }
    params["consensus_weight_b0"] = 1.0 - params["consensus_weight_b1"]
    return params


def sample_edge(rng: random.Random) -> dict:
    params = sample_plateau(rng)
    key = rng.choice(list(EDGE))
    lo, hi = rng.choice(EDGE[key])
    params[key] = _rand_range(rng, lo, hi, integer=key.endswith("lookback"))
    if key == "consensus_weight_b1":
        params["consensus_weight_b0"] = 1.0 - params[key]
    params["edge_axis"] = key
    return params


def random_split(rng: random.Random, *, variable_train: bool) -> dict:
    val_months = rng.randint(6, 12)
    latest_start = pd.Timestamp("2025-12-31") - pd.DateOffset(months=val_months) + pd.Timedelta(days=1)
    earliest_start = pd.Timestamp("2023-01-01")
    span_days = int((latest_start - earliest_start).days)
    val_start = earliest_start + pd.Timedelta(days=rng.randint(0, span_days))
    val_end = val_start + pd.DateOffset(months=val_months) - pd.Timedelta(days=1)
    train_months = rng.randint(18, 36) if variable_train else 36
    train_end = val_start - pd.Timedelta(days=1)
    train_start = val_start - pd.DateOffset(months=train_months)
    return {
        "train_start": str(train_start.date()),
        "train_end": str(train_end.date()),
        "val_start": str(val_start.date()),
        "val_end": str(val_end.date()),
        "train_months": train_months,
        "val_months": val_months,
    }


def base_config(experiment_name: str, run_name: str, params: dict) -> dict:
    edge_axis = params.pop("edge_axis", None)
    return {
        "mode": "backtest",
        "mlflow": {
            "enabled": True,
            "tracking_uri": "http://z440.lan:5000",
            "experiment_name": "Trading Backtest",
            "auto_log_system_info": True,
        },
        "logging": {
            "level": "INFO",
            "console": False,
            "file_logging": False,
            "folder": "logs",
            "filename": "trading.log",
            "retention_days": 7,
            "quiet_loggers": ["websockets.client", "pymongo", "pymongo.connection", "pymongo.serverSelection"],
        },
        "state_store": {"enabled": False, "connection_uri": "mongodb://z440.lan:27017", "database": "trading_test"},
        "algorithm": {
            "implementation": "TopologyConsensusAlgorithm",
            "class_name": "TopologyConsensusAlgorithm",
            "source_path": SOURCE_PATH,
            "params": {
                "symbol": "UPRO",
                "symbols": ["UPRO"],
                "tradable_symbols": ["UPRO"],
                "macro_symbols": [],
                "normalization_window": 60,
                "max_position_pct": 5.0,
                "history_length": 70,
                **params,
            },
        },
        "portfolio": {
            "implementation": "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio",
            "params": {
                "cash": 100000.0,
                "keep_history": True,
                "stop_pct": 2.0,
                "profit_pct": 5.0,
                "tx_cost": 0.0005,
                "symbol": "UPRO",
            },
        },
        "order_manager": {
            "implementation": "trading.core.om.backtesting_om.BacktestingOrderManager",
            "params": {"market_hours_only": False},
        },
        "data_provider": {
            "implementation": "trading.data_providers.alpaca_data_provider.AlpacaDataProvider",
            "params": {
                "provider": "alpaca",
                "symbols": ["UPRO"],
                "timeframe": "Minute",
                "adjustment": "split",
                "market_hours_only": True,
                "start_date": "2023-01-01",
                "end_date": "2025-12-31",
                "limit": 500000,
            },
        },
        "analysis": {
            "enabled": True,
            "log_to_mlflow": True,
            "experiment_name": experiment_name,
            "run_name": run_name,
            "description": f"Randomized topology plateau stability sample. edge_axis={edge_axis or ''}",
            "sample_negative_rate": 20,
        },
    }


def write_samples(args: argparse.Namespace, *, variable_train: bool) -> list[Path]:
    rng = random.Random(args.seed)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    total = args.plateau_samples + args.edge_samples
    for i in range(total):
        kind = "plateau" if i < args.plateau_samples else "edge"
        params = sample_plateau(rng) if kind == "plateau" else sample_edge(rng)
        split = random_split(rng, variable_train=variable_train)
        sample_id = f"{args.prefix}_{kind}_{i + 1:03d}"
        cfg = base_config(args.experiment_name, sample_id, params)
        cfg["split_validation"] = {"splits": [{"name": sample_id, **split}], "sample_kind": kind}
        path = out_dir / f"{sample_id}.yaml"
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        paths.append(path)
    return paths


def run_one(path: str, account: str, log_dir: str) -> dict:
    log_path = Path(log_dir) / f"{Path(path).stem}.log"
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    cmd = [
        sys.executable,
        "run.py",
        "split-backtest",
        "--config",
        path,
        "--account",
        account,
    ]
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return {"config": path, "returncode": proc.returncode, "log": str(log_path)}


def run_with_ray(paths: list[Path], args: argparse.Namespace) -> list[dict]:
    import ray

    log_dir = ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    ray.init(num_cpus=args.max_concurrent_trials, ignore_reinit_error=True)
    remote = ray.remote(num_cpus=1)(run_one)
    refs = [remote.remote(str(path), args.account, str(log_dir)) for path in paths]
    results = []
    while refs:
        done, refs = ray.wait(refs, num_returns=1)
        result = ray.get(done[0])
        results.append(result)
        print(f"{len(results)}/{len(paths)} rc={result['returncode']} {result['config']} log={result['log']}", flush=True)
        if result["returncode"] != 0 and args.stop_on_error:
            ray.shutdown()
            raise SystemExit(result["returncode"])
    ray.shutdown()
    return results


def parse_args(default_prefix: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plateau-samples", type=int, default=100)
    parser.add_argument("--edge-samples", type=int, default=30)
    parser.add_argument("--max-concurrent-trials", type=int, default=8)
    parser.add_argument("--account", default="tom_paper")
    parser.add_argument("--experiment-name", default="topology_targets")
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--prefix", default=default_prefix)
    parser.add_argument("--output-dir", default=f"scratch/generated_{default_prefix}_configs")
    parser.add_argument("--log-dir", default=f".tmp/{default_prefix}_logs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main(default_prefix: str, *, variable_train: bool) -> int:
    args = parse_args(default_prefix)
    paths = write_samples(args, variable_train=variable_train)
    print(f"Wrote {len(paths)} configs")
    if args.dry_run:
        for path in paths:
            print(path)
        return 0
    results = run_with_ray(paths, args)
    failed = [result for result in results if result["returncode"] != 0]
    print(f"Completed {len(results)} runs, failed {len(failed)}")
    return 1 if failed else 0
