from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = (
    "trading/promoted/"
    "Topology-Based_UPRO_Predictor_Dual_Feature_B0_B1_Consensus_backtest_2a924236/"
    "TopologyConsensusAlgorithm.py"
)
OUT_DIR = ROOT / "scratch" / "generated_plateau_validation_configs"


BASE_CFG = {
    "mode": "backtest",
    "mlflow": {
        "enabled": True,
        "tracking_uri": "http://hp.lan:8899",
        "experiment_name": "Trading Backtest",
        "auto_log_system_info": True,
    },
    "logging": {
        "level": "INFO",
        "console": True,
        "file_logging": False,
        "folder": "logs",
        "filename": "trading.log",
        "retention_days": 7,
        "quiet_loggers": ["websockets.client", "pymongo", "pymongo.connection", "pymongo.serverSelection"],
    },
    "state_store": {"enabled": False, "connection_uri": "mongodb://hp.lan:27017", "database": "trading_test"},
    "algorithm": {
        "implementation": "TopologyConsensusAlgorithm",
        "class_name": "TopologyConsensusAlgorithm",
        "source_path": BASE_SOURCE,
        "params": {
            "symbol": "UPRO",
            "symbols": ["UPRO"],
            "tradable_symbols": ["UPRO"],
            "macro_symbols": [],
            "normalization_window": 60,
            "filtration_radius": 0.18,
            "momentum_lookback": 1500,
            "stability_lookback": 4300,
            "b1_momentum_threshold": 0.0,
            "b0_stability_threshold": 1.5,
            "consensus_weight_b1": 0.68,
            "consensus_weight_b0": 0.32,
            "position_sizing_pct": 1.5,
            "max_position_pct": 5.0,
            "history_length": 70,
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
            "start_date": "2025-11-02",
            "end_date": "2025-12-31",
            "limit": 500000,
        },
    },
    "analysis": {
        "enabled": True,
        "log_to_mlflow": True,
        "experiment_name": "topology_plateau_split_validation",
        "run_name": "topology_plateau_validation",
        "description": "Regular split-validation backtests for the topology plateau and edge candidates.",
        "mlflow_policy": {"annualized_return_threshold": 0.0, "sample_negative_rate": 20},
    },
    "aggregation": {"enabled": False, "aggregation_period_minutes": 1},
}


SPLITS = [
    ("2024_h1", "2021-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2024_h2", "2021-07-01", "2024-06-30", "2024-07-01", "2024-12-31"),
    ("2025_h1", "2022-01-01", "2024-12-31", "2025-01-01", "2025-06-30"),
    ("2025_h2", "2022-07-01", "2025-06-30", "2025-07-01", "2025-12-31"),
]


QUICK_SPLITS = ["2024_h2", "2025_h1", "2025_h2"]


CANDIDATES = [
    {
        "name": "plateau_center",
        "filtration_radius": 0.18,
        "momentum_lookback": 1500,
        "stability_lookback": 4300,
        "consensus_weight_b1": 0.68,
        "b0_stability_threshold": 1.5,
        "b1_momentum_threshold": 0.0,
        "position_sizing_pct": 1.5,
    },
    {
        "name": "low_filtration_edge",
        "filtration_radius": 0.10,
        "momentum_lookback": 1500,
        "stability_lookback": 4300,
        "consensus_weight_b1": 0.68,
        "b0_stability_threshold": 1.5,
        "b1_momentum_threshold": 0.0,
        "position_sizing_pct": 1.5,
    },
    {
        "name": "high_filtration_edge",
        "filtration_radius": 0.30,
        "momentum_lookback": 1500,
        "stability_lookback": 4300,
        "consensus_weight_b1": 0.68,
        "b0_stability_threshold": 1.5,
        "b1_momentum_threshold": 0.0,
        "position_sizing_pct": 1.5,
    },
    {
        "name": "low_b1_weight_edge",
        "filtration_radius": 0.18,
        "momentum_lookback": 1500,
        "stability_lookback": 4300,
        "consensus_weight_b1": 0.55,
        "b0_stability_threshold": 1.5,
        "b1_momentum_threshold": 0.0,
        "position_sizing_pct": 1.5,
    },
    {
        "name": "high_b1_weight_edge",
        "filtration_radius": 0.18,
        "momentum_lookback": 1500,
        "stability_lookback": 4300,
        "consensus_weight_b1": 0.80,
        "b0_stability_threshold": 1.5,
        "b1_momentum_threshold": 0.0,
        "position_sizing_pct": 1.5,
    },
    {
        "name": "low_b0_threshold_edge",
        "filtration_radius": 0.18,
        "momentum_lookback": 1500,
        "stability_lookback": 4300,
        "consensus_weight_b1": 0.68,
        "b0_stability_threshold": 0.8,
        "b1_momentum_threshold": 0.0,
        "position_sizing_pct": 1.5,
    },
    {
        "name": "high_b0_threshold_edge",
        "filtration_radius": 0.18,
        "momentum_lookback": 1500,
        "stability_lookback": 4300,
        "consensus_weight_b1": 0.68,
        "b0_stability_threshold": 2.3,
        "b1_momentum_threshold": 0.0,
        "position_sizing_pct": 1.5,
    },
    {
        "name": "short_lookbacks_edge",
        "filtration_radius": 0.18,
        "momentum_lookback": 700,
        "stability_lookback": 3300,
        "consensus_weight_b1": 0.68,
        "b0_stability_threshold": 1.5,
        "b1_momentum_threshold": 0.0,
        "position_sizing_pct": 1.5,
    },
    {
        "name": "long_lookbacks_edge",
        "filtration_radius": 0.18,
        "momentum_lookback": 2500,
        "stability_lookback": 5000,
        "consensus_weight_b1": 0.68,
        "b0_stability_threshold": 1.5,
        "b1_momentum_threshold": 0.0,
        "position_sizing_pct": 1.5,
    },
    {
        "name": "low_position_size",
        "filtration_radius": 0.18,
        "momentum_lookback": 1500,
        "stability_lookback": 4300,
        "consensus_weight_b1": 0.68,
        "b0_stability_threshold": 1.5,
        "b1_momentum_threshold": 0.0,
        "position_sizing_pct": 0.5,
    },
    {
        "name": "high_position_size",
        "filtration_radius": 0.18,
        "momentum_lookback": 1500,
        "stability_lookback": 4300,
        "consensus_weight_b1": 0.68,
        "b0_stability_threshold": 1.5,
        "b1_momentum_threshold": 0.0,
        "position_sizing_pct": 4.0,
    },
    {
        "name": "outside_high_filtration",
        "filtration_radius": 0.45,
        "momentum_lookback": 1500,
        "stability_lookback": 4300,
        "consensus_weight_b1": 0.68,
        "b0_stability_threshold": 1.5,
        "b1_momentum_threshold": 0.0,
        "position_sizing_pct": 1.5,
    },
    {
        "name": "outside_low_b1_weight_high_b0",
        "filtration_radius": 0.18,
        "momentum_lookback": 1500,
        "stability_lookback": 4300,
        "consensus_weight_b1": 0.35,
        "b0_stability_threshold": 3.0,
        "b1_momentum_threshold": 0.0,
        "position_sizing_pct": 1.5,
    },
]


def selected_splits(preset: str) -> list[tuple[str, str, str, str, str]]:
    if preset == "full":
        return SPLITS
    allowed = set(QUICK_SPLITS)
    return [split for split in SPLITS if split[0] in allowed]


def selected_candidates(preset: str) -> list[dict]:
    if preset == "full":
        return CANDIDATES
    quick_names = {
        "plateau_center",
        "low_filtration_edge",
        "high_filtration_edge",
        "low_b1_weight_edge",
        "high_b1_weight_edge",
        "low_b0_threshold_edge",
        "high_b0_threshold_edge",
        "outside_high_filtration",
    }
    return [candidate for candidate in CANDIDATES if candidate["name"] in quick_names]


def build_config(candidate: dict, split: tuple[str, str, str, str, str], phase: str, args: argparse.Namespace) -> dict:
    split_name, train_start, train_end, val_start, val_end = split
    start_date, end_date = (train_start, train_end) if phase == "train" else (val_start, val_end)
    cfg = copy.deepcopy(BASE_CFG)
    cfg["mlflow"]["tracking_uri"] = args.tracking_uri
    cfg["analysis"]["experiment_name"] = args.experiment_name
    cfg["analysis"]["log_to_mlflow"] = not args.no_mlflow
    cfg["analysis"]["run_name"] = f"topology_plateau_{candidate['name']}_{split_name}_{phase}"
    cfg["analysis"]["description"] = (
        "Regular validation-window backtest for topology plateau stability. "
        f"candidate={candidate['name']} split={split_name} phase={phase} dates={start_date}..{end_date}"
    )
    cfg["data_provider"]["params"]["start_date"] = start_date
    cfg["data_provider"]["params"]["end_date"] = end_date
    cfg["algorithm"]["params"].update({k: v for k, v in candidate.items() if k != "name"})
    cfg["algorithm"]["params"]["consensus_weight_b0"] = 1.0 - float(candidate["consensus_weight_b1"])
    cfg["analysis"]["validation_split"] = {
        "kind": "regular_backtest_validation",
        "preset": args.preset,
        "candidate": candidate["name"],
        "split": split_name,
        "phase": phase,
        "train_start_date": train_start,
        "train_end_date": train_end,
        "val_start_date": val_start,
        "val_end_date": val_end,
        "train_window_years": 3,
        "validation_window_months": 6,
    }
    return cfg


def write_configs(args: argparse.Namespace) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    phases = ["val"] if args.validation_only else ["train", "val"]
    for candidate in selected_candidates(args.preset):
        for split in selected_splits(args.preset):
            for phase in phases:
                cfg = build_config(candidate, split, phase, args)
                path = OUT_DIR / f"{cfg['analysis']['run_name']}.yaml"
                with path.open("w", encoding="utf-8") as fh:
                    yaml.safe_dump(cfg, fh, sort_keys=False)
                paths.append(path)
    return paths


def run_configs(paths: list[Path], args: argparse.Namespace) -> int:
    for idx, path in enumerate(paths, start=1):
        cmd = [
            sys.executable,
            "run.py",
            "backtest",
            "--config",
            str(path),
            "--account",
            args.account,
        ]
        print(f"[{idx}/{len(paths)}] {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(cmd, cwd=ROOT)
        if completed.returncode != 0:
            return completed.returncode
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run regular validation backtests across the topology plateau and edge candidates."
    )
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--account", default="paper")
    parser.add_argument("--experiment-name", default="topology_plateau_split_validation")
    parser.add_argument("--tracking-uri", default="http://hp.lan:8899")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--validation-only", action="store_true", help="Skip the 3-year train-window checks.")
    parser.add_argument("--dry-run", action="store_true", help="Only write configs and print commands.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = write_configs(args)
    print(f"Wrote {len(paths)} configs to {OUT_DIR}")
    return run_configs(paths, args)


if __name__ == "__main__":
    raise SystemExit(main())
