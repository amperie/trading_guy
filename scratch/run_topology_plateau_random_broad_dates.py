from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import yaml

from topology_plateau_random_common import (
    ROOT,
    base_config,
    run_with_ray,
    sample_edge,
    sample_plateau,
)


def random_broad_split(rng: random.Random, *, earliest: str, latest: str, min_train_days: int) -> dict:
    earliest_ts = pd.Timestamp(earliest)
    latest_ts = pd.Timestamp(latest)
    val_days = rng.randint(60, 180)
    min_val_end = earliest_ts + pd.Timedelta(days=min_train_days + val_days)
    val_end = min_val_end + pd.Timedelta(days=rng.randint(0, int((latest_ts - min_val_end).days)))
    val_start = val_end - pd.Timedelta(days=val_days - 1)
    latest_train_start = val_start - pd.Timedelta(days=min_train_days)
    train_start = earliest_ts + pd.Timedelta(days=rng.randint(0, int((latest_train_start - earliest_ts).days)))
    return {
        "train_start": str(train_start.date()),
        "train_end": str((val_start - pd.Timedelta(days=1)).date()),
        "val_start": str(val_start.date()),
        "val_end": str(val_end.date()),
        "train_days": int((val_start - train_start).days),
        "val_days": val_days,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plateau-samples", type=int, default=100)
    parser.add_argument("--edge-samples", type=int, default=30)
    parser.add_argument("--max-concurrent-trials", type=int, default=8)
    parser.add_argument("--account", default="tom_paper")
    parser.add_argument("--experiment-name", default="topology_targets")
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--earliest-start", default="2010-01-01")
    parser.add_argument("--latest-end", default=str(pd.Timestamp.today().normalize().date()))
    parser.add_argument("--min-train-days", type=int, default=365)
    parser.add_argument("--prefix", default="topology_plateau_random_broad_dates")
    parser.add_argument("--output-dir", default="scratch/generated_topology_plateau_random_broad_dates_configs")
    parser.add_argument("--log-dir", default=".tmp/topology_plateau_random_broad_dates_logs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def write_samples(args: argparse.Namespace) -> list[Path]:
    rng = random.Random(args.seed)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(args.plateau_samples + args.edge_samples):
        kind = "plateau" if i < args.plateau_samples else "edge"
        params = sample_plateau(rng) if kind == "plateau" else sample_edge(rng)
        split = random_broad_split(
            rng,
            earliest=args.earliest_start,
            latest=args.latest_end,
            min_train_days=args.min_train_days,
        )
        sample_id = f"{args.prefix}_{kind}_{i + 1:03d}"
        cfg = base_config(args.experiment_name, sample_id, params)
        cfg["data_provider"]["params"].pop("limit", None)
        cfg["analysis"]["description"] = (
            f"{cfg['analysis']['description']} broad_random_dates=true "
            f"earliest_start={args.earliest_start} latest_end={args.latest_end}"
        )
        cfg["split_validation"] = {"splits": [{"name": sample_id, **split}], "sample_kind": kind}
        path = out_dir / f"{sample_id}.yaml"
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        paths.append(path)
    return paths


def main() -> int:
    args = parse_args()
    paths = write_samples(args)
    print(f"Wrote {len(paths)} configs")
    if args.dry_run:
        for path in paths:
            print(path)
        return 0
    results = run_with_ray(paths, args)
    failed = [result for result in results if result["returncode"] != 0]
    print(f"Completed {len(results)} runs, failed {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
