from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

from trading.commands.backtest import cmd_backtest
from trading.commands.common import apply_cli_overrides, load_raw_config


def _split_specs(raw_cfg: dict[str, Any], args: argparse.Namespace) -> list[dict[str, str]]:
    if args.train_start or args.train_end or args.val_start or args.val_end:
        missing = [
            name
            for name in ("train_start", "train_end", "val_start", "val_end")
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError(f"Explicit split dates missing: {', '.join(missing)}")
        return [
            {
                "name": args.split_name,
                "train_start": args.train_start,
                "train_end": args.train_end,
                "val_start": args.val_start,
                "val_end": args.val_end,
            }
        ]

    cfg = raw_cfg.get("split_validation", {}) or {}
    splits = cfg.get("splits") or []
    if not splits:
        raise ValueError(
            "split-backtest needs --train-start/--train-end/--val-start/--val-end "
            "or config split_validation.splits entries."
        )
    return splits


def _phase_cfg(base_cfg: dict[str, Any], split: dict[str, str], phase: str) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    start_key, end_key = (("train_start", "train_end") if phase == "train" else ("val_start", "val_end"))
    dp_params = cfg.setdefault("data_provider", {}).setdefault("params", {})
    dp_params["start_date"] = split[start_key]
    dp_params["end_date"] = split[end_key]

    analysis = cfg.setdefault("analysis", {})
    base_run_name = analysis.get("run_name") or "split_backtest"
    split_name = split.get("name") or "split"
    analysis["run_name"] = f"{base_run_name}_{split_name}_{phase}"
    analysis["description"] = (
        f"{analysis.get('description', '').strip()} "
        f"split_backtest={split_name} phase={phase} dates={split[start_key]}..{split[end_key]}"
    ).strip()
    analysis.setdefault("split_validation", {})
    analysis["split_validation"].update(
        {
            "split": split_name,
            "phase": phase,
            "train_start": split["train_start"],
            "train_end": split["train_end"],
            "val_start": split["val_start"],
            "val_end": split["val_end"],
        }
    )
    return cfg


def _write_cfg(cfg: dict[str, Any], out_dir: Path, run_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_name}.yaml"
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path


def cmd_split_backtest(args: argparse.Namespace):
    raw_cfg = load_raw_config(args.config)
    split_cfg = raw_cfg.pop("split_validation", None)
    raw_cfg = apply_cli_overrides(raw_cfg, args)
    if split_cfg is not None:
        raw_cfg["split_validation"] = split_cfg

    out_dir = Path(args.output_dir)
    results = []
    base_cfg = copy.deepcopy(raw_cfg)
    base_cfg.pop("split_validation", None)
    for split in _split_specs(raw_cfg, args):
        for phase in ("train", "val"):
            cfg = _phase_cfg(base_cfg, split, phase)
            run_name = cfg["analysis"]["run_name"]
            path = _write_cfg(cfg, out_dir, run_name)
            phase_args = copy.copy(args)
            phase_args.config = str(path)
            phase_args.run_name = None
            result = cmd_backtest(phase_args)
            results.append({"split": split.get("name"), "phase": phase, "config": str(path), "result": result})
    return results
