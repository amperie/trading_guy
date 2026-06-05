from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("output/mlflow_train_val_915448476991171828")
APR_START = pd.Timestamp("2025-04-01")
APR_END = pd.Timestamp("2025-04-30")


def main() -> None:
    pairs = pd.read_csv(OUT_DIR / "complete_train_val_pairs.csv")
    raw = pd.read_csv(OUT_DIR / "raw_runs.csv")
    dates = raw[
        [
            "params.config.analysis.split_validation.split",
            "params.config.analysis.split_validation.train_start",
            "params.config.analysis.split_validation.train_end",
            "params.config.analysis.split_validation.val_start",
            "params.config.analysis.split_validation.val_end",
        ]
    ].dropna().drop_duplicates()
    dates = dates.rename(
        columns={
            "params.config.analysis.split_validation.split": "split",
            "params.config.analysis.split_validation.train_start": "train_start",
            "params.config.analysis.split_validation.train_end": "train_end",
            "params.config.analysis.split_validation.val_start": "val_start",
            "params.config.analysis.split_validation.val_end": "val_end",
        }
    )
    data = pairs.merge(dates, on="split", how="left")
    for col in ("train_start", "train_end", "val_start", "val_end"):
        data[col] = pd.to_datetime(data[col])

    data["train_apr2025"] = (data.train_start <= APR_END) & (data.train_end >= APR_START)
    data["val_apr2025"] = (data.val_start <= APR_END) & (data.val_end >= APR_START)
    data["bucket"] = np.select(
        [data.train_apr2025 & data.val_apr2025, data.train_apr2025, data.val_apr2025],
        ["both", "train only", "val only"],
        default="neither",
    )

    order = ["neither", "val only", "train only", "both"]
    summary = (
        data.groupby("bucket")
        .agg(
            n=("split", "size"),
            train_ann=("train_metrics.annualized_return", "median"),
            val_ann=("val_metrics.annualized_return", "median"),
            gap_ann=("gap_annualized_return", "median"),
            train_dd=("train_metrics.max_drawdown_pct", "median"),
            val_dd=("val_metrics.max_drawdown_pct", "median"),
            train_trades=("train_metrics.total_trades", "median"),
            val_trades=("val_metrics.total_trades", "median"),
        )
        .reindex(order)
        .dropna(how="all")
    )
    summary.to_csv(OUT_DIR / "april_2025_bucket_summary.csv")
    data.to_csv(OUT_DIR / "april_2025_control_pairs.csv", index=False)

    labels = [f"{idx}\n(n={int(row.n)})" for idx, row in summary.iterrows()]
    x = np.arange(len(summary))
    width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].bar(x - width / 2, summary.train_ann, width, label="train")
    axes[0].bar(x + width / 2, summary.val_ann, width, label="validation")
    axes[0].set_title("Median annualized return")
    axes[0].set_ylabel("%")
    axes[0].set_xticks(x, labels)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].legend()

    axes[1].bar(x - width / 2, summary.train_dd, width, label="train")
    axes[1].bar(x + width / 2, summary.val_dd, width, label="validation")
    axes[1].set_title("Median max drawdown")
    axes[1].set_ylabel("%")
    axes[1].set_xticks(x, summary.index)
    axes[1].axhline(0, color="black", lw=0.8)

    axes[2].bar(x - width / 2, summary.train_trades, width, label="train")
    axes[2].bar(x + width / 2, summary.val_trades, width, label="validation")
    axes[2].set_title("Median total trades")
    axes[2].set_xticks(x, summary.index)

    fig.suptitle("April 2025 exposure control buckets")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "april_2025_control_buckets.png", dpi=160)
    print(summary.to_string())


if __name__ == "__main__":
    main()
