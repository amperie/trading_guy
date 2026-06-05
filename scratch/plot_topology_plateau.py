from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path("output/mlflow_train_val_915448476991171828")
TARGET = "topology_plateau_random_train_plateau_076"


def main() -> None:
    pairs = pd.read_csv(OUT_DIR / "april_2025_control_pairs.csv")
    raw = pd.read_csv(OUT_DIR / "raw_runs.csv")
    params = [
        "params.config.algorithm.params.filtration_radius",
        "params.config.algorithm.params.b0_stability_threshold",
        "params.config.algorithm.params.momentum_lookback",
        "params.config.algorithm.params.stability_lookback",
        "params.config.algorithm.params.position_sizing_pct",
        "params.config.algorithm.params.b1_momentum_threshold",
    ]
    train_params = (
        raw[raw["params.config.analysis.split_validation.phase"].eq("train")]
        [["params.config.analysis.split_validation.split", *params]]
        .rename(columns={"params.config.analysis.split_validation.split": "split"})
        .drop_duplicates("split")
    )
    data = pairs.merge(train_params, on="split", how="left")
    data["stable"] = (
        (data["train_metrics.annualized_return"] > 5)
        & (data["val_metrics.annualized_return"] > 5)
        & (data["val_metrics.total_trades"] >= 70)
        & (data["val_metrics.max_drawdown_pct"].abs() <= 35)
        & (data["gap_annualized_return"].abs() <= 35)
        & data["bucket"].eq("neither")
    )
    stable = data[data.stable]
    target = data[data.split.eq(TARGET)].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Topology Strategy Stable Plateau", fontsize=16)

    ax = axes[0, 0]
    sc = ax.scatter(
        data["params.config.algorithm.params.filtration_radius"],
        data["params.config.algorithm.params.b0_stability_threshold"],
        c=data["gap_annualized_return"].clip(-100, 100),
        cmap="RdYlGn_r",
        s=36,
        alpha=0.55,
    )
    ax.scatter(
        stable["params.config.algorithm.params.filtration_radius"],
        stable["params.config.algorithm.params.b0_stability_threshold"],
        facecolors="none",
        edgecolors="black",
        s=95,
        linewidth=1.2,
        label="stable plateau candidates",
    )
    ax.scatter(
        target["params.config.algorithm.params.filtration_radius"],
        target["params.config.algorithm.params.b0_stability_threshold"],
        marker="*",
        s=260,
        color="#1f4e9e",
        edgecolors="white",
        linewidth=0.9,
        label="recommended candidate",
    )
    ax.set_xlabel("filtration radius")
    ax.set_ylabel("B0 stability threshold")
    ax.set_title("Shape sensitivity vs stability gate")
    ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, label="annualized return gap, val - train")

    ax = axes[0, 1]
    ax.scatter(
        data["params.config.algorithm.params.momentum_lookback"],
        data["params.config.algorithm.params.stability_lookback"],
        c=data["val_metrics.annualized_return"].clip(-50, 150),
        cmap="viridis",
        s=36,
        alpha=0.55,
    )
    ax.scatter(
        stable["params.config.algorithm.params.momentum_lookback"],
        stable["params.config.algorithm.params.stability_lookback"],
        facecolors="none",
        edgecolors="black",
        s=95,
        linewidth=1.2,
    )
    ax.scatter(
        target["params.config.algorithm.params.momentum_lookback"],
        target["params.config.algorithm.params.stability_lookback"],
        marker="*",
        s=260,
        color="#1f4e9e",
        edgecolors="white",
        linewidth=0.9,
    )
    ax.set_xlabel("momentum lookback")
    ax.set_ylabel("stability lookback")
    ax.set_title("Time horizon region")

    ax = axes[1, 0]
    ax.scatter(
        data["train_metrics.annualized_return"],
        data["val_metrics.annualized_return"],
        c=data["val_metrics.max_drawdown_pct"].abs().clip(0, 80),
        cmap="magma_r",
        s=36,
        alpha=0.55,
    )
    ax.scatter(
        stable["train_metrics.annualized_return"],
        stable["val_metrics.annualized_return"],
        facecolors="none",
        edgecolors="black",
        s=95,
        linewidth=1.2,
    )
    ax.scatter(
        target["train_metrics.annualized_return"],
        target["val_metrics.annualized_return"],
        marker="*",
        s=260,
        color="#1f4e9e",
        edgecolors="white",
        linewidth=0.9,
    )
    low = min(data["train_metrics.annualized_return"].min(), data["val_metrics.annualized_return"].min())
    high = max(data["train_metrics.annualized_return"].max(), data["val_metrics.annualized_return"].max())
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set_xlabel("training annualized return")
    ax.set_ylabel("validation annualized return")
    ax.set_title("Performance agreement")

    ax = axes[1, 1]
    text = (
        "Recommended plateau center\n"
        f"{TARGET}\n\n"
        "Stable-candidate envelope:\n"
        f"filtration radius: {stable['params.config.algorithm.params.filtration_radius'].min():.3f} - "
        f"{stable['params.config.algorithm.params.filtration_radius'].max():.3f}\n"
        f"B0 threshold: {stable['params.config.algorithm.params.b0_stability_threshold'].min():.2f} - "
        f"{stable['params.config.algorithm.params.b0_stability_threshold'].max():.2f}\n"
        f"momentum lookback: {stable['params.config.algorithm.params.momentum_lookback'].min():.0f} - "
        f"{stable['params.config.algorithm.params.momentum_lookback'].max():.0f}\n"
        f"stability lookback: {stable['params.config.algorithm.params.stability_lookback'].min():.0f} - "
        f"{stable['params.config.algorithm.params.stability_lookback'].max():.0f}\n\n"
        "Target metrics:\n"
        f"train annualized: {target['train_metrics.annualized_return']:.2f}%\n"
        f"validation annualized: {target['val_metrics.annualized_return']:.2f}%\n"
        f"annualized gap: {target['gap_annualized_return']:.2f} pp\n"
        f"validation drawdown: {target['val_metrics.max_drawdown_pct']:.2f}%\n"
        f"validation trades: {target['val_metrics.total_trades']:.0f}"
    )
    ax.axis("off")
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=12, family="monospace")

    fig.tight_layout()
    path = OUT_DIR / "topology_stable_plateau.png"
    fig.savefig(path, dpi=170)
    print(path)


if __name__ == "__main__":
    main()
