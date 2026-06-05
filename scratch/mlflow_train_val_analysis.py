from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd


TRACKING_URI = "http://hp.lan:8899"
EXPERIMENT_ID = "915448476991171828"
OUT_DIR = Path("output/mlflow_train_val_915448476991171828")
PHASE_COL = "params.config.analysis.split_validation.phase"
SPLIT_COL = "params.config.analysis.split_validation.split"
NAME_COL = "tags.mlflow.runName"

METRICS = [
    "annualized_return",
    "total_return_pct",
    "max_drawdown_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "profit_factor",
    "total_trades",
    "win_rate",
]


def metric_col(metric: str) -> str:
    return f"metrics.{metric}"


def pct(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x:,.2f}%"


def num(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x:,.2f}"


def latest_per_phase(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.sort_values("start_time").drop_duplicates([SPLIT_COL, PHASE_COL], keep="last")
    return df


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["run_id", "status", "start_time", NAME_COL, SPLIT_COL, PHASE_COL]
    cols += [metric_col(m) for m in METRICS if metric_col(m) in df.columns]
    paired = latest_per_phase(df[df[PHASE_COL].isin(["train", "val"])][cols])
    wide = paired.pivot(index=SPLIT_COL, columns=PHASE_COL)
    wide.columns = [f"{phase}_{col}" for col, phase in wide.columns]
    wide = wide.reset_index().rename(columns={SPLIT_COL: "split"})

    for metric in METRICS:
        t, v = f"train_{metric_col(metric)}", f"val_{metric_col(metric)}"
        if t in wide.columns and v in wide.columns:
            wide[f"gap_{metric}"] = wide[v] - wide[t]
            denom = wide[t].abs().replace(0, np.nan)
            wide[f"gap_pct_{metric}"] = wide[f"gap_{metric}"] / denom * 100
            wide[f"abs_gap_{metric}"] = wide[f"gap_{metric}"].abs()
    return wide


def plot_scatter(pairs: pd.DataFrame, metric: str, path: Path) -> None:
    t, v = f"train_{metric_col(metric)}", f"val_{metric_col(metric)}"
    data = pairs[[t, v, "split"]].dropna()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(data[t], data[v], s=28, alpha=0.75)
    low = min(data[t].min(), data[v].min())
    high = max(data[t].max(), data[v].max())
    pad = (high - low) * 0.06 or 1
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color="black", linewidth=1)
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel(f"Training {metric}")
    ax.set_ylabel(f"Validation {metric}")
    ax.set_title(f"Training vs validation: {metric}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_gap_bars(pairs: pd.DataFrame, metric: str, path: Path, n: int = 30) -> None:
    gap = f"gap_{metric}"
    data = pairs[["split", gap]].dropna().assign(abs_gap=lambda x: x[gap].abs())
    data = data.sort_values("abs_gap", ascending=False).head(n).sort_values(gap)
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = np.where(data[gap] >= 0, "#287c5b", "#b33b3b")
    ax.barh(data["split"], data[gap], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Validation minus training")
    ax.set_title(f"Largest {metric} gaps")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_gap_distribution(pairs: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, metric in zip(axes.ravel(), ["annualized_return", "total_return_pct", "max_drawdown_pct", "sharpe_ratio"]):
        data = pairs[f"gap_{metric}"].dropna()
        ax.hist(data, bins=24, color="#4d6f91", alpha=0.85)
        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(data.median(), color="#c27a2c", linewidth=1.5, label=f"median {data.median():.2f}")
        ax.set_title(metric)
        ax.legend()
    fig.suptitle("Validation minus training gap distributions", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize(df: pd.DataFrame, pairs: pd.DataFrame, complete: pd.DataFrame) -> str:
    lines = [
        "# MLflow Training vs Validation Analysis",
        "",
        f"- Tracking URI: `{TRACKING_URI}`",
        f"- Experiment ID: `{EXPERIMENT_ID}`",
        f"- Runs pulled: {len(df):,}",
        f"- Train runs: {(df[PHASE_COL] == 'train').sum():,}",
        f"- Validation runs: {(df[PHASE_COL] == 'val').sum():,}",
        f"- Untagged runs: {df[PHASE_COL].isna().sum():,}",
        f"- One-to-one pairs with both phases: {len(complete):,}",
        f"- Splits missing train: {pairs['train_run_id'].isna().sum():,}",
        f"- Splits missing validation: {pairs['val_run_id'].isna().sum():,}",
        "",
        "## Pairwise Gap Summary",
        "",
        "| Metric | Train median | Val median | Median gap | Mean gap | MAE | Val > Train | Correlation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        t, v, g = f"train_{metric_col(metric)}", f"val_{metric_col(metric)}", f"gap_{metric}"
        if t not in complete or v not in complete:
            continue
        valid = complete[[t, v, g]].dropna()
        if valid.empty:
            continue
        corr = valid[t].corr(valid[v])
        fmt = pct if metric.endswith("return") or metric.endswith("pct") or metric == "annualized_return" else num
        lines.append(
            f"| {metric} | {fmt(valid[t].median())} | {fmt(valid[v].median())} | "
            f"{fmt(valid[g].median())} | {fmt(valid[g].mean())} | {fmt(valid[g].abs().mean())} | "
            f"{(valid[g] > 0).mean() * 100:.1f}% | {corr:.3f} |"
        )

    ann = complete.dropna(subset=["gap_annualized_return"]).copy()
    ann["abs_gap"] = ann["gap_annualized_return"].abs()
    best_val = ann.sort_values("val_metrics.annualized_return", ascending=False).head(10)
    largest = ann.sort_values("abs_gap", ascending=False).head(10)
    lines += [
        "",
        "## Top Validation Annualized Return",
        "",
        "| Split | Train ann. | Val ann. | Gap | Train DD | Val DD | Train trades | Val trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in best_val.iterrows():
        lines.append(
            f"| {r['split']} | {pct(r['train_metrics.annualized_return'])} | {pct(r['val_metrics.annualized_return'])} | "
            f"{pct(r['gap_annualized_return'])} | {pct(r['train_metrics.max_drawdown_pct'])} | "
            f"{pct(r['val_metrics.max_drawdown_pct'])} | {num(r['train_metrics.total_trades'])} | {num(r['val_metrics.total_trades'])} |"
        )

    lines += [
        "",
        "## Largest Annualized Return Gaps",
        "",
        "| Split | Train ann. | Val ann. | Gap | Train sharpe | Val sharpe |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in largest.iterrows():
        lines.append(
            f"| {r['split']} | {pct(r['train_metrics.annualized_return'])} | {pct(r['val_metrics.annualized_return'])} | "
            f"{pct(r['gap_annualized_return'])} | {num(r['train_metrics.sharpe_ratio'])} | {num(r['val_metrics.sharpe_ratio'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    df = mlflow.search_runs(experiment_ids=[EXPERIMENT_ID], output_format="pandas", max_results=10000)
    pairs = build_pairs(df)
    complete = pairs[pairs["train_run_id"].notna() & pairs["val_run_id"].notna()].copy()

    df.to_csv(OUT_DIR / "raw_runs.csv", index=False)
    pairs.to_csv(OUT_DIR / "train_val_pairs.csv", index=False)
    complete.to_csv(OUT_DIR / "complete_train_val_pairs.csv", index=False)

    plot_scatter(complete, "annualized_return", OUT_DIR / "annualized_return_scatter.png")
    plot_scatter(complete, "max_drawdown_pct", OUT_DIR / "max_drawdown_scatter.png")
    plot_gap_bars(complete, "annualized_return", OUT_DIR / "largest_annualized_return_gaps.png")
    plot_gap_distribution(complete, OUT_DIR / "gap_distributions.png")
    (OUT_DIR / "summary.md").write_text(summarize(df, pairs, complete), encoding="utf-8")
    print(f"wrote {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
