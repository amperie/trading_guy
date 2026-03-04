"""
Sweep over aggregation periods and compare backtest results.

Builds fresh pipeline components for each period, runs a backtest through the
TickAggregationPassthroughEngine, analyzes results, and logs each run to MLflow
under the "Aggregation Sweep" experiment.  Prints a comparison table at the end.

Usage:
    python scratch/run_agg_sweep.py
    python scratch/run_agg_sweep.py --periods 1 3 5 10 15 --data data/SPY_1min.csv
    python scratch/run_agg_sweep.py --no-mlflow
"""

import argparse
import sys

sys.path.insert(0, ".")

PERIODS_DEFAULT = [1, 3, 5, 10, 15]
DATA_DEFAULT = "data/SPY_1min.csv"
EXPERIMENT_NAME = "Aggregation Sweep"
CASH = 100_000.0
HISTORY_LENGTH = 200


def build_components(data_path: str):
    """Instantiate fresh pipeline components for one trial."""
    from trading.data_providers.test_data_provider import TestDataProvider
    from trading.algorithms.spy_trend_macd_algorithm import SpyTrendMACDAlgorithm
    from trading.core.om.backtesting_om import BacktestingOrderManager
    from trading.core.pf.single_symbol_portfolio import SingleSymbolPortfolio

    dp = TestDataProvider(cfg={"path": data_path})

    al = SpyTrendMACDAlgorithm(
        cfg={
            "spy_symbol": "SPY",
            "upro_symbol": "SPY",
            "spxu_symbol": "SPY",
            "macd_fast_period": 12,
            "macd_slow_period": 26,
            "macd_signal_period": 9,
        },
        history_length=HISTORY_LENGTH,
    )

    om = BacktestingOrderManager()

    pf = SingleSymbolPortfolio(
        cfg={
            "symbol": "SPY",
            "cash": CASH,
            "keep_history": True,
            "stop_pct": 3.0,
            "profit_pct": 6.0,
        },
        order_manager=om,
    )

    return dp, al, om, pf


def run_period(period: int, data_path: str, log_to_mlflow: bool) -> dict:
    """Run one backtest trial for the given aggregation period.  Returns a metrics dict."""
    from trading.engines.backtest_engine import BacktestingEngine
    from trading.engines.tick_aggregation_passthrough_engine import TickAggregationPassthroughEngine
    from trading.analysis.analysis_engine import AnalysisEngine

    dp, al, om, pf = build_components(data_path)

    bt_engine = BacktestingEngine(cfg={}, dp=dp, al=al, om=om, pf=pf)

    agg_cfg = {
        "downstream_engine": bt_engine,
        "aggregation_period_minutes": period,
        "use_market_open": True,
        "market_open_hour": 9,
        "market_open_minute": 30,
    }
    agg_engine = TickAggregationPassthroughEngine(cfg=agg_cfg)
    agg_engine.dp = dp

    print(f"  Running period={period}min ...", flush=True)
    agg_engine.run()

    engine = AnalysisEngine(pf, om)
    results = engine.run_full_analysis(
        log_to_mlflow=log_to_mlflow,
        experiment_name=EXPERIMENT_NAME,
        run_name=f"agg_{period}min",
        description=f"Aggregation sweep — {period}-min bars",
        parameters={"aggregation_period_minutes": period, "data_path": data_path},
        show_summary=False,
    )

    metrics = results.get("metrics", {}) if results else {}
    return metrics


def print_table(rows: list[dict], periods: list[int]):
    """Print a simple comparison table."""
    cols = [
        ("period", "Period"),
        ("total_return_pct", "Return %"),
        ("sharpe_ratio", "Sharpe"),
        ("max_drawdown_pct", "MaxDD %"),
        ("win_rate", "Win%"),
        ("total_trades", "Trades"),
        ("final_value", "Final Value"),
    ]

    header = "  ".join(f"{label:>12}" for _, label in cols)
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for period, m in zip(periods, rows):
        row_vals = {"period": period, **m}
        row = "  ".join(
            f"{row_vals.get(key, 'n/a'):>12}"
            if isinstance(row_vals.get(key), str)
            else f"{row_vals.get(key, float('nan')):>12.2f}"
            if isinstance(row_vals.get(key), float)
            else f"{row_vals.get(key, 'n/a'):>12}"
            for key, _ in cols
        )
        print(row)
    print(sep + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregation period sweep")
    parser.add_argument(
        "--periods", nargs="+", type=int, default=PERIODS_DEFAULT,
        help="Aggregation periods in minutes (default: 1 3 5 10 15)"
    )
    parser.add_argument(
        "--data", default=DATA_DEFAULT,
        help=f"Path to 1-min CSV data file (default: {DATA_DEFAULT})"
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Disable MLflow logging"
    )
    args = parser.parse_args()

    log_to_mlflow = not args.no_mlflow

    print(f"Aggregation sweep over periods: {args.periods}")
    print(f"Data: {args.data}")
    print(f"MLflow: {'enabled' if log_to_mlflow else 'disabled'}")
    print(f"Experiment: {EXPERIMENT_NAME}\n")

    results = []
    for period in args.periods:
        try:
            metrics = run_period(period, args.data, log_to_mlflow)
            results.append(metrics)
            ret = metrics.get("total_return_pct", float("nan"))
            sharpe = metrics.get("sharpe_ratio", float("nan"))
            print(f"  Period {period}min → return={ret:.2f}%  sharpe={sharpe:.3f}")
        except Exception as e:
            print(f"  Period {period}min → ERROR: {e}")
            results.append({})

    print_table(results, args.periods)


if __name__ == "__main__":
    main()
