"""
Example: Using MLflow to track backtesting experiments

This script demonstrates how to use the MLflowClient to track and log
backtesting runs, including parameters, metrics, and visualizations.
"""
from utils.mlflow_client import MLflowClient
from utils.config_manager import ConfigManager
from engines.backtest_engine import BacktestEngine
from engines.analysis_engine import AnalysisEngine


def run_backtest_with_mlflow():
    """
    Run a backtest and log everything to MLflow
    """
    # Initialize MLflow client from config
    mlflow = MLflowClient.from_config()

    # Start a run with context manager (auto-closes on completion/error)
    with mlflow.start_run(
        run_name="Example Backtest Run",
        description="Demonstration of MLflow tracking with backtesting framework",
        tags={"environment": "development", "version": "1.0"}
    ):
        # Log configuration parameters
        config = ConfigManager().get_as_object()
        mlflow.log_params({
            "data_provider": config.simulator.data_provider.provider,
            "data_path": config.simulator.data_provider.path,
            "algorithm": config.simulator.algorithm.algorithm,
            "initial_cash": config.simulator.portfolio.cash,
            "symbol": config.simulator.portfolio.symbol,
        })

        # Run the backtest
        print("Running backtest...")
        engine = BacktestEngine(cfg_section_to_use="simulator")
        engine.run()

        portfolio = engine.pf
        order_manager = engine.om

        # Analyze results
        print("Analyzing results...")
        analysis = AnalysisEngine(portfolio, order_manager)
        metrics = analysis.calculate_metrics()
        trades = analysis.extract_trades()

        # Log performance metrics
        print("Logging metrics to MLflow...")
        mlflow.log_metrics({
            # Returns
            "total_return": metrics.total_return,
            "total_return_pct": metrics.total_return_pct,
            "annualized_return": metrics.annualized_return,

            # Risk metrics
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "max_drawdown": metrics.max_drawdown,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "volatility": metrics.volatility,

            # Trade statistics
            "total_trades": float(metrics.total_trades),
            "winning_trades": float(metrics.winning_trades),
            "losing_trades": float(metrics.losing_trades),
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,

            # Average metrics
            "avg_win": metrics.avg_win,
            "avg_loss": metrics.avg_loss,
            "avg_trade_pnl": metrics.avg_trade_pnl,

            # Bracket orders
            "bracket_trades": float(metrics.bracket_trades),
            "bracket_stop_rate": metrics.bracket_stop_rate,
            "bracket_profit_rate": metrics.bracket_profit_rate,

            # Equity metrics
            "final_equity": metrics.final_equity,
            "initial_equity": metrics.initial_equity,
            "peak_equity": metrics.peak_equity,
        })

        # Generate and log visualizations
        print("Generating and logging charts...")

        # Equity curve
        fig = analysis.plot_equity_curve(show=False)
        mlflow.log_chart(fig, "equity_curve", format="png", dpi=150)

        # Portfolio with trades
        fig = analysis.plot_portfolio_with_trades(show=False)
        mlflow.log_chart(fig, "portfolio_with_trades", format="png", dpi=150)

        # Drawdown chart
        fig = analysis.plot_drawdown(show=False)
        mlflow.log_chart(fig, "drawdown", format="png", dpi=150)

        # Trade P&L
        fig = analysis.plot_trade_pnl(show=False)
        mlflow.log_chart(fig, "trade_pnl", format="png", dpi=150)

        # Returns distribution
        fig = analysis.plot_returns_distribution(show=False)
        mlflow.log_chart(fig, "returns_distribution", format="png", dpi=150)

        # Comprehensive dashboard
        fig = analysis.plot_comprehensive_dashboard(save_path=None)
        mlflow.log_chart(fig, "dashboard", format="png", dpi=200)

        # Generate and log text report
        print("Generating and logging reports...")
        report = analysis.generate_report()
        mlflow.log_text(report, "performance_report.txt")

        # Log trades as JSON
        trades_data = [
            {
                "symbol": t.symbol,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "duration_hours": t.duration / 3600,
                "is_bracket": t.is_bracket,
                "bracket_exit_type": t.bracket_exit_type,
            }
            for t in trades
        ]
        mlflow.log_json(trades_data, "trades.json")

        # Log bracket order analysis
        bracket_analysis = analysis.analyze_bracket_effectiveness()
        mlflow.log_json(bracket_analysis, "bracket_analysis.json")

        # Log summary as markdown
        markdown_summary = f"""
# Backtest Summary

## Strategy Details
- **Symbol:** {config.simulator.portfolio.symbol}
- **Algorithm:** {config.simulator.algorithm.algorithm}
- **Initial Capital:** ${metrics.initial_equity:,.2f}

## Performance
- **Total Return:** ${metrics.total_return:,.2f} ({metrics.total_return_pct:.2f}%)
- **Annualized Return:** {metrics.annualized_return:.2f}%
- **Sharpe Ratio:** {metrics.sharpe_ratio:.2f}
- **Max Drawdown:** {metrics.max_drawdown_pct:.2f}%

## Trade Statistics
- **Total Trades:** {metrics.total_trades}
- **Win Rate:** {metrics.win_rate:.2f}%
- **Profit Factor:** {metrics.profit_factor:.2f}
- **Average Win:** ${metrics.avg_win:,.2f}
- **Average Loss:** ${metrics.avg_loss:,.2f}

## Bracket Orders
- **Bracket Trades:** {metrics.bracket_trades}
- **Stop Loss Rate:** {metrics.bracket_stop_rate:.1f}%
- **Profit Taker Rate:** {metrics.bracket_profit_rate:.1f}%
"""
        mlflow.log_markdown(markdown_summary, "summary.md")

        # Print MLflow UI URL
        print(f"\n{'='*80}")
        print(f"Backtest complete! View results in MLflow UI:")
        print(f"{mlflow.get_run_url()}")
        print(f"{'='*80}\n")


def run_multiple_backtests():
    """
    Run multiple backtests with different parameters and log to MLflow
    Useful for hyperparameter tuning and strategy comparison
    """
    mlflow = MLflowClient.from_config()

    # Define different parameter sets to test
    test_configs = [
        {"sma_short": 5, "sma_long": 20, "symbol": "AAPL"},
        {"sma_short": 10, "sma_long": 30, "symbol": "AAPL"},
        {"sma_short": 5, "sma_long": 20, "symbol": "MSFT"},
    ]

    for config_params in test_configs:
        run_name = f"SMA_{config_params['sma_short']}_{config_params['sma_long']}_{config_params['symbol']}"

        with mlflow.start_run(
            run_name=run_name,
            description=f"Testing SMA strategy with {config_params}"
        ):
            # Log parameters
            mlflow.log_params(config_params)

            # Run backtest with these parameters
            # (You would modify your config or algorithm here)
            # ...

            print(f"Completed run: {run_name}")


if __name__ == "__main__":
    # Run single backtest with full logging
    run_backtest_with_mlflow()

    # Uncomment to run multiple backtests for comparison
    # run_multiple_backtests()
