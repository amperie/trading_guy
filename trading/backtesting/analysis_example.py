"""
Example: Using the AnalysisEngine to analyze backtesting results

This example shows how to use the AnalysisEngine to extract trades,
calculate performance metrics, and generate visualizations and reports.
"""
from pathlib import Path

from trading.core.algorithms.macd_rsi_algorithm import MacdRsiAlgorithm
from trading.core.om.backtesting_om import BacktestingOM
from trading.core.pf.single_symbol_portfolio import SingleSymbolPortfolio
from trading.data_providers.test_data_provider import TestDataProvider
from trading.engines.backtest_engine import BacktestingEngine
from trading.engines.analysis_engine import AnalysisEngine


def main():
    # Run a backtest using the simulator
    print("Running backtest...")
    project_root = Path(__file__).parent.parent
    data_path = str(project_root / "data" / "UPRO_5min.csv")
    om = BacktestingOM()

    alg_cfg = {
        "macd_fastperiod": 480,
        "macd_slowperiod": 1040,
        "macd_signalperiod": 360,
        "rsi_period": 14,
    }

    al = MacdRsiAlgorithm(alg_cfg, history_length=1500)


    dp_cfg = {"path": data_path, "provider":"data_providers.test_data_provider.TestDataProvider"}
    dp = TestDataProvider(dp_cfg)
    pf = SingleSymbolPortfolio({"symbol": "UPRO", "stop_pct": 5, "profit_pct": 5}, om, 1000.0, {}, True)

    sim = BacktestingEngine({}, dp, al, om, pf)
    sim.run()

    # Create the analysis engine
    print("\nAnalyzing results...")
    engine = AnalysisEngine(sim.pf, pf.om)

    results = engine.run_full_analysis(
        run_name="SMA Strategy Test",
        description="Testing 5/20 SMA crossover",
        parameters={"sma_short": 5, "sma_long": 20, "symbol": "AAPL"}
    )

    # Extract trades from order history
    trades = engine.extract_trades()
    print(f"\nExtracted {len(trades)} trades:")
    for i, trade in enumerate(trades[:5], 1):  # Show first 5 trades
        print(f"  Trade {i}: {trade.symbol} - Entry: ${trade.entry_price:.2f}, "
              f"Exit: ${trade.exit_price:.2f}, P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
        if trade.is_bracket:
            print(f"    Bracket exit type: {trade.bracket_exit_type}")

    # Calculate comprehensive metrics
    metrics = engine.calculate_metrics()
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Total Return:       ${metrics.total_return:,.2f} ({metrics.total_return_pct:.2f}%)")
    print(f"Annualized Return:  {metrics.annualized_return:.2f}%")
    print(f"Sharpe Ratio:       {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown:       {metrics.max_drawdown_pct:.2f}%")
    print(f"\nTotal Trades:       {metrics.total_trades}")
    print(f"Win Rate:           {metrics.win_rate:.1f}%")
    print(f"Profit Factor:      {metrics.profit_factor:.2f}")

    if metrics.bracket_trades > 0:
        print(f"\nBracket Orders:     {metrics.bracket_trades}")
        print(f"Stop Loss Rate:     {metrics.bracket_stop_rate:.1f}%")
        print(f"Profit Taker Rate:  {metrics.bracket_profit_rate:.1f}%")

    # Get returns at different time scales
    print("\n" + "="*80)
    print("RETURNS ANALYSIS")
    print("="*80)

    tick_returns = engine.get_tick_returns()
    print(f"Tick Returns:       {len(tick_returns)} ticks")
    print(f"  Mean:             {tick_returns.mean()*100:.4f}%")
    print(f"  Std Dev:          {tick_returns.std()*100:.4f}%")

    daily_returns = engine.get_daily_returns()
    print(f"\nDaily Returns:      {len(daily_returns)} days")
    print(f"  Mean:             {daily_returns.mean()*100:.4f}%")
    print(f"  Best Day:         {daily_returns.max()*100:.2f}%")
    print(f"  Worst Day:        {daily_returns.min()*100:.2f}%")

    monthly_returns = engine.get_monthly_returns()
    if len(monthly_returns) > 0:
        print(f"\nMonthly Returns:    {len(monthly_returns)} months")
        print(f"  Mean:             {monthly_returns.mean()*100:.2f}%")

    # Analyze bracket order effectiveness
    if metrics.bracket_trades > 0:
        bracket_analysis = engine.analyze_bracket_effectiveness()
        print("\n" + "="*80)
        print("BRACKET ORDER ANALYSIS")
        print("="*80)
        print(f"Total Bracket Trades: {bracket_analysis['total_bracket_trades']}")

        if bracket_analysis['stop_loss_exits']:
            stats = bracket_analysis['stop_loss_exits']
            print(f"\nStop Loss Exits ({stats['count']}):")
            print(f"  Avg P&L:          ${stats['avg_pnl']:,.2f}")
            print(f"  Win Rate:         {stats['win_rate']:.1f}%")
            print(f"  Avg Duration:     {stats['avg_duration_hours']:.1f} hours")

        if bracket_analysis['profit_taker_exits']:
            stats = bracket_analysis['profit_taker_exits']
            print(f"\nProfit Taker Exits ({stats['count']}):")
            print(f"  Avg P&L:          ${stats['avg_pnl']:,.2f}")
            print(f"  Win Rate:         {stats['win_rate']:.1f}%")
            print(f"  Avg Duration:     {stats['avg_duration_hours']:.1f} hours")

    # Generate full text report
    report = engine.generate_report()

    # Save report to file
    with open("../../examples/backtest_report.txt", "w") as f:
        f.write(report)
    print("\n" + "="*80)
    print("Full report saved to: backtest_report.txt")
    print("="*80)

    # Generate visualizations (requires matplotlib)
    try:
        print("\nGenerating visualizations...")
        engine.plot_equity_curve(show=False, save_path="../../examples/equity_curve.png")
        print("  - Equity curve saved to: equity_curve.png")

        engine.plot_portfolio_with_trades(show=False, save_path="../../examples/portfolio_with_trades.png")
        print("  - Portfolio with trade markers saved to: portfolio_with_trades.png")

        engine.plot_drawdown(show=False, save_path="../../examples/drawdown.png")
        print("  - Drawdown chart saved to: drawdown.png")

        engine.plot_trade_pnl(show=False, save_path="../../examples/trade_pnl.png")
        print("  - Trade P&L chart saved to: trade_pnl.png")

        engine.plot_returns_distribution(show=False, save_path="../../examples/returns_dist.png")
        print("  - Returns distribution saved to: returns_dist.png")

        engine.plot_stock_performance(show=False, save_path="../../examples/stock_performance.png")
        print("  - Stock performance (prices & returns) saved to: stock_performance.png")

        engine.plot_comprehensive_dashboard(save_path="../../examples/dashboard.png")
        print("  - Comprehensive dashboard saved to: dashboard.png")

    except ImportError:
        print("\nNote: Install matplotlib to generate visualizations:")
        print("  pip install matplotlib")


if __name__ == "__main__":
    main()
