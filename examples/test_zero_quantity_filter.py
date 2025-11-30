"""
Test script to verify that zero-quantity orders are filtered from charts
"""
from pathlib import Path
from algorithms.test_algorithm import TestAlgorithm
from core.om.backtesting_om import BacktestingOM
from core.pf.single_symbol_portfolio import SingleSymbolPortfolio
from data_providers.test_data_provider import TestDataProvider
from engines.backtest_engine import BacktestingEngine
from engines.analysis_engine import AnalysisEngine

print("="*80)
print("TESTING ZERO-QUANTITY ORDER FILTERING")
print("="*80)

# Run backtest
print("\n1. Running backtest...")
project_root = Path(__file__).parent.parent
data_path = str(project_root / "data" / "test_data.csv")

om = BacktestingOM()
al = TestAlgorithm({"history_length": 10, "full_history": True})
dp_cfg = {"path": data_path, "provider": "data_providers.test_data_provider.TestDataProvider"}
dp = TestDataProvider(dp_cfg)
pf = SingleSymbolPortfolio({"symbol": "AAPL"}, om, 10000.0, {}, True)

sim = BacktestingEngine({}, dp, al, om, pf)
sim.run()
print("   ✓ Backtest complete")

# Analyze orders
print("\n2. Analyzing filled orders...")
engine = AnalysisEngine(sim.pf, pf.om)

filled_orders = list(om.filled_orders_by_id.values())
total_filled = len(filled_orders)
zero_quantity = [o for o in filled_orders if o.quantity == 0]
positive_quantity = [o for o in filled_orders if o.quantity > 0]

print(f"   Total filled orders: {total_filled}")
print(f"   Orders with quantity > 0: {len(positive_quantity)}")
print(f"   Orders with quantity = 0: {len(zero_quantity)}")

if zero_quantity:
    print("\n   ⚠️  WARNING: Found orders with zero quantity:")
    for order in zero_quantity[:5]:  # Show first 5
        print(f"      - {order.action.name} {order.symbol}: quantity={order.quantity}, status={order.status.name}")
    if len(zero_quantity) > 5:
        print(f"      ... and {len(zero_quantity) - 5} more")
else:
    print("   ✓ No zero-quantity orders found")

# Test static chart
print("\n3. Generating static chart with trades...")
try:
    fig = engine.plot_portfolio_with_trades(show=False, save_path="test_zero_qty_static.png")
    print("   ✓ Static chart generated: test_zero_qty_static.png")
    print("   ✓ Chart should only show markers for orders with quantity > 0")
except Exception as e:
    print(f"   ✗ Failed to generate static chart: {e}")

# Test interactive chart
print("\n4. Generating interactive chart...")
try:
    fig = engine.plot_interactive_portfolio(show=False, save_path="test_zero_qty_interactive.html")
    print("   ✓ Interactive chart generated: test_zero_qty_interactive.html")
    print("   ✓ Chart should only show markers for orders with quantity > 0")
except Exception as e:
    print(f"   ✗ Failed to generate interactive chart: {e}")

# Extract trades to verify filtering
print("\n5. Extracting trades...")
trades = engine.extract_trades()
print(f"   Total trades: {len(trades)}")

# Check if any trade has zero quantity
zero_qty_trades = [t for t in trades if t.quantity == 0]
if zero_qty_trades:
    print(f"   ⚠️  WARNING: Found {len(zero_qty_trades)} trades with zero quantity!")
    for trade in zero_qty_trades[:3]:
        print(f"      - {trade.symbol}: entry_price={trade.entry_price}, quantity={trade.quantity}")
else:
    print("   ✓ All trades have positive quantity")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"\n✓ Filled orders with quantity > 0: {len(positive_quantity)}")
if zero_quantity:
    print(f"⚠️  Filled orders with quantity = 0: {len(zero_quantity)} (these should NOT appear in charts)")
else:
    print(f"✓ No zero-quantity orders found")

print(f"\n✓ Trades extracted: {len(trades)}")
if zero_qty_trades:
    print(f"⚠️  Trades with zero quantity: {len(zero_qty_trades)} (unexpected!)")
else:
    print(f"✓ All trades have positive quantity")

print("\nCharts generated:")
print("  📊 test_zero_qty_static.png")
print("  🎯 test_zero_qty_interactive.html")
print("\nOpen these files to verify that ONLY orders with quantity > 0")
print("are shown as BUY/SELL markers on the charts.")

if zero_quantity:
    print("\n" + "="*80)
    print("EXPECTED BEHAVIOR:")
    print("="*80)
    print(f"  • {len(zero_quantity)} orders have quantity = 0")
    print(f"  • These should NOT appear as markers in the charts")
    print(f"  • Charts should only show {len(positive_quantity)} markers total")
    print("="*80)

print("\n✅ Test complete! Review the generated charts to verify filtering.")
print("="*80)
