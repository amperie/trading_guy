# Zero-Quantity Order Filter - Fix Summary

## Problem
Buy and sell markers were appearing on charts for orders with 0 quantity, cluttering the visualization with meaningless markers.

## Root Cause
The chart generation code was filtering orders by status (FILLED) but not checking the quantity. This meant orders with `quantity = 0` were being included as markers on the charts.

## Solution
Added quantity validation to both chart methods to filter out orders with zero or negative quantities.

## Changes Made

### 1. Static Chart: `plot_portfolio_with_trades()`
**File**: `engines/analysis_engine.py` (line 632-633)

**Added filter**:
```python
# Skip orders with zero or negative quantity
if order.quantity <= 0:
    continue
```

**Location**: After status check, before processing order for markers

### 2. Interactive Chart: `plot_interactive_portfolio()`
**File**: `engines/analysis_engine.py` (line 873-874)

**Added filter**:
```python
# Skip orders with zero or negative quantity
if order.quantity <= 0:
    continue
```

**Location**: After status check, before processing order for markers

## Filter Logic

Both charts now apply this filtering sequence:
1. ✅ Check if order status is FILLED
2. ✅ **NEW**: Check if quantity > 0
3. ✅ Check if not a canceled child order (for bracket orders)
4. ✅ Add to buy_orders or sell_orders list

## Expected Behavior

### Before Fix
- Orders with quantity = 0 appeared as markers
- Cluttered charts with meaningless BUY/SELL markers
- Confusing visualization

### After Fix
- ✅ Only orders with quantity > 0 appear as markers
- ✅ Clean charts showing actual transactions
- ✅ Each marker represents real shares bought/sold

## Testing

### Test Script
Run: `python examples/test_zero_quantity_filter.py`

This will:
1. Run a backtest
2. Count orders by quantity
3. Generate both chart types
4. Verify filtering worked correctly

### Manual Verification

**Check the charts**:
- Open `test_zero_qty_static.png`
- Open `test_zero_qty_interactive.html`
- Verify that BUY/SELL markers match actual trades

**Count markers**:
- Number of green X (BUY) = Number of buy orders with quantity > 0
- Number of red X (SELL) = Number of sell orders with quantity > 0

## Impact

### Charts Affected
✅ `plot_portfolio_with_trades()` - Static matplotlib chart
✅ `plot_interactive_portfolio()` - Interactive Plotly chart

### Charts NOT Affected (don't show markers)
- `plot_equity_curve()` - No markers
- `plot_drawdown()` - No markers
- `plot_trade_pnl()` - Shows trades (which already filter by quantity)
- `plot_returns_distribution()` - No markers
- `plot_stock_performance()` - No markers
- `plot_comprehensive_dashboard()` - No trade markers

## Documentation Updates

Updated `CLAUDE.md`:
- `plot_portfolio_with_trades()` description now mentions "quantity > 0"
- `plot_interactive_portfolio()` description now mentions "shows only quantity > 0"

## Backwards Compatibility

✅ **Fully backwards compatible**
- Only filters out invalid orders (quantity <= 0)
- No impact on valid orders
- No API changes
- No config changes needed

## Edge Cases Handled

| Case | Behavior |
|------|----------|
| quantity = 0 | ❌ Filtered out (not shown) |
| quantity < 0 | ❌ Filtered out (not shown) |
| quantity > 0 | ✅ Shown as marker |
| FILLED status but quantity = 0 | ❌ Filtered out |
| PENDING order with quantity > 0 | ❌ Filtered out (wrong status) |

## Future Enhancements

Potential improvements:
- Add logging for filtered orders (debug mode)
- Add statistics on filtered vs shown orders
- Add tooltip showing why an order was filtered

## Questions?

If you still see zero-quantity markers:
1. Run the test script: `python examples/test_zero_quantity_filter.py`
2. Check if orders actually have quantity > 0
3. Verify you're using the updated code
4. Check order status (should be FILLED)

## Summary

✅ **Fixed**: Zero-quantity orders no longer appear as markers
✅ **Applied to**: Both static and interactive charts
✅ **Filter added at**: Lines 632-633 and 873-874 in `engines/analysis_engine.py`
✅ **Test script**: `examples/test_zero_quantity_filter.py`
✅ **Documentation**: Updated in `CLAUDE.md`

The charts now accurately represent only real transactions with actual share quantities!
