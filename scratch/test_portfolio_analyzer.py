"""
Quick test / demo for PortfolioAnalyzer.

Creates a synthetic portfolio with realistic-looking trades and price history,
then runs the full analysis pipeline so you can iterate on chart/report changes
without needing a live backtest.

Run from the project root:
    python scratch/test_portfolio_analyzer.py
"""
import random
import sys
import os
from datetime import datetime, timedelta, timezone

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from trading.core.classes import (
    Order, OrderAction, OrderType, OrderStatus, OrderTimeInForce,
    PriceData, MarketSignal, SignalType,
)
from trading.analysis.portfolio_analyzer import PortfolioAnalyzer

# ---------------------------------------------------------------------------
# Reproducible seed
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
START_DATE = datetime(2024, 1, 2, tzinfo=timezone.utc)
NUM_BARS = 252          # ~1 year of daily bars
INITIAL_CASH = 100_000.0
OUTPUT_DIR = "output/test_analyzer"
SYMBOLS = ["SPY", "QQQ"]

# ---------------------------------------------------------------------------
# 1. Generate synthetic OHLCV price data
# ---------------------------------------------------------------------------

def generate_prices(start_price: float, n: int, mu: float = 0.0003, sigma: float = 0.012):
    """Geometric Brownian Motion daily close prices."""
    log_returns = np.random.normal(mu, sigma, n)
    prices = start_price * np.cumprod(np.exp(log_returns))
    return prices.tolist()


spy_closes = generate_prices(450.0, NUM_BARS)
qqq_closes = generate_prices(380.0, NUM_BARS)

timestamps = [START_DATE + timedelta(days=i) for i in range(NUM_BARS)]


def make_price_data(symbol: str, ts: datetime, close: float) -> PriceData:
    spread = close * 0.001
    return PriceData(
        symbol=symbol,
        timestamp=ts,
        open=round(close - spread * 0.5, 2),
        high=round(close + spread, 2),
        low=round(close - spread, 2),
        close=round(close, 2),
        volume=random.randint(50_000_000, 150_000_000),
    )


# ---------------------------------------------------------------------------
# 2. Build tick_history, value_history, cash_history, signals_history
# ---------------------------------------------------------------------------

tick_history: dict[datetime, list[PriceData]] = {}
value_history: dict[datetime, float] = {}
cash_history: dict[datetime, float] = {}
signals_history: dict[datetime, list[MarketSignal]] = {}

# Simulate portfolio value as a noisy version of SPY growth + alpha
portfolio_value = INITIAL_CASH
for i, ts in enumerate(timestamps):
    spy_c = spy_closes[i]
    qqq_c = qqq_closes[i]
    tick_history[ts] = [
        make_price_data("SPY", ts, spy_c),
        make_price_data("QQQ", ts, qqq_c),
    ]
    # Portfolio value: tracks SPY with some noise / alpha
    spy_daily_ret = (spy_closes[i] / spy_closes[i - 1] - 1) if i > 0 else 0.0
    pf_ret = spy_daily_ret + np.random.normal(0.0002, 0.004)
    portfolio_value *= (1 + pf_ret)
    value_history[ts] = round(portfolio_value, 2)
    cash_history[ts] = round(portfolio_value * random.uniform(0.0, 0.3), 2)

# Add signals at every 15th bar
for i, ts in enumerate(timestamps):
    if i % 15 == 0 and i > 0:
        sig_type = SignalType.BUY if spy_closes[i] > spy_closes[i - 1] else SignalType.SELL
        signals_history[ts] = [
            MarketSignal(type=sig_type, symbol="SPY", strength=random.randint(50, 95)),
            MarketSignal(type=sig_type, symbol="QQQ", strength=random.randint(40, 85)),
        ]


# ---------------------------------------------------------------------------
# 3. Build fake orders for 16 trades
#    Mix of: market (BUY+SELL) and bracket (BUY PENDING_SALE + STOP/PROFIT exit)
# ---------------------------------------------------------------------------

filled_orders: dict[str, Order] = {}
pending_orders: dict[str, Order] = {}


def add_market_trade(
    symbol: str,
    entry_bar: int,
    exit_bar: int,
    quantity: int,
    tx_cost: float = 1.0,
):
    """Create a completed market trade (BUY filled + SELL filled)."""
    entry_ts = timestamps[entry_bar]
    exit_ts = timestamps[exit_bar]
    closes = spy_closes if symbol == "SPY" else qqq_closes

    buy = Order(
        placed_datetime=entry_ts,
        action=OrderAction.BUY,
        type=OrderType.MARKET,
        symbol=symbol,
        price=round(closes[entry_bar], 2),
        quantity=quantity,
        cash=0.0,
        executed_datetime=entry_ts,
        status=OrderStatus.FILLED,
        tx_cost=tx_cost,
    )
    sell = Order(
        placed_datetime=exit_ts,
        action=OrderAction.SELL,
        type=OrderType.MARKET,
        symbol=symbol,
        price=round(closes[exit_bar], 2),
        quantity=quantity,
        cash=0.0,
        executed_datetime=exit_ts,
        status=OrderStatus.FILLED,
        tx_cost=tx_cost,
    )
    filled_orders[buy.order_id] = buy
    filled_orders[sell.order_id] = sell


def add_bracket_trade(
    symbol: str,
    entry_bar: int,
    exit_bar: int,
    quantity: int,
    exit_type: str,   # "STOP" or "PROFIT"
    tx_cost: float = 1.0,
):
    """Create a completed bracket trade (BUY in pending + STOP/PROFIT in filled)."""
    entry_ts = timestamps[entry_bar]
    exit_ts = timestamps[exit_bar]
    closes = spy_closes if symbol == "SPY" else qqq_closes

    entry_price = round(closes[entry_bar], 2)
    if exit_type == "PROFIT":
        exit_price = round(entry_price * random.uniform(1.02, 1.06), 2)
    else:  # STOP
        exit_price = round(entry_price * random.uniform(0.94, 0.98), 2)

    buy = Order(
        placed_datetime=entry_ts,
        action=OrderAction.BUY,
        type=OrderType.BRACKET,
        symbol=symbol,
        price=entry_price,
        quantity=quantity,
        cash=0.0,
        executed_datetime=entry_ts,
        status=OrderStatus.PENDING_SALE,
        tx_cost=tx_cost,
    )

    sell_type = OrderType.PROFIT_TAKER if exit_type == "PROFIT" else OrderType.STOP_LOSS
    sell = Order(
        placed_datetime=exit_ts,
        action=OrderAction.SELL,
        type=sell_type,
        symbol=symbol,
        price=exit_price,
        quantity=quantity,
        cash=0.0,
        executed_datetime=exit_ts,
        status=OrderStatus.FILLED,
        tx_cost=tx_cost,
        parent_id=buy.order_id,
    )
    pending_orders[buy.order_id] = buy
    filled_orders[sell.order_id] = sell


# Market trades — SPY
add_market_trade("SPY", entry_bar=5,   exit_bar=18,  quantity=50)   # win
add_market_trade("SPY", entry_bar=22,  exit_bar=35,  quantity=60)   # loss (manually set)
add_market_trade("SPY", entry_bar=40,  exit_bar=55,  quantity=45)   # win
add_market_trade("SPY", entry_bar=60,  exit_bar=70,  quantity=80)   # win
add_market_trade("SPY", entry_bar=80,  exit_bar=88,  quantity=55)   # loss

# Force one loss by adjusting price (bar 80 > bar 88 isn't guaranteed, but GBM will vary)
# (PnL is determined by actual simulated prices — can be win or loss)

# Market trades — QQQ
add_market_trade("QQQ", entry_bar=10,  exit_bar=25,  quantity=40)
add_market_trade("QQQ", entry_bar=50,  exit_bar=65,  quantity=35)
add_market_trade("QQQ", entry_bar=90,  exit_bar=110, quantity=50)

# Bracket trades — SPY (profit exits)
add_bracket_trade("SPY", entry_bar=95,  exit_bar=108, quantity=30, exit_type="PROFIT")
add_bracket_trade("SPY", entry_bar=120, exit_bar=133, quantity=40, exit_type="PROFIT")
add_bracket_trade("SPY", entry_bar=150, exit_bar=162, quantity=35, exit_type="PROFIT")

# Bracket trades — SPY (stop loss exits)
add_bracket_trade("SPY", entry_bar=140, exit_bar=148, quantity=25, exit_type="STOP")
add_bracket_trade("SPY", entry_bar=170, exit_bar=178, quantity=45, exit_type="STOP")

# Bracket trades — QQQ (mixed)
add_bracket_trade("QQQ", entry_bar=130, exit_bar=145, quantity=30, exit_type="PROFIT")
add_bracket_trade("QQQ", entry_bar=160, exit_bar=168, quantity=40, exit_type="STOP")
add_bracket_trade("QQQ", entry_bar=200, exit_bar=215, quantity=35, exit_type="PROFIT")


# ---------------------------------------------------------------------------
# 4. Fake OrderManager
# ---------------------------------------------------------------------------

class FakeOM:
    def __init__(self, filled: dict, pending: dict):
        self.filled_orders_by_id = filled
        self.pending_orders_by_id = pending


# ---------------------------------------------------------------------------
# 5. Fake Portfolio
# ---------------------------------------------------------------------------

class FakePortfolio:
    """Minimal object that satisfies PortfolioAnalyzer's interface."""
    def __init__(self):
        self.om = FakeOM(filled_orders, pending_orders)
        self.keep_history = True
        self.tick_history = tick_history
        self.value_history = value_history
        self.cash_history = cash_history
        self.signals_history = signals_history


# ---------------------------------------------------------------------------
# 6. Run analysis
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'=' * 60}")
    print("  PortfolioAnalyzer — test run")
    print(f"{'=' * 60}")
    print(f"  Bars:         {NUM_BARS}")
    print(f"  Symbols:      {SYMBOLS}")
    print(f"  Trades added: {len(filled_orders) + len(pending_orders)} orders")
    print(f"  Output dir:   {OUTPUT_DIR}")
    print()

    pf = FakePortfolio()
    analyzer = PortfolioAnalyzer(pf)

    result = analyzer.run_analysis(
        output_dir=OUTPUT_DIR,
        experiment_name="test_fake_portfolio",
        log_to_mlflow=True,     # set True to also log to MLflow
        run_name="test_fake_portfolio",
        parameters={"algorithm": "fake_macd", "fast": 12, "slow": 26, "signal": 9},
    )

    print("Files saved:")
    for fname, ok in result["files"].items():
        status = "OK" if ok else "FAILED"
        print(f"  [{status}]  {fname}")

    print()
    m = result["metrics"]
    if m:
        print("Key metrics:")
        print(f"  Total return:      {m.total_return_pct:.2f}%")
        print(f"  Annualized return: {m.annualized_return:.2f}%")
        print(f"  Sharpe ratio:      {m.sharpe_ratio:.3f}")
        print(f"  Max drawdown:      {m.max_drawdown_pct:.2f}%")
        print(f"  Total trades:      {m.total_trades}")
        print(f"  Win rate:          {m.win_rate:.1f}%")
        print(f"  Profit factor:     {m.profit_factor:.3f}")
        print(f"  Bracket trades:    {m.bracket_trades}")
        print(f"    - stop exits:    {m.bracket_stop_triggers}")
        print(f"    - profit exits:  {m.bracket_profit_triggers}")

    print()
    trades = result["trades"]
    print(f"Trades extracted: {len(trades)}")
    for t in trades[:5]:
        print(f"  {t.symbol:4s}  entry={t.entry_price:.2f}  exit={t.exit_price:.2f}"
              f"  qty={t.quantity:3d}  pnl=${t.pnl:+,.2f}"
              f"  {'[bracket:' + str(t.bracket_exit_type) + ']' if t.is_bracket else '[market]'}")
    if len(trades) > 5:
        print(f"  ... ({len(trades) - 5} more trades in trades.csv)")

    print(f"\nDone. Check {OUTPUT_DIR}/ for all 7 files.\n")


def test_portfolio_analyzer():
    """Pytest entry point — runs the full analysis pipeline on synthetic data."""
    main()


# ---------------------------------------------------------------------------
# MongoDB round-trip test (uses mongomock — no real Mongo needed)
# ---------------------------------------------------------------------------

def test_from_mongodb():
    """
    Persist the synthetic dataset into a mongomock store, reload it via
    PortfolioAnalyzer.from_mongodb(), and verify the results match a
    directly-constructed analyzer.
    """
    try:
        import mongomock
    except ImportError:
        print("\n  [SKIP] mongomock not installed — run: pip install mongomock")
        return

    from unittest.mock import patch
    from utils.trading_state_store import TradingStateStore

    print(f"\n{'=' * 60}")
    print("  test_from_mongodb  (mongomock)")
    print(f"{'=' * 60}")

    # ---- Build in-memory store and populate it ----
    mock_client = mongomock.MongoClient()
    store = TradingStateStore(client=mock_client, database="trading")
    sid = store.create_session(name="test_mongo_session")

    for ts in timestamps:
        sigs = signals_history.get(ts) or None
        store.save_tick(sid, ts, tick_history[ts], cash_history[ts], value_history[ts], sigs)

    store.save_orders(sid, {**filled_orders, **pending_orders})

    # ---- Load via from_mongodb (patch so it uses our mock store) ----
    with patch("utils.trading_state_store.TradingStateStore", return_value=store):
        mongo_analyzer = PortfolioAnalyzer.from_mongodb(sid)

    # ---- Verify history was round-tripped correctly ----
    assert mongo_analyzer.portfolio.keep_history is True
    assert len(mongo_analyzer.portfolio.value_history) == NUM_BARS, (
        f"Expected {NUM_BARS} snapshots, got {len(mongo_analyzer.portfolio.value_history)}"
    )
    assert len(mongo_analyzer.portfolio.tick_history) == NUM_BARS

    # ---- Cross-check trades and metrics against the direct analyzer ----
    direct = PortfolioAnalyzer(FakePortfolio())
    direct_trades  = direct.get_trades()
    direct_metrics = direct.get_metrics()

    mongo_trades  = mongo_analyzer.get_trades()
    mongo_metrics = mongo_analyzer.get_metrics()

    assert len(mongo_trades) == len(direct_trades), (
        f"Trade count mismatch: mongo={len(mongo_trades)}, direct={len(direct_trades)}"
    )
    assert abs(mongo_metrics.total_return_pct - direct_metrics.total_return_pct) < 0.01, (
        f"Return mismatch: mongo={mongo_metrics.total_return_pct:.4f}, "
        f"direct={direct_metrics.total_return_pct:.4f}"
    )

    print(f"  [OK]  {len(mongo_analyzer.portfolio.value_history)} snapshots round-tripped")
    print(f"  [OK]  {len(mongo_trades)} trades (matches direct)")
    print(f"  [OK]  total_return_pct  = {mongo_metrics.total_return_pct:.2f}%")
    print(f"  [OK]  win_rate          = {mongo_metrics.win_rate:.1f}%")
    print(f"  [OK]  bracket_trades    = {mongo_metrics.bracket_trades}")

    # ---- Log analysis to MLflow ----
    result = mongo_analyzer.run_analysis(
        output_dir="output/test_from_mongodb",
        log_to_mlflow=True,
        experiment_name="Portfolio Analyzer Tests",
        run_name="test_from_mongodb",
        parameters={
            "source": "mongomock",
            "num_bars": NUM_BARS,
            "symbols": str(SYMBOLS),
            "initial_cash": INITIAL_CASH,
        },
    )
    saved = sum(1 for ok in result["files"].values() if ok)
    print(f"  [OK]  MLflow + local artifacts: {saved}/{len(result['files'])} files saved")


def test_from_mongodb_multi():
    """
    Split the synthetic data into two half-sessions, persist them separately,
    then reload via from_mongodb_multi() and verify the merged result matches
    a single-session load.
    """
    try:
        import mongomock
    except ImportError:
        print("\n  [SKIP] mongomock not installed — skipping multi-session test")
        return

    from unittest.mock import patch
    from utils.trading_state_store import TradingStateStore

    print(f"\n{'=' * 60}")
    print("  test_from_mongodb_multi  (mongomock)")
    print(f"{'=' * 60}")

    mock_client = mongomock.MongoClient()
    store = TradingStateStore(client=mock_client, database="trading")

    mid = NUM_BARS // 2
    ts_first  = timestamps[:mid]
    ts_second = timestamps[mid:]

    sid1 = store.create_session(name="session_first_half")
    sid2 = store.create_session(name="session_second_half")

    for ts in ts_first:
        sigs = signals_history.get(ts) or None
        store.save_tick(sid1, ts, tick_history[ts], cash_history[ts], value_history[ts], sigs)

    for ts in ts_second:
        sigs = signals_history.get(ts) or None
        store.save_tick(sid2, ts, tick_history[ts], cash_history[ts], value_history[ts], sigs)

    store.save_orders(sid1, {**filled_orders, **pending_orders})
    store.save_orders(sid2, {**filled_orders, **pending_orders})

    with patch("utils.trading_state_store.TradingStateStore", return_value=store):
        multi_analyzer = PortfolioAnalyzer.from_mongodb_multi([sid1, sid2])

    assert len(multi_analyzer.portfolio.value_history) == NUM_BARS, (
        f"Expected {NUM_BARS} merged snapshots, got {len(multi_analyzer.portfolio.value_history)}"
    )

    print(f"  [OK]  {len(multi_analyzer.portfolio.value_history)} snapshots merged from 2 sessions")
    print(f"  [OK]  {len(multi_analyzer.get_trades())} trades")
    print(f"  [OK]  total_return_pct = {multi_analyzer.get_metrics().total_return_pct:.2f}%")

    # ---- Log analysis to MLflow ----
    result = multi_analyzer.run_analysis(
        output_dir="output/test_from_mongodb_multi",
        log_to_mlflow=True,
        experiment_name="Portfolio Analyzer Tests",
        run_name="test_from_mongodb_multi",
        parameters={
            "source": "mongomock_multi",
            "sessions": 2,
            "num_bars": NUM_BARS,
            "symbols": str(SYMBOLS),
            "initial_cash": INITIAL_CASH,
        },
    )
    saved = sum(1 for ok in result["files"].values() if ok)
    print(f"  [OK]  MLflow + local artifacts: {saved}/{len(result['files'])} files saved")


# ---------------------------------------------------------------------------
# Live MongoDB session analysis
# ---------------------------------------------------------------------------

def test_live_mongodb_session():
    """
    Load a real MongoDB session, run full analysis, and log results to MLflow.

    Connection URI and database are read from config.yaml state_store section
    (mongodb://z440.lan:27017 / trading_z_desktop by default).

    Pytest usage::
        pytest scratch/test_portfolio_analyzer.py::test_live_mongodb_session -s
    """
    session_id = "live_SPY_switching_2214_1716_1722"

    print(f"\n{'=' * 60}")
    print("  Live MongoDB Session Analysis")
    print(f"{'=' * 60}")
    print(f"  Session ID: {session_id}")

    # ---- Load from real MongoDB ----
    try:
        analyzer = PortfolioAnalyzer.from_mongodb(session_id, database="trading_hp")
    except Exception as e:
        print(f"  [FAIL] Could not load session from MongoDB: {e}")
        return

    bars = len(analyzer.portfolio.value_history)
    print(f"  Loaded:     {bars} portfolio snapshots")

    if bars == 0:
        print("  [SKIP] Session has no history data — nothing to analyze")
        return

    # ---- Session metadata ----
    session_meta = getattr(analyzer, '_session_metadata', {}) or {}
    session_name  = session_meta.get('name') or session_id[:8]
    account       = session_meta.get('account', '')
    output_dir    = f"output/live_{session_id[:8]}"

    print(f"  Session:    {session_name}" + (f"  ({account})" if account else ""))
    print(f"  Output:     {output_dir}/")

    # ---- Full analysis + MLflow logging ----
    result = analyzer.run_analysis(
        output_dir=output_dir,
        log_to_mlflow=True,
        experiment_name="Live Sessions",
        run_name=session_name,
        parameters={
            "session_id": session_id,
            "source": "mongodb_live",
            "account": account,
        },
    )

    # ---- Summary ----
    m = result["metrics"]
    saved = sum(1 for ok in result["files"].values() if ok)
    print()
    print(f"  Total Return:      {m.total_return_pct:.2f}%")
    print(f"  Annualized Return: {m.annualized_return:.2f}%")
    print(f"  Sharpe Ratio:      {m.sharpe_ratio:.3f}")
    print(f"  Max Drawdown:      {m.max_drawdown_pct:.2f}%")
    print(f"  Trades:            {m.total_trades}  (win rate {m.win_rate:.1f}%)")
    print(f"  Profit Factor:     {m.profit_factor:.3f}")
    print()
    print(f"  [OK]  {saved}/{len(result['files'])} artifacts saved to {output_dir}/")
    print(f"  [OK]  MLflow run '{session_name}' in experiment 'Live Sessions'")


if __name__ == "__main__":
    main()
    #test_from_mongodb()
    #test_from_mongodb_multi()
    test_live_mongodb_session()
