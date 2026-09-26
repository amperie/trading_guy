"""Portable, lossless inputs deferred to the regime stage; never executable state."""
from datetime import datetime
from types import SimpleNamespace

from .scoring import regime_scorecard


def capture(portfolio, ticks, config, trades):
    return dict(config=config,
        ticks=[[[bar.symbol, bar.timestamp.isoformat(), bar.close] for bar in tick] for tick in ticks],
        equity=[[stamp.isoformat(), value] for stamp, value in portfolio.value_history.items()],
        trade_times=[(trade.exit_time or trade.entry_time).isoformat() for trade in trades])


def score(inputs):
    ticks = [[SimpleNamespace(symbol=symbol, timestamp=datetime.fromisoformat(stamp), close=close)
              for symbol, stamp, close in tick] for tick in inputs['ticks']]
    portfolio = SimpleNamespace(value_history={datetime.fromisoformat(stamp): value for stamp, value in inputs['equity']})
    trades = [SimpleNamespace(exit_time=datetime.fromisoformat(stamp)) for stamp in inputs['trade_times']]
    return regime_scorecard(portfolio, ticks, inputs['config'], trades)
