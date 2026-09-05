"""
PortfolioAnalyzer: Clean-room replacement for AnalysisEngine.

Works for both backtesting and live runs. Does NOT import from analysis_engine.py.
Interface-compatible with AnalysisEngine — drop-in replacement.

Construction
------------
From an in-memory portfolio (backtest or live)::

    analyzer = PortfolioAnalyzer(portfolio)
    result = analyzer.run_analysis(output_dir="output/my_run")

From a stored MongoDB session (live trading post-mortem)::

    analyzer = PortfolioAnalyzer.from_mongodb("session-uuid")
    result = analyzer.run_analysis(output_dir="output/session_abc")

From multiple merged sessions (live bot restarted across several sessions)::

    analyzer = PortfolioAnalyzer.from_mongodb_multi(["sid1", "sid2", "sid3"])
    result = analyzer.run_analysis(output_dir="output/combined")

MongoDB connections fall back to the ``state_store`` section of config.yaml if
``connection_uri``/``database`` are not passed explicitly.  Session metadata
(algorithm class, config params) stored by the engine is automatically included
as MLflow parameters whenever ``run_analysis()`` or ``run_full_analysis()`` logs
a run — no extra caller work required.

Return dict keys
----------------
``run_analysis()``       → ``{"metrics", "trades", "files"}``
``run_full_analysis()``  → ``{"metrics", "trades", "lifecycle_chains",
                              "tick_returns", "daily_returns", "monthly_returns",
                              "bracket_analysis", "report", "benchmarks"}``
"""
from __future__ import annotations

import bisect
import csv
import dataclasses
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading.core.classes import OrderType, OrderAction
from trading.core.portfolio import Portfolio
from utils.indicators import calculate_macd_series, calculate_rsi_series
from utils.logger import Logger
from utils.mlflow_client import MLflowClient

logger = Logger().get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes (defined locally — no dependency on analysis_engine.py)
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Represents a completed trade (entry + exit)."""
    symbol: str
    entry_order: object  # Order — typed as object to avoid circular issues
    exit_order: object   # Order
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    duration: float             # seconds
    is_bracket: bool = False
    bracket_exit_type: Optional[str] = None  # "STOP", "PROFIT", "MANUAL"


@dataclass
class LifecycleChain:
    """Links a signal to the trade it spawned (signal_time is None if unmatched)."""
    signal_time: Optional[datetime]
    signal_type: Optional[str]       # "BUY" / "SELL" / None
    signal_strength: Optional[int]
    trade: Trade


@dataclass
class PerformanceMetrics:
    """30+ comprehensive performance metrics."""
    # Returns
    total_return: float
    total_return_pct: float
    annualized_return: float

    # Risk
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration: float   # days

    # Trade counts
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float

    # Trade analysis
    avg_trade_pnl: float
    avg_trade_duration: float   # hours
    avg_bars_in_trade: float

    # Bracket order specific
    bracket_trades: int
    bracket_stop_triggers: int
    bracket_profit_triggers: int
    bracket_manual_exits: int
    bracket_stop_rate: float
    bracket_profit_rate: float

    # Risk-adjusted
    calmar_ratio: float
    ulcer_index: float

    # Additional
    volatility: float
    skewness: float
    kurtosis: float
    best_day: float
    worst_day: float
    avg_daily_return: float

    # Equity curve
    final_equity: float
    initial_equity: float
    peak_equity: float

    # Time
    total_days: float
    trading_days: int


# ---------------------------------------------------------------------------
# PortfolioAnalyzer
# ---------------------------------------------------------------------------

class PortfolioAnalyzer:
    """
    Clean-room portfolio analysis engine.

    Constructor:
        PortfolioAnalyzer(portfolio)

    The portfolio's OrderManager is accessed via portfolio.om (may be None).
    Requires portfolio.keep_history=True for most outputs.
    """

    def __init__(self, portfolio: Portfolio, order_manager=None):
        """
        Args:
            portfolio:      Portfolio with keep_history=True.
            order_manager:  Accepted for AnalysisEngine interface compatibility but
                            ignored — the OM is accessed via portfolio.om internally.
        """
        self.portfolio = portfolio
        self._trades: Optional[list[Trade]] = None
        self._metrics: Optional[PerformanceMetrics] = None
        self._benchmark_paths: dict[str, str] = {}  # symbol → CSV path
        self._session_metadata: dict = {}  # populated by from_mongodb* classmethods

    # -----------------------------------------------------------------------
    # Core computation (lazy, cached)
    # -----------------------------------------------------------------------

    def get_trades(self) -> list[Trade]:
        """Extract completed trades via FIFO BUY/SELL matching."""
        if self._trades is not None:
            return self._trades

        om = self.portfolio.om
        if om is None:
            self._trades = []
            return self._trades

        trades: list[Trade] = []
        open_positions: dict[str, list] = defaultdict(list)

        all_orders = {**om.filled_orders_by_id, **om.pending_orders_by_id}
        sorted_orders = sorted(
            all_orders.values(),
            key=lambda o: o.executed_datetime if o.executed_datetime else o.placed_datetime,
        )

        for order in sorted_orders:
            symbol = order.symbol
            if order.action == OrderAction.BUY:
                open_positions[symbol].append(order)

            elif order.action == OrderAction.SELL:
                remaining_qty = order.quantity
                while remaining_qty > 0 and open_positions[symbol]:
                    buy_order = open_positions[symbol][0]
                    match_qty = min(remaining_qty, buy_order.quantity)

                    entry_price = buy_order.price
                    exit_price = order.price
                    pnl = (exit_price - entry_price) * match_qty - buy_order.tx_cost - order.tx_cost
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price else 0.0

                    is_bracket = buy_order.type == OrderType.BRACKET
                    bracket_exit_type = None
                    if is_bracket:
                        if order.type == OrderType.STOP_LOSS:
                            bracket_exit_type = "STOP"
                        elif order.type == OrderType.TRAILING_STOP:
                            bracket_exit_type = "TRAILING_STOP"
                        elif order.type == OrderType.PROFIT_TAKER:
                            bracket_exit_type = "PROFIT"
                        else:
                            bracket_exit_type = "MANUAL"

                    entry_time = buy_order.executed_datetime or buy_order.placed_datetime
                    exit_time = order.executed_datetime or order.placed_datetime

                    trades.append(Trade(
                        symbol=symbol,
                        entry_order=buy_order,
                        exit_order=order,
                        entry_time=entry_time,
                        exit_time=exit_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=match_qty,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        duration=(exit_time - entry_time).total_seconds(),
                        is_bracket=is_bracket,
                        bracket_exit_type=bracket_exit_type,
                    ))

                    remaining_qty -= match_qty
                    buy_order.quantity -= match_qty
                    if buy_order.quantity == 0:
                        open_positions[symbol].pop(0)

        self._trades = trades
        return trades

    def get_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        if self._metrics is not None:
            return self._metrics

        trades = self.get_trades()
        pf = self.portfolio

        if not pf.keep_history:
            raise ValueError("Portfolio must have keep_history=True to calculate metrics.")

        equity_curve = self._get_equity_series()
        initial_equity = float(equity_curve.iloc[0])
        final_equity = float(equity_curve.iloc[-1])
        peak_equity = float(equity_curve.max())

        returns = equity_curve.pct_change().dropna()
        total_days = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / 86400
        trading_days = len(equity_curve)

        total_return = final_equity - initial_equity
        total_return_pct = (total_return / initial_equity) * 100 if initial_equity else 0.0
        annualized_return = (
            ((final_equity / initial_equity) ** (365.25 / total_days) - 1) * 100
            if total_days > 0 and initial_equity > 0 else 0.0
        )

        # Drawdown
        running_max = equity_curve.expanding().max()
        drawdown = equity_curve - running_max
        drawdown_pct = (drawdown / running_max) * 100
        max_drawdown = float(drawdown.min())
        max_drawdown_pct = float(drawdown_pct.min())

        is_dd = drawdown < 0
        dp = is_dd.astype(int).diff().fillna(0)
        dd_starts = equity_curve.index[dp == 1]
        dd_ends = equity_curve.index[dp == -1]
        max_dd_duration = 0.0
        if len(dd_starts) > 0 and len(dd_ends) > 0:
            max_dd_duration = float(max(
                (end - start).total_seconds() / 86400
                for start, end in zip(dd_starts, dd_ends)
            ))

        # Risk metrics
        if len(returns) > 1:
            volatility = float(returns.std() * np.sqrt(252) * 100)
            sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0.0
            downside = returns[returns < 0]
            downside_std = float(downside.std() * np.sqrt(252) * 100) if len(downside) > 0 else 0.0
            sortino_ratio = (annualized_return / downside_std) if downside_std > 0 else 0.0
            skewness = float(returns.skew())
            kurtosis = float(returns.kurtosis())
        else:
            volatility = sharpe_ratio = sortino_ratio = skewness = kurtosis = 0.0

        calmar_ratio = (annualized_return / abs(max_drawdown_pct)) if max_drawdown_pct != 0 else 0.0
        ulcer_index = float(np.sqrt((drawdown_pct ** 2).mean()))

        best_day = float(returns.max() * 100) if len(returns) > 0 else 0.0
        worst_day = float(returns.min() * 100) if len(returns) > 0 else 0.0
        avg_daily_return = float(returns.mean() * 100) if len(returns) > 0 else 0.0

        # Trade stats
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl < 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        largest_win = float(max(wins)) if wins else 0.0
        largest_loss = float(min(losses)) if losses else 0.0
        total_wins = sum(wins) if wins else 0.0
        total_losses = abs(sum(losses)) if losses else 0.0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0.0

        avg_trade_pnl = float(np.mean([t.pnl for t in trades])) if trades else 0.0
        avg_trade_duration = float(np.mean([t.duration / 3600 for t in trades])) if trades else 0.0

        avg_bars_in_trade = 0.0
        if trades and pf.tick_history:
            tick_times = sorted(pf.tick_history.keys())
            bars_list = [
                sum(1 for tt in tick_times if trade.entry_time <= tt <= trade.exit_time)
                for trade in trades
            ]
            avg_bars_in_trade = float(np.mean(bars_list)) if bars_list else 0.0

        bracket_list = [t for t in trades if t.is_bracket]
        bracket_trades = len(bracket_list)
        bracket_stop_triggers = sum(1 for t in bracket_list if t.bracket_exit_type == "STOP")
        bracket_profit_triggers = sum(1 for t in bracket_list if t.bracket_exit_type == "PROFIT")
        bracket_manual_exits = sum(1 for t in bracket_list if t.bracket_exit_type == "MANUAL")
        bracket_stop_rate = (bracket_stop_triggers / bracket_trades * 100) if bracket_trades > 0 else 0.0
        bracket_profit_rate = (bracket_profit_triggers / bracket_trades * 100) if bracket_trades > 0 else 0.0

        self._metrics = PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            max_drawdown_duration=max_dd_duration,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            avg_trade_duration=avg_trade_duration,
            avg_bars_in_trade=avg_bars_in_trade,
            bracket_trades=bracket_trades,
            bracket_stop_triggers=bracket_stop_triggers,
            bracket_profit_triggers=bracket_profit_triggers,
            bracket_manual_exits=bracket_manual_exits,
            bracket_stop_rate=bracket_stop_rate,
            bracket_profit_rate=bracket_profit_rate,
            calmar_ratio=calmar_ratio,
            ulcer_index=ulcer_index,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            best_day=best_day,
            worst_day=worst_day,
            avg_daily_return=avg_daily_return,
            final_equity=final_equity,
            initial_equity=initial_equity,
            peak_equity=peak_equity,
            total_days=total_days,
            trading_days=trading_days,
        )
        return self._metrics

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _get_equity_series(self) -> pd.Series:
        pf = self.portfolio
        s = pd.Series(list(pf.value_history.values()), dtype=float)
        s.index = pd.to_datetime(list(pf.tick_history.keys()), utc=True)
        return s

    def _market_closed_vrects(
        self,
        index: "pd.DatetimeIndex",
        tz: str = "America/New_York",
        open_h: int = 9, open_m: int = 30,
        close_h: int = 16, close_m: int = 0,
    ) -> list:
        """Return [(x0_utc, x1_utc)] covering pre-market and after-hours per day."""
        local = index.tz_convert(tz)
        dates = pd.DatetimeIndex(local.normalize().unique())
        periods = []
        for d in dates:
            mo = d + pd.Timedelta(hours=open_h, minutes=open_m)
            mc = d + pd.Timedelta(hours=close_h, minutes=close_m)
            de = d + pd.Timedelta(days=1)
            periods.append((d.tz_convert("UTC"), mo.tz_convert("UTC")))   # pre-market
            periods.append((mc.tz_convert("UTC"), de.tz_convert("UTC")))  # after-hours
        return periods

    def _get_price_series_by_symbol(self) -> dict[str, pd.Series]:
        """Returns {symbol: pd.Series(close, index=timestamps)} from tick_history."""
        pf = self.portfolio
        prices_by_symbol: dict[str, dict] = defaultdict(dict)
        for ts, tick in pf.tick_history.items():
            for price_data in tick:
                prices_by_symbol[price_data.symbol][ts] = price_data.close
        return {
            sym: pd.Series(prices, dtype=float).sort_index()
            for sym, prices in prices_by_symbol.items()
        }

    # -----------------------------------------------------------------------
    # Save methods
    # -----------------------------------------------------------------------

    def save_equity_curve(self, path: str) -> None:
        """Save equity curve chart to path."""
        equity = self._get_equity_series()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(equity.index, equity.values, linewidth=1.5, color='steelblue')
        ax.set_title('Portfolio Equity Curve', fontsize=13)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def save_combined_equity_curve(
        self,
        other: "PortfolioAnalyzer",
        path: str,
        self_label: str = "Replay",
        other_label: str = "Live",
        self_color: str = "steelblue",
        other_color: str = "darkorange",
        align_start=None,
    ) -> None:
        """Save a combined equity curve chart comparing two portfolio histories.

        Both curves are normalized to % return from their first data point so that
        they share a common baseline, making divergence easy to spot.

        Args:
            other: The second PortfolioAnalyzer (e.g. live session) to compare against.
            path: Output file path (.png).
            self_label: Legend label for this analyzer's curve (default: "Replay").
            other_label: Legend label for other's curve (default: "Live").
            self_color: Matplotlib color for this curve (default: "steelblue").
            other_color: Matplotlib color for other's curve (default: "darkorange").
            align_start: Optional datetime — both series are trimmed to start at or
                after this timestamp. Pass session_start to exclude warmup bars from
                the replay curve so both curves cover the same live-session window.
        """
        replay_eq = self._get_equity_series()
        live_eq = other._get_equity_series()

        if align_start is None and len(live_eq) > 0:
            align_start = live_eq.index[0]
        if align_start is not None:
            align_ts = pd.to_datetime(align_start, utc=True)
            replay_eq = replay_eq[replay_eq.index >= align_ts]
            live_eq = live_eq[live_eq.index >= align_ts]

        # Normalize to % return from first point
        if len(replay_eq) > 0:
            replay_eq = (replay_eq / replay_eq.iloc[0] - 1) * 100
        if len(live_eq) > 0:
            live_eq = (live_eq / live_eq.iloc[0] - 1) * 100

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(replay_eq.index, replay_eq.values, linewidth=1.5, color=self_color, label=self_label)
        ax.plot(live_eq.index, live_eq.values, linewidth=1.5, color=other_color, label=other_label, alpha=0.85)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_title("Equity Curve: Replay vs. Live Session", fontsize=13)
        ax.set_xlabel("Date")
        ax.set_ylabel("Return (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_drawdown_chart(self, path: str) -> None:
        """Save drawdown % from peak chart to path."""
        equity = self._get_equity_series()
        running_max = equity.expanding().max()
        drawdown_pct = ((equity - running_max) / running_max) * 100

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(drawdown_pct.index, drawdown_pct.values, 0, alpha=0.45, color='crimson')
        ax.plot(drawdown_pct.index, drawdown_pct.values, linewidth=0.8, color='darkred')
        ax.set_title('Drawdown (% from Peak)', fontsize=13)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def save_orders_chart(self, path: str) -> None:
        """
        Save an orders-over-equity chart.

        The portfolio equity curve is drawn as the background. Every executed order
        is overlaid as a scatter point at the time it was filled, at the portfolio
        value at that moment.

        Color  = symbol  (tab10 palette — same assignment as save_technical_chart)
        Shape  = order category:
            ▲ Market Buy       ▼ Market Sell
            ◆ Bracket Entry    ✕ Stop Loss
            ★ Profit Taker     ✛ Manual Exit
        """
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        equity = self._get_equity_series()
        om = self.portfolio.om

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(equity.index, equity.values, linewidth=1.5, color='steelblue', alpha=0.8, zorder=1)
        ax.set_title('Portfolio Equity Curve — All Orders', fontsize=13)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)

        if om is None:
            fig.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return

        # ---- Order style: (marker, point-size) per category ----
        ORDER_STYLE = {
            'Market Buy':    ('^', 110),   # triangle up
            'Bracket Entry': ('D',  85),   # diamond
            'Market Sell':   ('v', 110),   # triangle down
            'Stop Loss':     ('X', 110),   # filled X
            'Trailing Stop': ('x', 110),   # x
            'Profit Taker':  ('*', 170),   # star
            'Manual Exit':   ('P',  95),   # filled plus
        }

        def _classify(order):
            if order.action == OrderAction.BUY:
                return 'Bracket Entry' if order.type == OrderType.BRACKET else 'Market Buy'
            if order.type == OrderType.STOP_LOSS:
                return 'Stop Loss'
            if order.type == OrderType.TRAILING_STOP:
                return 'Trailing Stop'
            if order.type == OrderType.PROFIT_TAKER:
                return 'Profit Taker'
            if order.parent_id:
                return 'Manual Exit'
            return 'Market Sell'

        all_orders = {**om.filled_orders_by_id, **om.pending_orders_by_id}

        # Stable symbol → color mapping (tab10, consistent with other charts)
        symbols_seen = sorted(set(o.symbol for o in all_orders.values()))
        tab_colors = plt.cm.tab10.colors
        symbol_colors = {
            sym: tab_colors[i % len(tab_colors)] for i, sym in enumerate(symbols_seen)
        }

        categories_used: set[str] = set()

        sorted_orders = sorted(
            all_orders.values(),
            key=lambda o: o.executed_datetime or o.placed_datetime,
        )
        for order in sorted_orders:
            ts = order.executed_datetime
            if ts is None:
                continue  # skip orders not yet executed
            try:
                equity_val = equity.asof(ts)
            except Exception:
                continue
            if pd.isna(equity_val):
                continue

            category = _classify(order)
            categories_used.add(category)
            marker, size = ORDER_STYLE[category]
            color = symbol_colors.get(order.symbol, 'gray')

            ax.scatter(ts, float(equity_val),
                       marker=marker, s=size,
                       color=color, zorder=5,
                       linewidths=1.2, edgecolors='white')

        # ---- Legend ----
        # Left legend: symbol colors
        symbol_handles = [
            Patch(facecolor=color, label=sym)
            for sym, color in symbol_colors.items()
        ]
        # Right legend: order type shapes (drawn in a neutral gray)
        type_handles = [
            Line2D([0], [0], marker=marker, color='w',
                   markerfacecolor='dimgray', markeredgecolor='white',
                   markersize=11 if cat == 'Profit Taker' else 9,
                   label=cat)
            for cat, (marker, _) in ORDER_STYLE.items()
            if cat in categories_used
        ]

        leg1 = ax.legend(handles=symbol_handles, title='Symbol',
                         fontsize=8, title_fontsize=9,
                         loc='upper left', framealpha=0.88)
        ax.add_artist(leg1)
        ax.legend(handles=type_handles, title='Order Type',
                  fontsize=8, title_fontsize=9,
                  loc='upper right', framealpha=0.88)

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _get_deduped_signals_by_symbol(self) -> dict:
        """
        Returns {symbol: [(timestamp, sig_type_str), ...]} where consecutive
        signals of the same type are collapsed — only the first occurrence of each
        type change is kept.  E.g. BUY, BUY, SELL → BUY, SELL.
        """
        pf = self.portfolio
        if not hasattr(pf, 'signals_history') or not pf.signals_history:
            return {}
        raw: dict = defaultdict(list)
        for ts in sorted(pf.signals_history.keys()):
            for sig in pf.signals_history[ts]:
                sig_type = sig.type.name if hasattr(sig.type, 'name') else str(sig.type)
                raw[sig.symbol].append((ts, sig_type))
        deduped = {}
        for symbol, signal_list in raw.items():
            out = []
            last_type = None
            for ts, sig_type in signal_list:
                if sig_type != last_type:
                    out.append((ts, sig_type))
                    last_type = sig_type
            deduped[symbol] = out
        return deduped

    def _build_lifecycle_chains(self) -> list[LifecycleChain]:
        """
        Link each trade to the nearest preceding signal of the same symbol.

        Matching rule: largest signal ts ≤ trade.entry_time with matching symbol,
        within 4 × estimated bar duration.  Unmatched trades get signal_time=None.
        """
        trades = self.get_trades()
        if not trades:
            return []

        pf = self.portfolio
        signals_history = getattr(pf, 'signals_history', None) or {}

        # Estimate bar duration from tick timestamps (seconds)
        tick_times_raw = sorted(pf.tick_history.keys()) if getattr(pf, 'tick_history', None) else []
        bar_duration = 86400.0
        if len(tick_times_raw) >= 2:
            gaps = [
                (tick_times_raw[i + 1] - tick_times_raw[i]).total_seconds()
                for i in range(min(len(tick_times_raw) - 1, 200))
            ]
            gaps = [g for g in gaps if g > 0]
            if gaps:
                bar_duration = float(np.median(gaps))

        def _norm(ts):
            t = pd.Timestamp(ts)
            return t.tz_localize('UTC') if t.tzinfo is None else t

        # Build per-symbol sorted signal lists: [(ts_norm, type, strength), ...]
        sig_by_symbol: dict[str, list] = defaultdict(list)
        for ts in sorted(signals_history.keys()):
            for sig in signals_history[ts]:
                sig_type = sig.type.name if hasattr(sig.type, 'name') else str(sig.type)
                sig_by_symbol[sig.symbol].append(
                    (_norm(ts), sig_type, getattr(sig, 'strength', None))
                )

        chains: list[LifecycleChain] = []
        for trade in trades:
            entry_norm = _norm(trade.entry_time)
            sig_list = sig_by_symbol.get(trade.symbol, [])
            matched_time = matched_type = matched_strength = None

            if sig_list:
                sig_times = [s[0] for s in sig_list]
                idx = bisect.bisect_right(sig_times, entry_norm) - 1
                if idx >= 0:
                    sig_ts, sig_type, strength = sig_list[idx]
                    delta = (entry_norm - sig_ts).total_seconds()
                    if 0 <= delta <= 4 * bar_duration:
                        matched_time = sig_ts
                        matched_type = sig_type
                        matched_strength = strength

            chains.append(LifecycleChain(
                signal_time=matched_time,
                signal_type=matched_type,
                signal_strength=matched_strength,
                trade=trade,
            ))
        return chains

    def get_lifecycle_chains(self) -> list[LifecycleChain]:
        """Return cached signal→entry→exit lifecycle chains."""
        return self._build_lifecycle_chains()

    def save_technical_chart(self, path: str) -> None:
        """
        Save 4-panel technical chart (shared x-axis):
          1. Portfolio equity curve  (+ vertical signal-change markers)
          2. Normalized close prices (+ triangle markers at each signal change per symbol)
          3. MACD histogram (green bars = positive, red = negative) + MACD/Signal lines
             + triangle markers at each signal change per symbol
          4. RSI lines (all symbols, 30/70 reference lines)

        Signal markers are deduplicated: a marker is drawn only when the signal type
        *changes* for that symbol (e.g. BUY→BUY is suppressed; BUY→SELL is shown).
        """
        equity = self._get_equity_series()
        prices_by_symbol = self._get_price_series_by_symbol()
        colors = plt.cm.tab10.colors

        # Build normalized series per symbol (needed for scatter y-values)
        norm_by_symbol: dict[str, pd.Series] = {}
        for symbol, price_series in prices_by_symbol.items():
            valid = price_series.dropna()
            if not valid.empty:
                norm_by_symbol[symbol] = (price_series / valid.iloc[0]) * 100

        # Deduplicated signals keyed by symbol
        sig_data = self._get_deduped_signals_by_symbol()
        _SIG_COLORS = {'BUY': 'limegreen', 'SELL': 'crimson'}
        _SIG_MARKERS = {'BUY': '^', 'SELL': 'v'}

        fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
        ax_equity, ax_price, ax_macd, ax_rsi = axes

        # Panel 1: Equity curve
        ax_equity.plot(equity.index, equity.values, linewidth=1.5, color='steelblue')
        ax_equity.set_title('Portfolio Equity Curve', fontsize=11)
        ax_equity.set_ylabel('Value ($)')
        ax_equity.grid(True, alpha=0.3)

        # Vertical lines on equity panel at each signal-type change (any symbol)
        if sig_data:
            # Collect unique (timestamp, type) pairs across all symbols (first type wins per ts)
            seen_ts: dict = {}
            for sym_signals in sig_data.values():
                for ts, sig_type in sym_signals:
                    if ts not in seen_ts:
                        seen_ts[ts] = sig_type
            for ts, sig_type in sorted(seen_ts.items()):
                ax_equity.axvline(ts, color=_SIG_COLORS.get(sig_type, 'gray'),
                                  alpha=0.35, linewidth=0.9, linestyle=':')

        # Build a stable symbol→color mapping (shared across panels)
        symbol_colors = {sym: colors[i % len(colors)] for i, sym in enumerate(prices_by_symbol)}

        # Panel 2: Normalized prices
        ax_price.set_title('Normalized Close Price (base = 100)  ▲ BUY  ▼ SELL (colored by symbol)', fontsize=11)
        ax_price.set_ylabel('Normalized Price')
        ax_price.grid(True, alpha=0.3)
        for symbol, price_series in prices_by_symbol.items():
            color = symbol_colors[symbol]
            valid = price_series.dropna()
            if valid.empty:
                continue
            base = valid.iloc[0]
            normalized = (valid / base) * 100
            ax_price.plot(normalized.index, normalized.values, linewidth=1.2, color=color, label=symbol)

            # Scatter markers colored by symbol — shape encodes BUY (▲) / SELL (▼)
            if symbol in sig_data and symbol in norm_by_symbol:
                norm_series = norm_by_symbol[symbol]
                for ts, sig_type in sig_data[symbol]:
                    try:
                        norm_val = norm_series.asof(ts)
                    except Exception:
                        continue
                    if pd.isna(norm_val):
                        continue
                    ax_price.scatter(
                        ts, norm_val,
                        marker=_SIG_MARKERS.get(sig_type, 'o'),
                        s=80, color=color,
                        zorder=6, linewidths=1.5, edgecolors='white',
                    )

        ax_price.legend(fontsize=8)

        # Panel 3: MACD Histogram — each symbol colored separately
        from matplotlib.patches import Patch  # proxy handles for histogram legend
        ax_macd.set_title('MACD Histogram  ▲ BUY  ▼ SELL (colored by symbol)', fontsize=11)
        ax_macd.set_ylabel('MACD')
        ax_macd.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax_macd.grid(True, alpha=0.3)

        # symbol → (histogram pd.Series, symbol color) for signal-marker lookup
        hist_series_by_symbol: dict[str, tuple] = {}

        for symbol, price_series in prices_by_symbol.items():
            color = symbol_colors[symbol]
            valid = price_series.dropna()
            if len(valid) < 26:
                continue
            closes = valid.tolist()
            idx = valid.index
            macd_s, signal_s, hist_s = calculate_macd_series(closes)
            macd_arr = np.array([v if v is not None else np.nan for v in macd_s])
            signal_arr = np.array([v if v is not None else np.nan for v in signal_s])
            hist_arr = np.array([v if v is not None else np.nan for v in hist_s])

            # Histogram fill: NaN→0 so fill_between has no visible area there.
            # Same symbol color for both positive/negative; opacity distinguishes them.
            hist_fill = np.where(np.isnan(hist_arr), 0.0, hist_arr)
            ax_macd.fill_between(idx, 0, hist_fill,
                                 where=hist_fill > 0,
                                 color=color, alpha=0.55)
            ax_macd.fill_between(idx, hist_fill, 0,
                                 where=hist_fill < 0,
                                 color=color, alpha=0.25)

            # MACD and signal lines on top of histogram
            ax_macd.plot(idx, macd_arr, linewidth=1.0, color=color, label=f'{symbol} MACD')
            ax_macd.plot(idx, signal_arr, linewidth=0.8, color=color,
                         linestyle='--', alpha=0.7, label=f'{symbol} Signal')

            hist_series_by_symbol[symbol] = (pd.Series(hist_arr, index=idx), color)

        # Signal change markers on MACD panel — colored by symbol, shaped by direction
        for sym, sym_signals in sig_data.items():
            entry = hist_series_by_symbol.get(sym)
            if entry is None and hist_series_by_symbol:
                entry = next(iter(hist_series_by_symbol.values()))
            if entry is None:
                continue
            hist_ser, sym_color = entry
            for ts, sig_type in sym_signals:
                try:
                    hist_val = hist_ser.asof(ts)
                except Exception:
                    continue
                y_val = 0.0 if pd.isna(hist_val) else float(hist_val)
                ax_macd.scatter(
                    ts, y_val,
                    marker=_SIG_MARKERS.get(sig_type, 'o'),
                    s=90, color=sym_color,
                    zorder=7, linewidths=1.5, edgecolors='white',
                )

        # Legend: Patch proxies for histograms + auto handles for MACD/Signal lines
        hist_patches = [
            Patch(facecolor=sym_color, alpha=0.55, label=f'{sym} hist')
            for sym, (_, sym_color) in hist_series_by_symbol.items()
        ]
        auto_handles, auto_labels = ax_macd.get_legend_handles_labels()
        ax_macd.legend(
            handles=hist_patches + auto_handles,
            labels=[p.get_label() for p in hist_patches] + auto_labels,
            fontsize=7,
        )

        # Panel 4: RSI
        ax_rsi.set_title('RSI (14)', fontsize=11)
        ax_rsi.set_ylabel('RSI')
        ax_rsi.axhline(70, color='red', linewidth=0.8, linestyle='--', alpha=0.7, label='70')
        ax_rsi.axhline(30, color='green', linewidth=0.8, linestyle='--', alpha=0.7, label='30')
        ax_rsi.set_ylim(0, 100)
        ax_rsi.grid(True, alpha=0.3)
        for i, (symbol, price_series) in enumerate(prices_by_symbol.items()):
            color = colors[i % len(colors)]
            valid = price_series.dropna()
            if len(valid) < 15:
                continue
            closes = valid.tolist()
            idx = valid.index
            rsi_s = calculate_rsi_series(closes)
            rsi_vals = [v if v is not None else float('nan') for v in rsi_s]
            ax_rsi.plot(idx, rsi_vals, linewidth=1.0, color=color, label=symbol)
        ax_rsi.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def save_performance_report(self, path: str) -> None:
        """Save plain-text performance report to path."""
        m = self.get_metrics()
        lines = [
            "=" * 62,
            "  PERFORMANCE REPORT",
            "=" * 62,
            "",
            "── Returns ──────────────────────────────────────────────────",
            f"  Total Return:            ${m.total_return:>14,.2f}",
            f"  Total Return (%):        {m.total_return_pct:>14.2f}%",
            f"  Annualized Return:       {m.annualized_return:>14.2f}%",
            f"  Initial Equity:          ${m.initial_equity:>14,.2f}",
            f"  Final Equity:            ${m.final_equity:>14,.2f}",
            f"  Peak Equity:             ${m.peak_equity:>14,.2f}",
            f"  Total Days:              {m.total_days:>14.1f}",
            f"  Trading Bars:            {m.trading_days:>14d}",
            "",
            "── Risk ─────────────────────────────────────────────────────",
            f"  Sharpe Ratio:            {m.sharpe_ratio:>14.3f}",
            f"  Sortino Ratio:           {m.sortino_ratio:>14.3f}",
            f"  Calmar Ratio:            {m.calmar_ratio:>14.3f}",
            f"  Max Drawdown:            ${m.max_drawdown:>14,.2f}",
            f"  Max Drawdown (%):        {m.max_drawdown_pct:>14.2f}%",
            f"  Max DD Duration:         {m.max_drawdown_duration:>14.1f} days",
            f"  Volatility (ann.):       {m.volatility:>14.2f}%",
            f"  Ulcer Index:             {m.ulcer_index:>14.3f}",
            f"  Skewness:                {m.skewness:>14.3f}",
            f"  Kurtosis:                {m.kurtosis:>14.3f}",
            f"  Best Bar Return:         {m.best_day:>14.2f}%",
            f"  Worst Bar Return:        {m.worst_day:>14.2f}%",
            f"  Avg Bar Return:          {m.avg_daily_return:>14.4f}%",
            "",
            "── Trades ───────────────────────────────────────────────────",
            f"  Total Trades:            {m.total_trades:>14d}",
            f"  Winning Trades:          {m.winning_trades:>14d}",
            f"  Losing Trades:           {m.losing_trades:>14d}",
            f"  Win Rate:                {m.win_rate:>14.1f}%",
            f"  Avg Trade P&L:           ${m.avg_trade_pnl:>14,.2f}",
            f"  Avg Win:                 ${m.avg_win:>14,.2f}",
            f"  Avg Loss:                ${m.avg_loss:>14,.2f}",
            f"  Largest Win:             ${m.largest_win:>14,.2f}",
            f"  Largest Loss:            ${m.largest_loss:>14,.2f}",
            f"  Profit Factor:           {m.profit_factor:>14.3f}",
            f"  Avg Duration:            {m.avg_trade_duration:>14.2f} hrs",
            f"  Avg Bars In Trade:       {m.avg_bars_in_trade:>14.1f}",
            "",
            "── Bracket Orders ───────────────────────────────────────────",
            f"  Bracket Trades:          {m.bracket_trades:>14d}",
            f"  Stop Triggers:           {m.bracket_stop_triggers:>14d}",
            f"  Profit Triggers:         {m.bracket_profit_triggers:>14d}",
            f"  Manual Exits:            {m.bracket_manual_exits:>14d}",
            f"  Stop Rate:               {m.bracket_stop_rate:>14.1f}%",
            f"  Profit Rate:             {m.bracket_profit_rate:>14.1f}%",
            "",
            "=" * 62,
        ]

        # ── Alpha / Benchmark Comparison ─────────────────────────────
        traded_bm: dict = {}
        external_bm: dict = {}
        try:
            traded_bm = self.calculate_benchmark_comparison()
            traded_bm.pop('_comparison', None)   # remove aggregate key; we recompute inline
        except Exception:
            pass
        try:
            external_bm = self.calculate_external_benchmarks()
        except Exception:
            pass

        # Merge both sources; external wins on key collision (more specific data)
        all_benchmarks: dict = {**traded_bm, **external_bm}

        if all_benchmarks:
            def _alpha_str(alpha: float) -> str:
                return f"{'+' if alpha >= 0 else ''}{alpha:.2f}%"

            def _bm_block(symbol: str, bm: dict) -> list[str]:
                alpha = bm.get('alpha', m.total_return_pct - bm['total_return_pct'])
                out = alpha >= 0
                return [
                    "",
                    f"  {symbol}",
                    f"    Total Return:        {bm['total_return_pct']:>12.2f}%"
                    f"    alpha: {_alpha_str(alpha)}",
                    f"    Sharpe Ratio:        {bm['sharpe_ratio']:>12.3f}",
                    f"    Max Drawdown:        {bm['max_drawdown_pct']:>12.2f}%",
                    f"    Volatility (ann.):   {bm['volatility']:>12.2f}%",
                    f"    Strategy Outperforms:{'YES' if out else 'NO':>13}",
                ]

            lines += [
                "",
                "=" * 62,
                "  ALPHA vs. BENCHMARKS",
                "=" * 62,
                "",
                "── Strategy (this run) ──────────────────────────────────",
                f"  Total Return:            {m.total_return_pct:>14.2f}%",
                f"  Annualized Return:       {m.annualized_return:>14.2f}%",
                f"  Sharpe Ratio:            {m.sharpe_ratio:>14.3f}",
                f"  Max Drawdown:            {m.max_drawdown_pct:>14.2f}%",
                f"  Volatility (ann.):       {m.volatility:>14.2f}%",
            ]

            if traded_bm:
                lines += ["", "── Traded Symbols — Buy-and-Hold ────────────────────────"]
                for symbol, bm in traded_bm.items():
                    lines.extend(_bm_block(symbol, bm))
                lines.append("")

            if external_bm:
                lines += ["", "── External Benchmarks ──────────────────────────────────"]
                for symbol, bm in external_bm.items():
                    lines.extend(_bm_block(symbol, bm))
                lines.append("")

            # Summary comparison table (all benchmarks vs strategy)
            lines.append("── Summary ──────────────────────────────────────────────")
            col_w = 10
            headers = ["Strategy"] + list(all_benchmarks.keys())
            header_row = "  " + "".join(h[:col_w].rjust(col_w + 2) for h in headers)
            lines.append(header_row)

            def _row(label: str, strat_val: str, bm_vals: list[str]) -> str:
                cells = [strat_val] + bm_vals
                return "  " + f"{label:<22}" + "".join(v.rjust(col_w + 2) for v in cells)

            lines += [
                _row("Total Return (%)",
                     f"{m.total_return_pct:.2f}%",
                     [f"{b['total_return_pct']:.2f}%" for b in all_benchmarks.values()]),
                _row("Sharpe Ratio",
                     f"{m.sharpe_ratio:.3f}",
                     [f"{b['sharpe_ratio']:.3f}" for b in all_benchmarks.values()]),
                _row("Max Drawdown (%)",
                     f"{m.max_drawdown_pct:.2f}%",
                     [f"{b['max_drawdown_pct']:.2f}%" for b in all_benchmarks.values()]),
                _row("Volatility (%)",
                     f"{m.volatility:.2f}%",
                     [f"{b['volatility']:.2f}%" for b in all_benchmarks.values()]),
                _row("Alpha vs Strategy",
                     "—",
                     [_alpha_str(b.get('alpha', m.total_return_pct - b['total_return_pct']))
                      for b in all_benchmarks.values()]),
                "",
                "=" * 62,
            ]

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def save_signals_orders_csv(self, path: str) -> None:
        """
        Save signals linked to their entry + exit orders.
        Each row is a signal tick. If a trade was entered near that signal,
        it is linked. Unmatched trades appear as extra rows without a signal.
        """
        pf = self.portfolio
        trades = self.get_trades()

        fieldnames = [
            'signal_timestamp', 'signal_type', 'signal_symbol', 'signal_strength',
            'entry_order_id', 'entry_price', 'entry_quantity', 'entry_time',
            'exit_order_id', 'exit_type', 'exit_price', 'exit_quantity', 'exit_time',
            'pnl', 'pnl_pct',
        ]

        # Build signal rows
        rows = []
        if pf.signals_history:
            for ts, signals in sorted(pf.signals_history.items()):
                for sig in signals:
                    rows.append({
                        'signal_timestamp': ts,
                        'signal_type': sig.type.name if hasattr(sig.type, 'name') else str(sig.type),
                        'signal_symbol': sig.symbol,
                        'signal_strength': sig.strength,
                        'entry_order_id': '',
                        'entry_price': '',
                        'entry_quantity': '',
                        'entry_time': '',
                        'exit_order_id': '',
                        'exit_type': '',
                        'exit_price': '',
                        'exit_quantity': '',
                        'exit_time': '',
                        'pnl': '',
                        'pnl_pct': '',
                    })

        # Match trades to signal rows (closest signal with same symbol, within 1 hour)
        unmatched_trades = []
        for trade in trades:
            matched = False
            for row in rows:
                if (row['signal_symbol'] == trade.symbol and row['entry_order_id'] == ''):
                    try:
                        sig_ts = pd.Timestamp(row['signal_timestamp'])
                        entry_ts = pd.Timestamp(trade.entry_time)
                        if sig_ts.tzinfo is None:
                            sig_ts = sig_ts.tz_localize('UTC')
                        if entry_ts.tzinfo is None:
                            entry_ts = entry_ts.tz_localize('UTC')
                        delta_secs = abs((sig_ts - entry_ts).total_seconds())
                    except Exception:
                        delta_secs = float('inf')
                    if delta_secs < 3600:
                        row.update({
                            'entry_order_id': trade.entry_order.order_id,
                            'entry_price': round(trade.entry_price, 4),
                            'entry_quantity': trade.quantity,
                            'entry_time': trade.entry_time,
                            'exit_order_id': trade.exit_order.order_id if trade.exit_order else '',
                            'exit_type': trade.bracket_exit_type or (
                                trade.exit_order.type.name if trade.exit_order else ''),
                            'exit_price': round(trade.exit_price, 4),
                            'exit_quantity': trade.quantity,
                            'exit_time': trade.exit_time,
                            'pnl': round(trade.pnl, 4),
                            'pnl_pct': round(trade.pnl_pct, 4),
                        })
                        matched = True
                        break
            if not matched:
                unmatched_trades.append({
                    'signal_timestamp': '',
                    'signal_type': '',
                    'signal_symbol': trade.symbol,
                    'signal_strength': '',
                    'entry_order_id': trade.entry_order.order_id,
                    'entry_price': round(trade.entry_price, 4),
                    'entry_quantity': trade.quantity,
                    'entry_time': trade.entry_time,
                    'exit_order_id': trade.exit_order.order_id if trade.exit_order else '',
                    'exit_type': trade.bracket_exit_type or (
                        trade.exit_order.type.name if trade.exit_order else ''),
                    'exit_price': round(trade.exit_price, 4),
                    'exit_quantity': trade.quantity,
                    'exit_time': trade.exit_time,
                    'pnl': round(trade.pnl, 4),
                    'pnl_pct': round(trade.pnl_pct, 4),
                })

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows + unmatched_trades)

    def save_trades_csv(self, path: str) -> None:
        """Save one row per completed trade to path."""
        trades = self.get_trades()
        fieldnames = [
            'symbol', 'entry_time', 'exit_time', 'entry_price', 'exit_price',
            'quantity', 'pnl', 'pnl_pct', 'duration_hrs',
            'is_bracket', 'bracket_exit_type',
            'entry_order_id', 'exit_order_id',
        ]
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trade in trades:
                writer.writerow({
                    'symbol': trade.symbol,
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'entry_price': round(trade.entry_price, 4),
                    'exit_price': round(trade.exit_price, 4),
                    'quantity': trade.quantity,
                    'pnl': round(trade.pnl, 4),
                    'pnl_pct': round(trade.pnl_pct, 4),
                    'duration_hrs': round(trade.duration / 3600, 3),
                    'is_bracket': trade.is_bracket,
                    'bracket_exit_type': trade.bracket_exit_type or '',
                    'entry_order_id': trade.entry_order.order_id,
                    'exit_order_id': trade.exit_order.order_id if trade.exit_order else '',
                })

    def save_summary_json(self, path: str) -> None:
        """Save all PerformanceMetrics fields as JSON to path."""
        m = self.get_metrics()
        data = {}
        for field in dataclasses.fields(m):
            v = getattr(m, field.name)
            data[field.name] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2, default=str)

    def save_lifecycle_chart(self, path: str) -> None:
        """
        Save Signal→Entry→Exit lifecycle chart to path.

        3-panel layout (equity + gantt share x-axis):
          1. Equity curve + drawdown fill + per-trade holding spans +
             signal ▲▼, entry ●, exit ★/✕/◆ markers + PnL annotations
          2. Trade Gantt (horizontal bars, entry→exit, colored by outcome)
          3. Trade table (max 25 rows, green/red row coloring)
        """
        from matplotlib.lines import Line2D

        chains = self._build_lifecycle_chains()
        equity = self._get_equity_series()
        n_trades = len(chains)
        n_table_rows = min(n_trades, 25)

        fig_height = 14 + 0.35 * n_table_rows
        fig = plt.figure(figsize=(16, fig_height))
        gs = fig.add_gridspec(3, 1, height_ratios=[5, 2, 3], hspace=0.38)
        ax_eq    = fig.add_subplot(gs[0])
        ax_gantt = fig.add_subplot(gs[1], sharex=ax_eq)
        ax_table = fig.add_subplot(gs[2])

        # ---- Colour / marker scheme ----
        _EXIT_COLOR  = {'PROFIT': 'green', 'STOP': 'red',    'MANUAL': 'darkorange', None: 'gray'}
        _EXIT_MARKER = {'PROFIT': '*',     'STOP': 'X',      'MANUAL': 'D',          None: 'D'}
        _EXIT_LABEL  = {'PROFIT': 'PROFIT ★', 'STOP': 'STOP ✕', 'MANUAL': 'MANUAL ◆'}
        _SPAN_COLOR  = {'PROFIT': 'green', 'STOP': 'red',    'MANUAL': 'orange',     None: 'gray'}
        _SIG_COLOR   = {'BUY': 'limegreen', 'SELL': 'crimson'}
        _SIG_MARKER  = {'BUY': '^', 'SELL': 'v'}

        def _ts(t):
            """UTC-aware pandas Timestamp."""
            ts = pd.Timestamp(t)
            return ts.tz_localize('UTC') if ts.tzinfo is None else ts

        # ================================================================
        # Panel 1 — Equity curve + overlays
        # ================================================================
        ax_eq.plot(equity.index, equity.values, linewidth=1.5, color='steelblue', zorder=2)
        ax_eq.set_title('Signal → Entry → Exit Lifecycle', fontsize=13, fontweight='bold')
        ax_eq.set_ylabel('Portfolio Value ($)')
        ax_eq.grid(True, alpha=0.3)
        ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

        # Drawdown fill (equity below running peak)
        running_max = equity.expanding().max()
        ax_eq.fill_between(
            equity.index, equity.values, running_max.values,
            where=(equity.values < running_max.values),
            color='crimson', alpha=0.07, zorder=1,
        )

        legend_exit_types: set = set()
        for chain in chains:
            t     = chain.trade
            etype = t.bracket_exit_type
            span_c = _SPAN_COLOR.get(etype, 'gray')
            entry_ts = _ts(t.entry_time)
            exit_ts  = _ts(t.exit_time)

            # Holding-period span
            try:
                ax_eq.axvspan(entry_ts, exit_ts, alpha=0.07, color=span_c, zorder=0)
            except Exception:
                pass

            # Entry dot
            try:
                eq_en = equity.asof(entry_ts)
                if not pd.isna(eq_en):
                    ax_eq.scatter(entry_ts, float(eq_en), marker='o', s=60,
                                  color='steelblue', zorder=6, linewidths=1.2,
                                  edgecolors='white')
            except Exception:
                pass

            # Exit marker + PnL annotation
            try:
                eq_ex = equity.asof(exit_ts)
                if not pd.isna(eq_ex):
                    eq_ex    = float(eq_ex)
                    m_color  = _EXIT_COLOR.get(etype, 'gray')
                    m_marker = _EXIT_MARKER.get(etype, 'D')
                    m_size   = 150 if etype == 'PROFIT' else 100
                    ax_eq.scatter(exit_ts, eq_ex, marker=m_marker, s=m_size,
                                  color=m_color, zorder=7, linewidths=1.2,
                                  edgecolors='white')
                    legend_exit_types.add(etype)
                    ax_eq.annotate(
                        f'${t.pnl:+,.0f}',
                        xy=(exit_ts, eq_ex),
                        xytext=(4, 6), textcoords='offset points',
                        fontsize=7, color=m_color, fontweight='bold',
                    )
            except Exception:
                pass

            # Signal marker + arrow → entry
            if chain.signal_time is not None:
                try:
                    sig_ts = _ts(chain.signal_time)
                    eq_sig = equity.asof(sig_ts)
                    if not pd.isna(eq_sig):
                        eq_sig   = float(eq_sig)
                        s_color  = _SIG_COLOR.get(chain.signal_type, 'gray')
                        s_marker = _SIG_MARKER.get(chain.signal_type, 'o')
                        ax_eq.scatter(sig_ts, eq_sig, marker=s_marker, s=100,
                                      color=s_color, zorder=8, linewidths=1.5,
                                      edgecolors='white')
                        # Arrow to entry if different timestamp
                        eq_en2 = equity.asof(entry_ts)
                        if not pd.isna(eq_en2) and sig_ts != entry_ts:
                            ax_eq.annotate(
                                '',
                                xy=(entry_ts, float(eq_en2)),
                                xytext=(sig_ts, eq_sig),
                                arrowprops=dict(arrowstyle='->', color='gray',
                                                lw=0.8, alpha=0.5),
                                zorder=5,
                            )
                except Exception:
                    pass

        ax_eq.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax_eq.xaxis.set_major_locator(mdates.AutoDateLocator())

        # Legend
        leg_handles = [
            Line2D([0], [0], color='steelblue', linewidth=1.5, label='Equity'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='limegreen',
                   markersize=9, label='BUY signal', linestyle='None'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='crimson',
                   markersize=9, label='SELL signal', linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                   markersize=8, label='Entry', linestyle='None'),
        ]
        for et in sorted(legend_exit_types, key=lambda x: (x is None, x or '')):
            leg_handles.append(Line2D(
                [0], [0], marker=_EXIT_MARKER.get(et, 'D'), color='w',
                markerfacecolor=_EXIT_COLOR.get(et, 'gray'),
                markersize=9, label=_EXIT_LABEL.get(et, str(et) if et else 'MARKET'),
                linestyle='None',
            ))
        ax_eq.legend(handles=leg_handles, fontsize=8, loc='upper left', framealpha=0.88)

        # ================================================================
        # Panel 2 — Trade Gantt
        # ================================================================
        ax_gantt.set_title('Holding Periods', fontsize=11)
        ax_gantt.set_ylabel('Trade #')
        ax_gantt.grid(True, alpha=0.2, axis='x')
        if n_trades > 0:
            ax_gantt.set_ylim(-0.5, n_trades - 0.5)
        ax_gantt.invert_yaxis()

        for i, chain in enumerate(chains):
            t       = chain.trade
            etype   = t.bracket_exit_type
            bar_c   = _SPAN_COLOR.get(etype, 'gray')
            en_num  = mdates.date2num(_ts(t.entry_time).to_pydatetime())
            ex_num  = mdates.date2num(_ts(t.exit_time).to_pydatetime())
            width   = max(ex_num - en_num, 0.1)
            ax_gantt.barh(i, width, left=en_num, height=0.6,
                          color=bar_c, alpha=0.75,
                          edgecolor='white', linewidth=0.5)
            suffix = f' [{etype}]' if etype else ''
            bar_label = f'{t.symbol}  ${t.pnl:+,.0f}{suffix}'
            ax_gantt.text(en_num + width / 2, i, bar_label,
                          ha='center', va='center', fontsize=6.5,
                          color='white', fontweight='bold')

        # Signal dotted vertical lines on Gantt
        seen_sig: set = set()
        for chain in chains:
            if chain.signal_time and chain.signal_time not in seen_sig:
                seen_sig.add(chain.signal_time)
                sn = mdates.date2num(_ts(chain.signal_time).to_pydatetime())
                ax_gantt.axvline(sn, color=_SIG_COLOR.get(chain.signal_type, 'gray'),
                                 alpha=0.35, linewidth=0.8, linestyle=':')

        ax_gantt.set_yticks(range(n_trades))
        ax_gantt.set_yticklabels([f'#{i}' for i in range(n_trades)], fontsize=7)
        plt.setp(ax_gantt.get_xticklabels(), visible=False)

        # ================================================================
        # Panel 3 — Table
        # ================================================================
        ax_table.axis('off')
        display = chains[:25]
        if display:
            headers = ['#', 'Sym', 'Signal', 'Sig Time', 'Entry $', 'Entry Time',
                       'Exit $', 'Exit Time', 'Type', 'Dur(h)', 'PnL $', 'PnL%']
            rows_data, pnl_vals = [], []
            for i, chain in enumerate(display):
                t = chain.trade
                rows_data.append([
                    str(i),
                    t.symbol,
                    chain.signal_type or '-',
                    _ts(chain.signal_time).strftime('%m/%d') if chain.signal_time else '-',
                    f'${t.entry_price:,.2f}',
                    _ts(t.entry_time).strftime('%m/%d'),
                    f'${t.exit_price:,.2f}',
                    _ts(t.exit_time).strftime('%m/%d'),
                    t.bracket_exit_type or 'MKT',
                    f'{t.duration / 3600:.1f}',
                    f'${t.pnl:+,.2f}',
                    f'{t.pnl_pct:+.2f}%',
                ])
                pnl_vals.append(t.pnl)

            bbox_y0 = 0.05 if n_trades > 25 else 0.0
            tbl = ax_table.table(
                cellText=rows_data,
                colLabels=headers,
                cellLoc='center',
                loc='center',
                bbox=[0, bbox_y0, 1, 1.0 - bbox_y0],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            for (row, col), cell in tbl.get_celld().items():
                if row == 0:
                    cell.set_facecolor('#343a40')
                    cell.set_text_props(color='white', fontweight='bold')
                else:
                    idx = row - 1
                    if idx < len(pnl_vals):
                        cell.set_facecolor('#d4edda' if pnl_vals[idx] > 0 else '#f8d7da')
            if n_trades > 25:
                ax_table.text(0.5, 0.02,
                              f'… {n_trades - 25} more trades not shown',
                              ha='center', va='bottom', fontsize=8, color='gray',
                              transform=ax_table.transAxes)
        else:
            ax_table.text(0.5, 0.5, 'No trades to display',
                          ha='center', va='center', fontsize=11, color='gray',
                          transform=ax_table.transAxes)

        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def save_lifecycle_chart_interactive(self, path: str) -> None:
        """
        Save interactive lifecycle chart as a self-contained HTML file (Plotly).

        Every signal ▲▼, entry ●, and exit ★/✕/◆ marker has a rich hover tooltip
        showing symbol, prices, P&L, order type, signal details, and duration.
        Holding-period spans are shaded by outcome. Includes a Gantt panel and a
        scrollable trade table with green/red row colouring.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("plotly not installed — skipping lifecycle_interactive.html")
            return

        chains = self._build_lifecycle_chains()
        equity = self._get_equity_series()

        # Cap trades shown in chart — every marker, hover string, Gantt bar,
        # and table row contributes to HTML size. Show the most recent N.
        _MAX_TRADES_CHART = 200
        if len(chains) > _MAX_TRADES_CHART:
            logger.warning(
                f"Interactive chart: showing last {_MAX_TRADES_CHART} of "
                f"{len(chains)} trades to keep HTML size manageable."
            )
            chains = chains[-_MAX_TRADES_CHART:]
        n_trades = len(chains)

        # Downsample equity for the chart when there are many ticks.
        _MAX_EQUITY_POINTS = 2000
        if len(equity) > _MAX_EQUITY_POINTS:
            step = max(1, len(equity) // _MAX_EQUITY_POINTS)
            equity = equity.iloc[::step]

        # ---- Colour / Plotly symbol maps ----
        _EXIT_COLOR = {'PROFIT': 'green', 'STOP': 'red', 'MANUAL': 'darkorange', None: 'gray'}
        _EXIT_SYM   = {'PROFIT': 'star', 'STOP': 'x', 'MANUAL': 'diamond', 'MARKET': 'diamond'}
        _EXIT_SIZE  = {'PROFIT': 16, 'STOP': 12, 'MANUAL': 12, 'MARKET': 12}
        _SIG_COLOR  = {'BUY': 'limegreen', 'SELL': 'crimson'}
        _SIG_SYM    = {'BUY': 'triangle-up', 'SELL': 'triangle-down'}
        _SPAN_RGBA  = {
            'PROFIT': 'rgba(0,128,0,0.08)',
            'STOP':   'rgba(220,20,60,0.08)',
            'MANUAL': 'rgba(255,165,0,0.08)',
            None:     'rgba(128,128,128,0.07)',
        }
        _GANTT_C    = {'PROFIT': 'green', 'STOP': 'red', 'MANUAL': 'orange', None: 'gray'}

        def _ts(t):
            ts = pd.Timestamp(t)
            return ts.tz_localize('UTC') if ts.tzinfo is None else ts

        def _fmt(ts):
            return _ts(ts).strftime('%Y-%m-%d %H:%M') if ts else '—'

        # 3-row layout: equity (55%) / gantt (18%) / table (27%)
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.55, 0.18, 0.27],
            vertical_spacing=0.04,
            subplot_titles=['Signal → Entry → Exit Lifecycle', 'Holding Periods (Gantt)', ''],
            specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "table"}]],
        )

        running_max = equity.expanding().max()

        # ── Drawdown fill: invisible running-max → equity fills "tonexty" ──────
        fig.add_trace(go.Scatter(
            x=equity.index, y=running_max.values,
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip',
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity.values,
            mode='lines', name='Portfolio',
            line=dict(width=1.5, color='steelblue'),
            fill='tonexty', fillcolor='rgba(220,20,60,0.07)',
            hovertemplate='<b>Portfolio</b><br>%{x|%b %d, %Y}<br>$%{y:,.0f}<extra></extra>',
        ), row=1, col=1)

        # ── Holding-period vrects (row 1) ─────────────────────────────────────
        # Cap at 300 to avoid layout bloat with large backtests
        _VRECT_MAX = 100
        for chain in chains[:_VRECT_MAX]:
            t = chain.trade
            try:
                fig.add_vrect(
                    x0=_ts(t.entry_time), x1=_ts(t.exit_time),
                    fillcolor=_SPAN_RGBA.get(t.bracket_exit_type, 'rgba(128,128,128,0.07)'),
                    opacity=1.0, layer='below', line_width=0, row=1, col=1,
                )
            except Exception:
                pass

        # ── Signal markers ────────────────────────────────────────────────────
        sx, sy, st_sig, sc_sig, ss_sig = [], [], [], [], []
        for i, chain in enumerate(chains):
            if chain.signal_time is None:
                continue
            sig_ts = _ts(chain.signal_time)
            eq_val = equity.asof(sig_ts)
            if pd.isna(eq_val):
                continue
            stype = chain.signal_type or 'UNKNOWN'
            sx.append(sig_ts)
            sy.append(float(eq_val))
            st_sig.append(
                f'<b>🔔 Signal: {stype}</b><br>'
                f'Symbol: {chain.trade.symbol}<br>'
                f'Time: {_fmt(chain.signal_time)}<br>'
                f'Strength: {chain.signal_strength if chain.signal_strength is not None else "N/A"}<br>'
                f'<i>→ Trade #{i}: {chain.trade.symbol} entry @ ${chain.trade.entry_price:,.2f}</i>'
            )
            sc_sig.append(_SIG_COLOR.get(stype, 'gray'))
            ss_sig.append(_SIG_SYM.get(stype, 'circle'))
        if sx:
            fig.add_trace(go.Scatter(
                x=sx, y=sy, mode='markers', name='Signal',
                marker=dict(symbol=ss_sig, size=12, color=sc_sig,
                            line=dict(width=1.5, color='white')),
                text=st_sig, hovertemplate='%{text}<extra></extra>',
            ), row=1, col=1)

        # ── Entry markers ─────────────────────────────────────────────────────
        en_x, en_y, en_t = [], [], []
        for i, chain in enumerate(chains):
            t = chain.trade
            entry_ts = _ts(t.entry_time)
            eq_val = equity.asof(entry_ts)
            if pd.isna(eq_val):
                continue
            order_type = 'Bracket' if t.is_bracket else 'Market'
            sig_line = (
                f'Signal: {chain.signal_type} @ {_fmt(chain.signal_time)} '
                f'(strength={chain.signal_strength})'
                if chain.signal_time else 'Signal: (no match within window)'
            )
            en_x.append(entry_ts)
            en_y.append(float(eq_val))
            en_t.append(
                f'<b>● Entry #{i} — {order_type}</b><br>'
                f'Symbol: {t.symbol}<br>'
                f'Time: {_fmt(t.entry_time)}<br>'
                f'Price: ${t.entry_price:,.2f}<br>'
                f'Quantity: {t.quantity:,}<br>'
                f'Order ID: {t.entry_order.order_id if hasattr(t.entry_order, "order_id") else "—"}<br>'
                f'{sig_line}'
            )
        if en_x:
            fig.add_trace(go.Scatter(
                x=en_x, y=en_y, mode='markers', name='Entry',
                marker=dict(symbol='circle', size=9, color='steelblue',
                            line=dict(width=1.5, color='white')),
                text=en_t, hovertemplate='%{text}<extra></extra>',
            ), row=1, col=1)

        # ── Exit markers (grouped by type for legend) ─────────────────────────
        exit_groups: dict[str, dict] = {}
        for i, chain in enumerate(chains):
            t = chain.trade
            etype = t.bracket_exit_type
            key = etype or 'MARKET'
            exit_ts = _ts(t.exit_time)
            eq_val = equity.asof(exit_ts)
            if pd.isna(eq_val):
                continue
            sig_line = (
                f'Signal: {chain.signal_type} @ {_fmt(chain.signal_time)} '
                f'(strength={chain.signal_strength})'
                if chain.signal_time else 'Signal: (no match)'
            )
            hover = (
                f'<b>Exit #{i}: {key}</b><br>'
                f'Symbol: {t.symbol}<br>'
                f'Exit Time: {_fmt(t.exit_time)}<br>'
                f'Exit Price: ${t.exit_price:,.2f}<br>'
                f'Entry Price: ${t.entry_price:,.2f} @ {_fmt(t.entry_time)}<br>'
                f'Quantity: {t.quantity:,}<br>'
                f'<b>P&L: ${t.pnl:+,.2f} ({t.pnl_pct:+.2f}%)</b><br>'
                f'Duration: {t.duration / 3600:.1f}h ({t.duration / 86400:.1f}d)<br>'
                f'Exit Order ID: {t.exit_order.order_id if t.exit_order and hasattr(t.exit_order, "order_id") else "—"}<br>'
                f'{sig_line}'
            )
            if key not in exit_groups:
                exit_groups[key] = {'xs': [], 'ys': [], 'texts': []}
            exit_groups[key]['xs'].append(exit_ts)
            exit_groups[key]['ys'].append(float(eq_val))
            exit_groups[key]['texts'].append(hover)

        for key, data in exit_groups.items():
            etype_key = key if key != 'MARKET' else None
            fig.add_trace(go.Scatter(
                x=data['xs'], y=data['ys'], mode='markers',
                name=f'Exit: {key}',
                marker=dict(
                    symbol=_EXIT_SYM.get(key, 'diamond'),
                    size=_EXIT_SIZE.get(key, 12),
                    color=_EXIT_COLOR.get(etype_key, 'gray'),
                    line=dict(width=1.5, color='white'),
                ),
                text=data['texts'], hovertemplate='%{text}<extra></extra>',
            ), row=1, col=1)

        # ── Trade Gantt (row 2) ───────────────────────────────────────────────
        # Group by exit type so we use O(exit_types) traces instead of O(trades).
        # Each group shares one NaN-separated bar trace + one hover-point trace.
        gantt_groups: dict[str, dict] = {}
        for i, chain in enumerate(chains):
            t = chain.trade
            etype = t.bracket_exit_type
            key   = etype or 'MARKET'
            bar_c = _GANTT_C.get(etype, 'gray')
            en_ts  = _ts(t.entry_time)
            ex_ts  = _ts(t.exit_time)
            mid_ts = en_ts + (ex_ts - en_ts) / 2
            sig_line = (
                f'Signal: {chain.signal_type} @ {_fmt(chain.signal_time)} '
                f'(strength={chain.signal_strength})'
                if chain.signal_time else 'Signal: (no match)'
            )
            gantt_hover = (
                f'<b>Trade #{i}: {t.symbol}</b><br>'
                f'Entry: ${t.entry_price:,.2f} @ {_fmt(t.entry_time)}<br>'
                f'Exit:  ${t.exit_price:,.2f} @ {_fmt(t.exit_time)}<br>'
                f'Exit Type: {etype or "MARKET"}<br>'
                f'Quantity: {t.quantity:,}<br>'
                f'<b>P&L: ${t.pnl:+,.2f} ({t.pnl_pct:+.2f}%)</b><br>'
                f'Duration: {t.duration / 3600:.1f}h<br>'
                f'{sig_line}'
            )
            if key not in gantt_groups:
                gantt_groups[key] = {
                    'bars_x': [], 'bars_y': [],
                    'hover_x': [], 'hover_y': [], 'texts': [],
                    'color': bar_c,
                }
            g = gantt_groups[key]
            # NaN separator keeps bars distinct within a single trace
            g['bars_x'] += [en_ts, ex_ts, None]
            g['bars_y'] += [i, i, None]
            g['hover_x'].append(mid_ts)
            g['hover_y'].append(i)
            g['texts'].append(gantt_hover)

        for key, g in gantt_groups.items():
            fig.add_trace(go.Scatter(
                x=g['bars_x'], y=g['bars_y'],
                mode='lines', line=dict(width=20, color=g['color']),
                showlegend=False, hoverinfo='skip',
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=g['hover_x'], y=g['hover_y'],
                mode='markers',
                marker=dict(size=20, color=g['color'], opacity=0.01),
                text=g['texts'], hovertemplate='%{text}<extra></extra>',
                showlegend=False, name='',
            ), row=2, col=1)

        # Signal dotted vlines on Gantt — cap at 300 to avoid layout bloat
        _VLINE_MAX = 100
        seen_sigs: set = set()
        for chain in chains:
            if len(seen_sigs) >= _VLINE_MAX:
                break
            if chain.signal_time and chain.signal_time not in seen_sigs:
                seen_sigs.add(chain.signal_time)
                sig_c = _SIG_COLOR.get(chain.signal_type, 'gray')
                try:
                    fig.add_vline(
                        x=_ts(chain.signal_time), line_dash='dot',
                        line_color=sig_c, opacity=0.4, line_width=1.5,
                        row=2, col=1,
                    )
                except Exception:
                    pass

        # ── Trade table (row 3) ───────────────────────────────────────────────
        if chains:
            headers = ['#', 'Symbol', 'Signal', 'Sig Time',
                       'Entry $', 'Entry Time', 'Exit $', 'Exit Time',
                       'Type', 'Dur (h)', 'P&L $', 'P&L %']
            col_vals: list[list] = [[] for _ in headers]
            row_fills: list[str] = []
            for i, chain in enumerate(chains):
                t = chain.trade
                vals = [
                    str(i), t.symbol,
                    chain.signal_type or '—',
                    _ts(chain.signal_time).strftime('%m/%d %H:%M') if chain.signal_time else '—',
                    f'${t.entry_price:,.2f}',
                    _ts(t.entry_time).strftime('%m/%d %H:%M'),
                    f'${t.exit_price:,.2f}',
                    _ts(t.exit_time).strftime('%m/%d %H:%M'),
                    t.bracket_exit_type or 'MKT',
                    f'{t.duration / 3600:.1f}',
                    f'${t.pnl:+,.2f}',
                    f'{t.pnl_pct:+.2f}%',
                ]
                for j, v in enumerate(vals):
                    col_vals[j].append(v)
                row_fills.append('#d4edda' if t.pnl > 0 else '#f8d7da')

            fig.add_trace(go.Table(
                header=dict(
                    values=[f'<b>{h}</b>' for h in headers],
                    fill_color='#343a40',
                    font=dict(color='white', size=11),
                    align='center', height=28,
                ),
                cells=dict(
                    values=col_vals,
                    fill_color=[row_fills] * len(headers),
                    align='center',
                    font=dict(size=10),
                    height=24,
                ),
            ), row=3, col=1)

        # ── Layout ────────────────────────────────────────────────────────────
        fig.update_layout(
            title=dict(
                text='Signal → Entry → Exit Lifecycle (Interactive)',
                x=0.5, font=dict(size=15),
            ),
            height=1100,
            hovermode='closest',
            legend=dict(
                x=0.01, y=0.985,
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='rgba(0,0,0,0.2)', borderwidth=1,
            ),
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa',
        )
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
            tickformat='%b %d',
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
        )
        if n_trades > 0:
            fig.update_yaxes(
                tickvals=list(range(n_trades)),
                ticktext=[f'#{i}' for i in range(n_trades)],
                autorange='reversed',
                row=2, col=1,
            )

        fig.write_html(path, include_plotlyjs='cdn')

    def save_combined_lifecycle_chart_interactive(
        self,
        other: "PortfolioAnalyzer",
        path: str,
        self_label: str = "Replay",
        other_label: str = "Live",
        align_start=None,
        market_hours_shading: bool = True,
    ) -> None:
        """
        Save a combined interactive lifecycle HTML chart overlaying two sessions.

        Both equity curves are normalized to % return from their first data point.
        Signal / entry / exit markers are differentiated by marker border and shape.
        Pre/post-market windows are shaded yellow when market_hours_shading=True.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("plotly not installed — skipping combined_lifecycle.html")
            return

        # ── Data preparation ─────────────────────────────────────────────────
        self_eq  = self._get_equity_series()
        other_eq = other._get_equity_series()

        if align_start is None and len(other_eq) > 0:
            align_start = other_eq.index[0]
        if align_start is not None:
            align_ts = pd.to_datetime(align_start, utc=True)
            self_eq  = self_eq[self_eq.index >= align_ts]
            other_eq = other_eq[other_eq.index >= align_ts]

        if len(self_eq) > 0:
            self_eq = (self_eq / self_eq.iloc[0] - 1) * 100
        if len(other_eq) > 0:
            other_eq = (other_eq / other_eq.iloc[0] - 1) * 100

        self_chains  = self._build_lifecycle_chains()
        other_chains = other._build_lifecycle_chains()

        _MAX_TRADES_CHART = 150
        _MAX_EQUITY_POINTS = 2000

        if len(self_chains) > _MAX_TRADES_CHART:
            self_chains = self_chains[-_MAX_TRADES_CHART:]
        if len(other_chains) > _MAX_TRADES_CHART:
            other_chains = other_chains[-_MAX_TRADES_CHART:]

        if len(self_eq) > _MAX_EQUITY_POINTS:
            step = max(1, len(self_eq) // _MAX_EQUITY_POINTS)
            self_eq = self_eq.iloc[::step]
        if len(other_eq) > _MAX_EQUITY_POINTS:
            step = max(1, len(other_eq) // _MAX_EQUITY_POINTS)
            other_eq = other_eq.iloc[::step]

        n_self  = len(self_chains)
        n_other = len(other_chains)

        # ── Colour / symbol scheme ────────────────────────────────────────────
        _SELF_COLOR   = "steelblue"
        _OTHER_COLOR  = "darkorange"
        _SELF_BORDER  = "white"
        _OTHER_BORDER = "black"
        _EXIT_COLOR = {'PROFIT': 'green', 'STOP': 'red', 'MANUAL': 'darkorange', None: 'gray'}
        _EXIT_SYM   = {'PROFIT': 'star', 'STOP': 'x', 'MANUAL': 'diamond', 'MARKET': 'diamond'}
        _EXIT_SIZE  = {'PROFIT': 16, 'STOP': 12, 'MANUAL': 12, 'MARKET': 12}
        _SIG_COLOR  = {'BUY': 'limegreen', 'SELL': 'crimson'}
        _SIG_SYM    = {'BUY': 'triangle-up', 'SELL': 'triangle-down'}
        _SPAN_RGBA  = {
            'PROFIT': 'rgba(0,128,0,0.06)',
            'STOP':   'rgba(220,20,60,0.06)',
            'MANUAL': 'rgba(255,165,0,0.06)',
            None:     'rgba(128,128,128,0.05)',
        }
        _GANTT_C = {'PROFIT': 'green', 'STOP': 'red', 'MANUAL': 'orange', None: 'gray'}

        def _ts(t):
            ts = pd.Timestamp(t)
            return ts.tz_localize('UTC') if ts.tzinfo is None else ts

        def _fmt(ts):
            return _ts(ts).strftime('%Y-%m-%d %H:%M') if ts else '—'

        # ── Layout ────────────────────────────────────────────────────────────
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.55, 0.18, 0.27],
            vertical_spacing=0.04,
            subplot_titles=[
                f'Signal → Entry → Exit Lifecycle: {self_label} vs {other_label}',
                'Holding Periods (Gantt)',
                '',
            ],
            specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "table"}]],
        )

        # ── Market-hours shading (row 1 only, layer=below) ────────────────────
        if market_hours_shading:
            combined_index = self_eq.index.union(other_eq.index)
            if len(combined_index) > 0:
                vrects = self._market_closed_vrects(combined_index)
                _MAX_VRECTS = 200
                for x0, x1 in vrects[:_MAX_VRECTS]:
                    try:
                        fig.add_vrect(
                            x0=x0, x1=x1,
                            fillcolor='rgba(255,215,0,0.25)',
                            opacity=1.0, layer='below', line_width=0,
                            row=1, col=1,
                        )
                    except Exception:
                        pass

        # ── Row 1: equity curves + drawdown fills ─────────────────────────────
        def _add_equity(eq, label, line_color, fill_color):
            if len(eq) == 0:
                return
            running_max = eq.expanding().max()
            fig.add_trace(go.Scatter(
                x=eq.index, y=running_max.values,
                mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip',
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=eq.index, y=eq.values,
                mode='lines', name=label,
                line=dict(width=1.5, color=line_color),
                fill='tonexty', fillcolor=fill_color,
                hovertemplate=f'<b>{label}</b><br>%{{x|%b %d, %Y %H:%M}}<br>%{{y:.2f}}%<extra></extra>',
            ), row=1, col=1)

        _add_equity(self_eq,  self_label,  _SELF_COLOR,  'rgba(70,130,180,0.07)')
        _add_equity(other_eq, other_label, _OTHER_COLOR, 'rgba(255,140,0,0.07)')

        # ── Holding-period vrects (row 1) ─────────────────────────────────────
        _VRECT_MAX = 100

        def _add_vrects(chains):
            for chain in chains[:_VRECT_MAX]:
                t = chain.trade
                try:
                    fig.add_vrect(
                        x0=_ts(t.entry_time), x1=_ts(t.exit_time),
                        fillcolor=_SPAN_RGBA.get(t.bracket_exit_type, 'rgba(128,128,128,0.05)'),
                        opacity=1.0, layer='below', line_width=0, row=1, col=1,
                    )
                except Exception:
                    pass

        _add_vrects(self_chains)
        _add_vrects(other_chains)

        # ── Signal markers ────────────────────────────────────────────────────
        def _add_signal_markers(chains, label, border_color):
            sx, sy, st, sc, ss = [], [], [], [], []
            for i, chain in enumerate(chains):
                if chain.signal_time is None:
                    continue
                sig_ts = _ts(chain.signal_time)
                eq = self_eq if label == self_label else other_eq
                eq_val = eq.asof(sig_ts)
                if pd.isna(eq_val):
                    continue
                stype = chain.signal_type or 'UNKNOWN'
                sx.append(sig_ts)
                sy.append(float(eq_val))
                st.append(
                    f'<b>🔔 Signal: {stype} ({label})</b><br>'
                    f'Symbol: {chain.trade.symbol}<br>'
                    f'Time: {_fmt(chain.signal_time)}<br>'
                    f'Strength: {chain.signal_strength if chain.signal_strength is not None else "N/A"}<br>'
                    f'<i>→ Trade #{i}: entry @ ${chain.trade.entry_price:,.2f}</i>'
                )
                sc.append(_SIG_COLOR.get(stype, 'gray'))
                ss.append(_SIG_SYM.get(stype, 'circle'))
            if sx:
                fig.add_trace(go.Scatter(
                    x=sx, y=sy, mode='markers', name=f'Signal ({label})',
                    marker=dict(symbol=ss, size=12, color=sc,
                                line=dict(width=1.5, color=border_color)),
                    text=st, hovertemplate='%{text}<extra></extra>',
                ), row=1, col=1)

        _add_signal_markers(self_chains,  self_label,  _SELF_BORDER)
        _add_signal_markers(other_chains, other_label, _OTHER_BORDER)

        # ── Entry markers ─────────────────────────────────────────────────────
        def _add_entry_markers(chains, label, line_color, marker_symbol, border_color):
            en_x, en_y, en_t = [], [], []
            eq = self_eq if label == self_label else other_eq
            for i, chain in enumerate(chains):
                t = chain.trade
                entry_ts = _ts(t.entry_time)
                eq_val = eq.asof(entry_ts)
                if pd.isna(eq_val):
                    continue
                order_type = 'Bracket' if t.is_bracket else 'Market'
                sig_line = (
                    f'Signal: {chain.signal_type} @ {_fmt(chain.signal_time)} '
                    f'(strength={chain.signal_strength})'
                    if chain.signal_time else 'Signal: (no match within window)'
                )
                en_x.append(entry_ts)
                en_y.append(float(eq_val))
                en_t.append(
                    f'<b>● Entry #{i} — {order_type} ({label})</b><br>'
                    f'Symbol: {t.symbol}<br>'
                    f'Time: {_fmt(t.entry_time)}<br>'
                    f'Price: ${t.entry_price:,.2f}<br>'
                    f'Quantity: {t.quantity:,}<br>'
                    f'{sig_line}'
                )
            if en_x:
                fig.add_trace(go.Scatter(
                    x=en_x, y=en_y, mode='markers', name=f'Entry ({label})',
                    marker=dict(symbol=marker_symbol, size=9, color=line_color,
                                line=dict(width=1.5, color=border_color)),
                    text=en_t, hovertemplate='%{text}<extra></extra>',
                ), row=1, col=1)

        _add_entry_markers(self_chains,  self_label,  _SELF_COLOR,  'circle', _SELF_BORDER)
        _add_entry_markers(other_chains, other_label, _OTHER_COLOR, 'square', _OTHER_BORDER)

        # ── Exit markers (grouped by type) ────────────────────────────────────
        def _add_exit_markers(chains, label, border_color):
            eq = self_eq if label == self_label else other_eq
            exit_groups: dict[str, dict] = {}
            for i, chain in enumerate(chains):
                t = chain.trade
                etype = t.bracket_exit_type
                key = etype or 'MARKET'
                exit_ts = _ts(t.exit_time)
                eq_val = eq.asof(exit_ts)
                if pd.isna(eq_val):
                    continue
                sig_line = (
                    f'Signal: {chain.signal_type} @ {_fmt(chain.signal_time)} '
                    f'(strength={chain.signal_strength})'
                    if chain.signal_time else 'Signal: (no match)'
                )
                hover = (
                    f'<b>Exit #{i}: {key} ({label})</b><br>'
                    f'Symbol: {t.symbol}<br>'
                    f'Exit Time: {_fmt(t.exit_time)}<br>'
                    f'Exit Price: ${t.exit_price:,.2f}<br>'
                    f'Entry Price: ${t.entry_price:,.2f} @ {_fmt(t.entry_time)}<br>'
                    f'Quantity: {t.quantity:,}<br>'
                    f'<b>P&L: ${t.pnl:+,.2f} ({t.pnl_pct:+.2f}%)</b><br>'
                    f'Duration: {t.duration / 3600:.1f}h<br>'
                    f'{sig_line}'
                )
                if key not in exit_groups:
                    exit_groups[key] = {'xs': [], 'ys': [], 'texts': []}
                exit_groups[key]['xs'].append(exit_ts)
                exit_groups[key]['ys'].append(float(eq_val))
                exit_groups[key]['texts'].append(hover)

            for key, data in exit_groups.items():
                etype_key = key if key != 'MARKET' else None
                fig.add_trace(go.Scatter(
                    x=data['xs'], y=data['ys'], mode='markers',
                    name=f'Exit: {key} ({label})',
                    marker=dict(
                        symbol=_EXIT_SYM.get(key, 'diamond'),
                        size=_EXIT_SIZE.get(key, 12),
                        color=_EXIT_COLOR.get(etype_key, 'gray'),
                        line=dict(width=1.5, color=border_color),
                    ),
                    text=data['texts'], hovertemplate='%{text}<extra></extra>',
                ), row=1, col=1)

        _add_exit_markers(self_chains,  self_label,  _SELF_BORDER)
        _add_exit_markers(other_chains, other_label, _OTHER_BORDER)

        # ── Row 2: Gantt ──────────────────────────────────────────────────────
        # Replay trades: y = 0..N-1 ; Live trades: y = N+1..N+M (gap of 1)
        gantt_gap = 1

        def _add_gantt(chains, y_offset, label_suffix):
            gantt_groups: dict[str, dict] = {}
            for i, chain in enumerate(chains):
                t = chain.trade
                etype = t.bracket_exit_type
                key = etype or 'MARKET'
                bar_c = _GANTT_C.get(etype, 'gray')
                en_ts  = _ts(t.entry_time)
                ex_ts  = _ts(t.exit_time)
                mid_ts = en_ts + (ex_ts - en_ts) / 2
                sig_line = (
                    f'Signal: {chain.signal_type} @ {_fmt(chain.signal_time)} '
                    f'(strength={chain.signal_strength})'
                    if chain.signal_time else 'Signal: (no match)'
                )
                gantt_hover = (
                    f'<b>Trade #{i}: {t.symbol} ({label_suffix})</b><br>'
                    f'Entry: ${t.entry_price:,.2f} @ {_fmt(t.entry_time)}<br>'
                    f'Exit:  ${t.exit_price:,.2f} @ {_fmt(t.exit_time)}<br>'
                    f'Exit Type: {etype or "MARKET"}<br>'
                    f'Quantity: {t.quantity:,}<br>'
                    f'<b>P&L: ${t.pnl:+,.2f} ({t.pnl_pct:+.2f}%)</b><br>'
                    f'Duration: {t.duration / 3600:.1f}h<br>'
                    f'{sig_line}'
                )
                y_pos = y_offset + i
                if key not in gantt_groups:
                    gantt_groups[key] = {
                        'bars_x': [], 'bars_y': [],
                        'hover_x': [], 'hover_y': [], 'texts': [],
                        'color': bar_c,
                    }
                g = gantt_groups[key]
                g['bars_x'] += [en_ts, ex_ts, None]
                g['bars_y'] += [y_pos, y_pos, None]
                g['hover_x'].append(mid_ts)
                g['hover_y'].append(y_pos)
                g['texts'].append(gantt_hover)

            for key, g in gantt_groups.items():
                fig.add_trace(go.Scatter(
                    x=g['bars_x'], y=g['bars_y'],
                    mode='lines', line=dict(width=20, color=g['color']),
                    showlegend=False, hoverinfo='skip',
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=g['hover_x'], y=g['hover_y'],
                    mode='markers',
                    marker=dict(size=20, color=g['color'], opacity=0.01),
                    text=g['texts'], hovertemplate='%{text}<extra></extra>',
                    showlegend=False, name='',
                ), row=2, col=1)

        _add_gantt(self_chains,  y_offset=0,                          label_suffix=self_label)
        _add_gantt(other_chains, y_offset=n_self + gantt_gap,         label_suffix=other_label)

        # Separator line between the two groups
        if n_self > 0 and n_other > 0:
            sep_y = n_self + gantt_gap / 2
            fig.add_hline(
                y=sep_y, line_dash='dash', line_color='gray', opacity=0.5,
                row=2, col=1,
            )

        # Signal vlines on Gantt for both sources
        _VLINE_MAX = 100
        seen_sigs: set = set()
        for chain in list(self_chains) + list(other_chains):
            if len(seen_sigs) >= _VLINE_MAX:
                break
            if chain.signal_time and chain.signal_time not in seen_sigs:
                seen_sigs.add(chain.signal_time)
                sig_c = _SIG_COLOR.get(chain.signal_type, 'gray')
                try:
                    fig.add_vline(
                        x=_ts(chain.signal_time), line_dash='dot',
                        line_color=sig_c, opacity=0.4, line_width=1.5,
                        row=2, col=1,
                    )
                except Exception:
                    pass

        # Gantt y-axis labels
        total_trades = n_self + n_other
        if total_trades > 0:
            tick_vals  = list(range(n_self)) + list(range(n_self + gantt_gap, n_self + gantt_gap + n_other))
            tick_texts = [f'#{i}(R)' for i in range(n_self)] + [f'#{i}(L)' for i in range(n_other)]
            fig.update_yaxes(
                tickvals=tick_vals,
                ticktext=tick_texts,
                autorange='reversed',
                row=2, col=1,
            )

        # ── Row 3: combined trade table ───────────────────────────────────────
        all_chains = [(c, 'R', self_label) for c in self_chains] + [(c, 'L', other_label) for c in other_chains]
        if all_chains:
            headers = ['Src', '#', 'Symbol', 'Signal', 'Sig Time',
                       'Entry $', 'Entry Time', 'Exit $', 'Exit Time',
                       'Type', 'Dur (h)', 'P&L $', 'P&L %']
            col_vals: list[list] = [[] for _ in headers]
            row_fills: list[str] = []
            for i, (chain, src, lbl) in enumerate(all_chains):
                t = chain.trade
                vals = [
                    src, str(i), t.symbol,
                    chain.signal_type or '—',
                    _ts(chain.signal_time).strftime('%m/%d %H:%M') if chain.signal_time else '—',
                    f'${t.entry_price:,.2f}',
                    _ts(t.entry_time).strftime('%m/%d %H:%M'),
                    f'${t.exit_price:,.2f}',
                    _ts(t.exit_time).strftime('%m/%d %H:%M'),
                    t.bracket_exit_type or 'MKT',
                    f'{t.duration / 3600:.1f}',
                    f'${t.pnl:+,.2f}',
                    f'{t.pnl_pct:+.2f}%',
                ]
                for j, v in enumerate(vals):
                    col_vals[j].append(v)
                if src == 'R':
                    row_fills.append('#d0e8f8' if t.pnl > 0 else '#f8d7da')
                else:
                    row_fills.append('#fde9d0' if t.pnl > 0 else '#f8d7da')

            fig.add_trace(go.Table(
                header=dict(
                    values=[f'<b>{h}</b>' for h in headers],
                    fill_color='#343a40',
                    font=dict(color='white', size=11),
                    align='center', height=28,
                ),
                cells=dict(
                    values=col_vals,
                    fill_color=[row_fills] * len(headers),
                    align='center',
                    font=dict(size=10),
                    height=24,
                ),
            ), row=3, col=1)

        # ── Final layout ──────────────────────────────────────────────────────
        fig.update_layout(
            title=dict(
                text=f'Signal → Entry → Exit Lifecycle: {self_label} vs {other_label}',
                x=0.5, font=dict(size=15),
            ),
            height=1100,
            hovermode='closest',
            legend=dict(
                x=0.01, y=0.985,
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='rgba(0,0,0,0.2)', borderwidth=1,
            ),
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa',
        )
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
            tickformat='%b %d',
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
        )
        fig.update_yaxes(
            tickformat='.1f', ticksuffix='%',
            row=1, col=1,
        )

        fig.write_html(path, include_plotlyjs='cdn')

    # -----------------------------------------------------------------------
    # AnalysisEngine-compatible interface
    # Allows drop-in replacement: swap AnalysisEngine for PortfolioAnalyzer
    # without modifying any calling code.
    # -----------------------------------------------------------------------

    def extract_trades(self) -> list:
        """Alias for get_trades() — AnalysisEngine compatible."""
        return self.get_trades()

    def calculate_metrics(self):
        """Alias for get_metrics() — AnalysisEngine compatible."""
        return self.get_metrics()

    def generate_report(self) -> str:
        """Return the performance report as a string — AnalysisEngine compatible."""
        import io
        buf = io.StringIO()
        # Reuse save_performance_report by writing to a temp path then reading it
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp_path = tmp.name
        try:
            self.save_performance_report(tmp_path)
            with open(tmp_path, encoding='utf-8') as f:
                return f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def get_tick_returns(self) -> pd.Series:
        """Tick-by-tick portfolio returns — AnalysisEngine compatible."""
        pf = self.portfolio
        if not pf.tick_history or not pf.value_history:
            raise ValueError("Portfolio must have keep_history=True")
        equity = pd.Series(list(pf.value_history.values()), dtype=float)
        equity.index = pd.to_datetime(list(pf.tick_history.keys()), utc=True)
        return equity.pct_change().dropna()

    def get_daily_returns(self) -> pd.Series:
        """Daily portfolio returns — AnalysisEngine compatible."""
        pf = self.portfolio
        if not pf.value_history:
            raise ValueError("Portfolio must have keep_history=True")
        equity = pd.Series(list(pf.value_history.values()), dtype=float)
        equity.index = pd.to_datetime(list(pf.tick_history.keys()), utc=True)
        return equity.resample('D').last().dropna().pct_change().dropna()

    def get_monthly_returns(self) -> pd.Series:
        """Monthly portfolio returns — AnalysisEngine compatible."""
        pf = self.portfolio
        if not pf.value_history:
            raise ValueError("Portfolio must have keep_history=True")
        equity = pd.Series(list(pf.value_history.values()), dtype=float)
        equity.index = pd.to_datetime(list(pf.tick_history.keys()), utc=True)
        return equity.resample('ME').last().dropna().pct_change().dropna()

    def analyze_bracket_effectiveness(self) -> dict:
        """Bracket order exit analysis — AnalysisEngine compatible."""
        trades = self.get_trades()
        bracket_trades = [t for t in trades if t.is_bracket]
        if not bracket_trades:
            return {"total_bracket_trades": 0, "message": "No bracket trades found"}
        stop_trades = [t for t in bracket_trades if t.bracket_exit_type == "STOP"]
        profit_trades = [t for t in bracket_trades if t.bracket_exit_type == "PROFIT"]
        manual_trades = [t for t in bracket_trades if t.bracket_exit_type == "MANUAL"]

        def _stats(ts):
            if not ts:
                return {}
            return {
                "count": len(ts),
                "avg_pnl": float(np.mean([t.pnl for t in ts])),
                "total_pnl": float(sum(t.pnl for t in ts)),
                "avg_pnl_pct": float(np.mean([t.pnl_pct for t in ts])),
                "avg_duration_hours": float(np.mean([t.duration / 3600 for t in ts])),
                "win_rate": len([t for t in ts if t.pnl > 0]) / len(ts) * 100,
            }

        return {
            "total_bracket_trades": len(bracket_trades),
            "stop_loss_exits": _stats(stop_trades),
            "profit_taker_exits": _stats(profit_trades),
            "manual_exits": _stats(manual_trades),
            "overall": {
                "avg_pnl": float(np.mean([t.pnl for t in bracket_trades])),
                "total_pnl": float(sum(t.pnl for t in bracket_trades)),
                "win_rate": len([t for t in bracket_trades if t.pnl > 0]) / len(bracket_trades) * 100,
                "stop_rate": len(stop_trades) / len(bracket_trades) * 100,
                "profit_rate": len(profit_trades) / len(bracket_trades) * 100,
                "manual_rate": len(manual_trades) / len(bracket_trades) * 100,
            },
        }

    def _compute_single_benchmark_metrics(self, prices: pd.Series, initial_equity: float) -> dict:
        """Shared buy-and-hold computation from a close-price series — AnalysisEngine compatible."""
        if len(prices) < 2:
            return {}
        initial_price = float(prices.iloc[0])
        final_price = float(prices.iloc[-1])
        total_return_pct = ((final_price - initial_price) / initial_price) * 100
        shares = initial_equity / initial_price
        total_return_dollars = shares * final_price - initial_equity
        returns = prices.pct_change().fillna(0)
        volatility = float(returns.std() * np.sqrt(252 * 78) * 100)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown_pct = float(drawdown.min() * 100)
        sharpe = (
            (returns.mean() * 252 * 78) / (returns.std() * np.sqrt(252 * 78))
            if returns.std() > 0 else 0.0
        )
        return {
            'total_return_pct': total_return_pct,
            'total_return_dollars': total_return_dollars,
            'volatility': volatility,
            'sharpe_ratio': float(sharpe),
            'max_drawdown_pct': max_drawdown_pct,
            'initial_price': initial_price,
            'final_price': final_price,
        }

    def calculate_benchmark_comparison(self) -> dict:
        """
        Buy-and-hold metrics for all symbols in tick_history — AnalysisEngine compatible.
        Returns {symbol: metrics_dict, '_comparison': comparison_dict}.
        """
        pf = self.portfolio
        if not pf.tick_history:
            return {}
        symbol_prices: dict = defaultdict(list)
        for ts, tick in pf.tick_history.items():
            for pd_obj in tick:
                symbol_prices[pd_obj.symbol].append(pd_obj.close)
        if not symbol_prices:
            return {}
        m = self.get_metrics()
        initial_equity = m.initial_equity
        results = {}
        for symbol, prices in symbol_prices.items():
            ps = pd.Series(prices, dtype=float)
            bm = self._compute_single_benchmark_metrics(ps, initial_equity)
            if bm:
                bm['price_change_pct'] = bm['total_return_pct']
                results[symbol] = bm
        if results:
            avg_return = float(np.mean([b['total_return_pct'] for b in results.values()]))
            avg_sharpe = float(np.mean([b['sharpe_ratio'] for b in results.values()]))
            avg_vol = float(np.mean([b['volatility'] for b in results.values()]))
            results['_comparison'] = {
                'portfolio_return_pct': m.total_return_pct,
                'benchmark_return_pct': avg_return,
                'alpha': m.total_return_pct - avg_return,
                'portfolio_sharpe': m.sharpe_ratio,
                'benchmark_sharpe': avg_sharpe,
                'portfolio_volatility': m.volatility,
                'benchmark_volatility': avg_vol,
                'outperformance': m.total_return_pct > avg_return,
            }
        return results

    def calculate_external_benchmarks(self) -> dict:
        """
        Load external benchmark CSVs and compute buy-and-hold metrics — AnalysisEngine compatible.
        Set self._benchmark_paths = {"SPY": "data/SPY_5min.csv"} before calling, or pass
        benchmark_paths to run_full_analysis().
        """
        if not getattr(self, '_benchmark_paths', {}):
            return {}
        pf = self.portfolio
        if not pf.tick_history:
            return {}
        ticks = list(pf.tick_history.keys())
        port_start = pd.Timestamp(min(ticks))
        port_end = pd.Timestamp(max(ticks))
        if port_start.tzinfo is None:
            port_start = port_start.tz_localize('UTC')
            port_end = port_end.tz_localize('UTC')
        m = self.get_metrics()
        initial_equity = m.initial_equity
        results = {}
        for symbol, csv_path in self._benchmark_paths.items():
            try:
                df = pd.read_csv(csv_path)
                if 'timestamp' not in df.columns or 'close' not in df.columns:
                    logger.warning(f"Benchmark CSV for {symbol} missing 'timestamp'/'close'")
                    continue
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if df['timestamp'].dt.tz is None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
                else:
                    df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
                if 'symbol' in df.columns and symbol in df['symbol'].values:
                    df = df[df['symbol'] == symbol]
                mask = (df['timestamp'] >= port_start) & (df['timestamp'] <= port_end)
                df = df[mask].sort_values('timestamp')
                if df.empty:
                    continue
                prices = df.set_index('timestamp')['close'].astype(float)
                bm = self._compute_single_benchmark_metrics(prices, initial_equity)
                if bm:
                    bm['alpha'] = m.total_return_pct - bm['total_return_pct']
                    bm['outperformance'] = m.total_return_pct > bm['total_return_pct']
                    results[symbol] = bm
            except Exception as e:
                logger.warning(f"Failed to load benchmark {symbol} from '{csv_path}': {e}")
        return results

    def plot_equity_curve(self, show: bool = False, save_path: Optional[str] = None):
        """Return equity curve figure — AnalysisEngine compatible."""
        equity = self._get_equity_series()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(equity.index, equity.values, linewidth=1.5, color='steelblue')
        ax.set_title('Portfolio Equity Curve', fontsize=13)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    def plot_portfolio_with_trades(self, show: bool = False, save_path: Optional[str] = None):
        """Equity curve with BUY/SELL order markers — AnalysisEngine compatible."""
        from trading.core.classes import OrderType, OrderAction, OrderStatus
        equity = self._get_equity_series()
        om = self.portfolio.om
        buy_times, sell_times = [], []
        if om:
            all_orders = {**om.filled_orders_by_id, **om.pending_orders_by_id}
            bracket_ids = {
                o.order_id for o in all_orders.values()
                if o.type == OrderType.BRACKET and o.action == OrderAction.BUY
            }
            for order in all_orders.values():
                if order.quantity <= 0:
                    continue
                t = order.executed_datetime or order.placed_datetime
                if order.type == OrderType.BRACKET and order.action == OrderAction.BUY:
                    if order.status in {OrderStatus.PENDING_SALE, OrderStatus.FILLED} or order.executed_datetime:
                        buy_times.append(t)
                elif order.status == OrderStatus.FILLED:
                    has_parent = hasattr(order, 'parent_id') and order.parent_id in bracket_ids
                    if has_parent and order.action == OrderAction.SELL:
                        sell_times.append(t)
                    elif not has_parent and not hasattr(order, 'parent_id'):
                        (buy_times if order.action == OrderAction.BUY else sell_times).append(t)
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(equity.index, equity.values, linewidth=2, color='blue', label='Portfolio Value')
        if buy_times:
            buy_ts = [pd.Timestamp(t, tz='UTC') for t in buy_times]
            bv = [equity.asof(ts) for ts in buy_ts]
            ax.scatter(buy_ts, bv, marker='x', s=150, c='green', linewidths=3, label='BUY', zorder=5)
        if sell_times:
            sell_ts = [pd.Timestamp(t, tz='UTC') for t in sell_times]
            sv = [equity.asof(ts) for ts in sell_ts]
            ax.scatter(sell_ts, sv, marker='x', s=150, c='red', linewidths=3, label='SELL', zorder=5)
        ax.set_title('Portfolio Value with Trade Markers', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    def plot_stock_performance(self, show: bool = False, save_path: Optional[str] = None):
        """Stock prices and normalized returns — AnalysisEngine compatible."""
        prices_by_symbol = self._get_price_series_by_symbol()
        if not prices_by_symbol:
            raise ValueError("No price data found in tick history")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        for symbol, series in sorted(prices_by_symbol.items()):
            ax1.plot(series.index, series.values, linewidth=2, label=symbol, marker='o', markersize=3)
        ax1.set_title('Stock Prices Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        for symbol, series in sorted(prices_by_symbol.items()):
            if len(series) > 1:
                norm = (series / series.iloc[0]) * 100
                ax2.plot(series.index, norm.values, linewidth=2, label=symbol, marker='o', markersize=3)
        ax2.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Initial Value')
        ax2.set_title('Stock Returns (Normalized to 100)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Normalized Value')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    def _contribute_to_run(
        self,
        client,
        metric_prefix: str = "",
        artifact_prefix: str = "",
        log_charts: bool = True,
        log_trades: bool = True,
        log_signals: bool = True,
        log_report: bool = True,
    ) -> None:
        """Backward-compatible wrapper over the shared reporting service."""
        from trading.reporting import AnalyzerReportTarget, ExperimentReporter

        summary = ExperimentReporter.build_single_report(
            analyzer=self,
            experiment_name=None,
            run_name=None,
            description=None,
            tags=None,
            parameters=None,
        )[1]
        target = AnalyzerReportTarget(
            name="compat",
            analyzer=self,
            summary=summary,
            metric_prefix=metric_prefix,
            artifact_prefix=artifact_prefix,
            log_charts=log_charts,
            log_trades=log_trades,
            log_signals=log_signals,
            log_report=log_report,
        )
        ExperimentReporter._log_target(client, target)

    def log_to_mlflow(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[dict] = None,
        parameters: Optional[dict] = None,
        log_charts: bool = True,
        log_trades: bool = True,
        log_signals: bool = True,
        log_report: bool = True,
        chart_dpi: int = 150,
        artifact_paths: Optional[list] = None,
    ) -> None:
        """Log analysis to MLflow through the shared reporting service."""
        from trading.reporting import AnalyzerReportTarget, ExperimentReport, ExperimentReporter

        merged_params = {**self._session_metadata, **(parameters or {})}
        report = ExperimentReport(
            experiment_name=experiment_name,
            run_name=run_name,
            description=description,
            tags=tags,
            parameters=_flatten_dict(merged_params) if merged_params else {},
            analyzers=[
                AnalyzerReportTarget(
                    name="primary",
                    analyzer=self,
                    summary=ExperimentReporter.build_single_report(
                        analyzer=self,
                        experiment_name=experiment_name,
                        run_name=run_name,
                        description=description,
                        tags=tags,
                        parameters=None,
                    )[1],
                    log_charts=log_charts,
                    log_trades=log_trades,
                    log_signals=log_signals,
                    log_report=log_report,
                )
            ],
            config_artifact_paths=list(artifact_paths or []),
        )
        ExperimentReporter.log_to_mlflow(report)

    def run_full_analysis(
        self,
        log_to_mlflow: bool = True,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[dict] = None,
        parameters: Optional[dict] = None,
        save_charts_locally: bool = False,
        save_report_locally: bool = False,
        output_dir: str = ".",
        show_summary: bool = True,
        artifact_paths: Optional[list] = None,
        benchmark_paths: Optional[dict] = None,
        result_output_dir: Optional[str] = None,
        analysis_sinks: Optional[list] = None,
    ) -> dict:
        """
        Full analysis pipeline — same signature and return dict as AnalysisEngine.run_full_analysis().

        Returns:
            {"trades", "metrics", "tick_returns", "daily_returns", "monthly_returns",
             "bracket_analysis", "report", "benchmarks"}
        """
        from trading.reporting import ExperimentReporter

        report_obj, summary = ExperimentReporter.build_single_report(
            analyzer=self,
            experiment_name=experiment_name,
            run_name=run_name,
            description=description,
            tags=tags,
            parameters=parameters,
            benchmark_paths=benchmark_paths,
            config_artifact_paths=[path for path in (artifact_paths or []) if path],
        )

        if show_summary:
            ExperimentReporter.show_summary(summary)

        if save_report_locally:
            rpath = os.path.join(output_dir, "backtest_report.txt")
            with open(rpath, 'w', encoding='utf-8') as f:
                f.write(summary.report)

        if save_charts_locally:
            os.makedirs(output_dir, exist_ok=True)
            self.save_all(output_dir)

        sinks = list(analysis_sinks or [])
        if result_output_dir:
            from trading.reporting import LocalRunResultSink

            sinks.append(LocalRunResultSink(result_output_dir))
        if log_to_mlflow:
            from trading.reporting import MlflowAnalysisSink

            sinks.append(MlflowAnalysisSink.from_report(report_obj))
        if sinks:
            try:
                ExperimentReporter.log(report_obj, sinks)
            except Exception as e:
                logger.warning(f"Analysis reporting failed: {e}", exc_info=True)

        return ExperimentReporter.summary_to_legacy_dict(summary)

    # -----------------------------------------------------------------------
    # MongoDB factory classmethods
    # -----------------------------------------------------------------------

    @staticmethod
    def _resolve_mongo_params(connection_uri, database):
        """Return (uri, db) falling back to ConfigManager state_store config."""
        try:
            from utils.config_manager import ConfigManager
            cfg = ConfigManager().get('state_store') or {}
        except Exception:
            cfg = {}
        uri = connection_uri or cfg.get('connection_uri', 'mongodb://localhost:27017')
        db  = database       or cfg.get('database',        'trading')
        return uri, db

    @classmethod
    def from_mongodb(
        cls,
        session_id: str,
        connection_uri: str = None,
        database: str = None,
        start: datetime = None,
        end: datetime = None,
    ) -> "PortfolioAnalyzer":
        """Reconstruct a PortfolioAnalyzer from a stored MongoDB session.

        Falls back to ``state_store`` config for connection details if not provided.

        Args:
            session_id:     Session UUID to load.
            connection_uri: MongoDB connection string (overrides config).
            database:       Database name (overrides config).
            start:          Optional start datetime filter.
            end:            Optional end datetime filter.

        Returns:
            PortfolioAnalyzer ready for analysis.

        Examples::

            # Config-driven connection
            analyzer = PortfolioAnalyzer.from_mongodb("a1b2c3d4-...")
            result = analyzer.run_analysis(output_dir="output/session_abc")

            # Explicit connection
            analyzer = PortfolioAnalyzer.from_mongodb(
                "a1b2c3d4-...",
                connection_uri="mongodb://z440.lan:27017",
                database="trading_live",
            )

            # Date-filtered slice
            analyzer = PortfolioAnalyzer.from_mongodb(
                "a1b2c3d4-...",
                start=datetime(2024, 6, 1),
                end=datetime(2024, 9, 30),
            )
        """
        import types
        from utils.trading_state_store import TradingStateStore

        uri, db = cls._resolve_mongo_params(connection_uri, database)
        store = TradingStateStore(uri, db)

        pf_data    = store.load_portfolio_history(session_id, start, end)
        order_data = store.load_orders(session_id, start, end)

        sorted_ts = sorted(pf_data["value_history"]) if pf_data["value_history"] else []
        pf_shell = types.SimpleNamespace(
            keep_history=True,
            tick_history=pf_data["tick_history"],
            value_history=pf_data["value_history"],
            cash_history=pf_data["cash_history"],
            signals_history=pf_data["signals_history"],
            total_value=pf_data["value_history"][sorted_ts[-1]] if sorted_ts else 0.0,
            cash=pf_data["cash_history"][sorted_ts[-1]]         if sorted_ts else 0.0,
            positions={},
        )
        om_shell = types.SimpleNamespace(
            filled_orders_by_id=order_data["filled_orders_by_id"],
            pending_orders_by_id=order_data["pending_orders_by_id"],
        )
        pf_shell.om = om_shell
        analyzer = cls(pf_shell)
        session_doc = store.get_session(session_id)
        if session_doc:
            analyzer._session_metadata = session_doc.get("metadata", {}) or {}
        return analyzer

    @classmethod
    def from_mongodb_multi(
        cls,
        session_ids: list[str],
        connection_uri: str = None,
        database: str = None,
        start: datetime = None,
        end: datetime = None,
    ) -> "PortfolioAnalyzer":
        """Reconstruct a PortfolioAnalyzer by merging multiple stored sessions.

        Useful when a live bot restarted and created several sessions that should
        be analyzed as one continuous run.

        Args:
            session_ids:    List of session UUIDs to merge (in any order).
            connection_uri: MongoDB connection string (overrides config).
            database:       Database name (overrides config).
            start:          Optional start datetime filter applied to all sessions.
            end:            Optional end datetime filter applied to all sessions.

        Returns:
            PortfolioAnalyzer ready for analysis.

        Example::

            analyzer = PortfolioAnalyzer.from_mongodb_multi(["sid1", "sid2", "sid3"])
            result = analyzer.run_analysis(output_dir="output/combined")
        """
        import types
        from utils.trading_state_store import TradingStateStore

        uri, db = cls._resolve_mongo_params(connection_uri, database)
        store = TradingStateStore(uri, db)

        pf_data    = store.load_portfolio_history_multi(session_ids, start, end)
        order_data = store.load_orders_multi(session_ids, start, end)

        sorted_ts = sorted(pf_data["value_history"]) if pf_data["value_history"] else []
        pf_shell = types.SimpleNamespace(
            keep_history=True,
            tick_history=pf_data["tick_history"],
            value_history=pf_data["value_history"],
            cash_history=pf_data["cash_history"],
            signals_history=pf_data["signals_history"],
            total_value=pf_data["value_history"][sorted_ts[-1]] if sorted_ts else 0.0,
            cash=pf_data["cash_history"][sorted_ts[-1]]         if sorted_ts else 0.0,
            positions={},
        )
        om_shell = types.SimpleNamespace(
            filled_orders_by_id=order_data["filled_orders_by_id"],
            pending_orders_by_id=order_data["pending_orders_by_id"],
        )
        pf_shell.om = om_shell
        analyzer = cls(pf_shell)
        first_doc = store.get_session(session_ids[0]) if session_ids else None
        if first_doc:
            analyzer._session_metadata = first_doc.get("metadata", {}) or {}
        return analyzer

    # -----------------------------------------------------------------------
    # save_all
    # -----------------------------------------------------------------------

    def save_all(self, output_dir: str) -> dict[str, bool]:
        """
        Save all 7 artifacts to output_dir (created if needed).
        Each save is independent — one failure never blocks the rest.
        Returns {filename: True/False} success dict.
        """
        os.makedirs(output_dir, exist_ok=True)
        artifact_methods = {
            'equity_curve.png':       self.save_equity_curve,
            'drawdown.png':           self.save_drawdown_chart,
            'technical_analysis.png': self.save_technical_chart,
            'orders_chart.png':       self.save_orders_chart,
            'lifecycle.png':               self.save_lifecycle_chart,
            'lifecycle_interactive.html': self.save_lifecycle_chart_interactive,
            'performance_report.txt':     self.save_performance_report,
            'signals_orders.csv':     self.save_signals_orders_csv,
            'trades.csv':             self.save_trades_csv,
            'summary.json':           self.save_summary_json,
        }
        results: dict[str, bool] = {}
        for filename, method in artifact_methods.items():
            fpath = os.path.join(output_dir, filename)
            try:
                method(fpath)
                results[filename] = True
                logger.info(f"Saved {filename}")
            except Exception as e:
                results[filename] = False
                logger.warning(f"Failed to save {filename}: {e}", exc_info=True)
        return results

    # -----------------------------------------------------------------------
    # MLflow logging
    # -----------------------------------------------------------------------

    def _mlflow_log_with_files(
        self,
        output_dir: str,
        files: dict[str, bool],
        experiment_name: str = None,
        run_name: str = None,
        parameters: dict = None,
        tags: dict = None,
    ) -> None:
        """Backward-compatible wrapper over shared MLflow reporting."""
        self.log_to_mlflow(
            experiment_name=experiment_name,
            run_name=run_name,
            parameters=parameters,
            tags=tags,
            artifact_paths=[os.path.join(output_dir, filename) for filename, ok in files.items() if ok],
        )

    # -----------------------------------------------------------------------
    # Main convenience method
    # -----------------------------------------------------------------------

    def run_analysis(
        self,
        output_dir: str = None,
        log_to_mlflow: bool = True,
        experiment_name: str = None,
        run_name: str = None,
        parameters: dict = None,
        tags: dict = None,
    ) -> dict:
        """
        Full analysis pipeline: compute metrics, save artifacts, log to MLflow.

        Args:
            output_dir:       Directory to save artifact files. Skipped if None.
            log_to_mlflow:    Log to MLflow (default True). Requires output_dir.
            experiment_name:  MLflow experiment name (falls back to config).
            run_name:         MLflow run name.
            parameters:       Extra params dict to log (e.g. algo config).
            tags:             MLflow tags dict.

        Returns:
            {"metrics": PerformanceMetrics, "trades": list[Trade], "files": dict}
        """
        trades = self.get_trades()

        metrics = None
        try:
            metrics = self.get_metrics()
        except Exception as e:
            logger.warning(f"Could not compute metrics: {e}")

        files: dict[str, bool] = {}
        if output_dir is not None:
            files = self.save_all(output_dir)

        if log_to_mlflow and output_dir is not None and metrics is not None:
            try:
                self._mlflow_log_with_files(
                    output_dir=output_dir,
                    files=files,
                    experiment_name=experiment_name,
                    run_name=run_name,
                    parameters=parameters,
                    tags=tags,
                )
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

        return {"metrics": metrics, "trades": trades, "files": files}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Flatten a nested dict to dot-notation keys."""
    items: dict = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
