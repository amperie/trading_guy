"""
Comprehensive analysis engine for backtesting results
Provides trade extraction, performance metrics, visualizations, and reports
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict

from trading.core.portfolio import Portfolio
from trading.core.om.order_manager import OrderManager
from trading.core.classes import Order, BracketOrder, OrderType, OrderAction, OrderStatus
from utils.logger import Logger

logger = Logger().get_logger(__name__)


@dataclass
class Trade:
    """Represents a complete trade (entry + exit)"""
    symbol: str
    entry_order: Order
    exit_order: Order
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    duration: float  # in seconds
    is_bracket: bool = False
    bracket_exit_type: Optional[str] = None  # "STOP", "PROFIT", "MANUAL"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Returns
    total_return: float
    total_return_pct: float
    annualized_return: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration: float  # in days

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L statistics
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float  # total wins / total losses

    # Trade analysis
    avg_trade_pnl: float
    avg_trade_duration: float  # in hours
    avg_bars_in_trade: float

    # Bracket order specific
    bracket_trades: int
    bracket_stop_triggers: int
    bracket_profit_triggers: int
    bracket_manual_exits: int
    bracket_stop_rate: float
    bracket_profit_rate: float

    # Risk-adjusted
    calmar_ratio: float  # annualized return / max drawdown
    ulcer_index: float

    # Additional metrics
    volatility: float  # annualized
    skewness: float
    kurtosis: float
    best_day: float
    worst_day: float
    avg_daily_return: float

    # Equity curve stats
    final_equity: float
    initial_equity: float
    peak_equity: float

    # Time in market
    total_days: float
    trading_days: int


class AnalysisEngine:
    """
    Comprehensive analysis engine for backtesting results

    Usage:
        engine = AnalysisEngine(portfolio, order_manager)
        trades = engine.extract_trades()
        metrics = engine.calculate_metrics()
        engine.plot_equity_curve()
        report = engine.generate_report()
    """

    def __init__(self, portfolio: Portfolio, order_manager: OrderManager):
        self.portfolio = portfolio
        self.order_manager = order_manager
        self._trades: Optional[list[Trade]] = None
        self._metrics: Optional[PerformanceMetrics] = None

    def extract_trades(self) -> list[Trade]:
        """
        Extract complete trades from filled orders
        Pairs buy and sell orders to create Trade objects
        """
        trades = []

        # Track open positions for each symbol (FIFO matching)
        open_positions: dict[str, list[Order]] = defaultdict(list)

        # Get all orders (filled and pending) sorted by executed_datetime
        # Include pending orders because bracket BUY orders stay in pending state
        all_orders_dict = {**self.order_manager.filled_orders_by_id, **self.order_manager.pending_orders_by_id}
        filled_orders = sorted(
            all_orders_dict.values(),
            key=lambda o: o.executed_datetime if o.executed_datetime else o.placed_datetime
        )

        for order in filled_orders:
            symbol = order.symbol

            if order.action == OrderAction.BUY:
                # Add to open positions (including BRACKET orders)
                open_positions[symbol].append(order)

            elif order.action == OrderAction.SELL:
                # Match with oldest buy orders (FIFO)
                remaining_qty = order.quantity

                while remaining_qty > 0 and open_positions[symbol]:
                    buy_order = open_positions[symbol][0]
                    match_qty = min(remaining_qty, buy_order.quantity)

                    # Calculate P&L for this trade
                    entry_price = buy_order.price
                    exit_price = order.price
                    pnl = (exit_price - entry_price) * match_qty - buy_order.tx_cost - order.tx_cost
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100

                    # Determine if this is a bracket order exit
                    is_bracket = False
                    bracket_exit_type = None

                    # Check if buy order is a BRACKET type
                    if buy_order.type == OrderType.BRACKET:
                        is_bracket = True
                        # Check which type of exit this was
                        if order.type == OrderType.STOP_LOSS:
                            bracket_exit_type = "STOP"
                        elif order.type == OrderType.PROFIT_TAKER:
                            bracket_exit_type = "PROFIT"
                        else:
                            bracket_exit_type = "MANUAL"

                    # Create trade
                    entry_time = buy_order.executed_datetime or buy_order.placed_datetime
                    exit_time = order.executed_datetime or order.placed_datetime

                    trade = Trade(
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
                        bracket_exit_type=bracket_exit_type
                    )
                    trades.append(trade)

                    # Update quantities
                    remaining_qty -= match_qty
                    buy_order.quantity -= match_qty

                    if buy_order.quantity == 0:
                        open_positions[symbol].pop(0)

        self._trades = trades
        return trades

    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if self._trades is None:
            self.extract_trades()

        if not self.portfolio.keep_history:
            raise ValueError("Portfolio must have keep_history=True to calculate metrics")

        # Get equity curve
        equity_curve = pd.Series(self.portfolio.value_history)
        equity_curve.index = pd.to_datetime(list(self.portfolio.tick_history.keys()), utc=True)

        initial_equity = equity_curve.iloc[0]
        final_equity = equity_curve.iloc[-1]
        peak_equity = equity_curve.max()

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        # Time metrics
        total_days = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / 86400
        trading_days = len(equity_curve)

        # Total return
        total_return = final_equity - initial_equity
        total_return_pct = (total_return / initial_equity) * 100
        annualized_return = ((final_equity / initial_equity) ** (365.25 / total_days) - 1) * 100 if total_days > 0 else 0

        # Drawdown calculation
        running_max = equity_curve.expanding().max()
        drawdown = equity_curve - running_max
        drawdown_pct = (drawdown / running_max) * 100
        max_drawdown = drawdown.min()
        max_drawdown_pct = drawdown_pct.min()

        # Max drawdown duration
        is_drawdown = drawdown < 0
        drawdown_periods = is_drawdown.astype(int).diff().fillna(0)
        drawdown_starts = equity_curve.index[drawdown_periods == 1]
        drawdown_ends = equity_curve.index[drawdown_periods == -1]

        if len(drawdown_starts) > 0 and len(drawdown_ends) > 0:
            max_dd_duration = max([(end - start).total_seconds() / 86400
                                   for start, end in zip(drawdown_starts, drawdown_ends)])
        else:
            max_dd_duration = 0

        # Risk metrics
        if len(returns) > 1:
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
            sortino_ratio = (annualized_return / downside_std) if downside_std > 0 else 0

            skewness = returns.skew()
            kurtosis = returns.kurtosis()
        else:
            volatility = 0
            sharpe_ratio = 0
            sortino_ratio = 0
            skewness = 0
            kurtosis = 0

        # Calmar ratio
        calmar_ratio = (annualized_return / abs(max_drawdown_pct)) if max_drawdown_pct != 0 else 0

        # Ulcer index
        squared_drawdown_pct = drawdown_pct ** 2
        ulcer_index = np.sqrt(squared_drawdown_pct.mean())

        # Daily stats
        daily_returns = returns
        best_day = daily_returns.max() * 100 if len(daily_returns) > 0 else 0
        worst_day = daily_returns.min() * 100 if len(daily_returns) > 0 else 0
        avg_daily_return = daily_returns.mean() * 100 if len(daily_returns) > 0 else 0

        # Trade statistics
        total_trades = len(self._trades)
        winning_trades = len([t for t in self._trades if t.pnl > 0])
        losing_trades = len([t for t in self._trades if t.pnl < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # P&L statistics
        wins = [t.pnl for t in self._trades if t.pnl > 0]
        losses = [t.pnl for t in self._trades if t.pnl < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0

        # Trade analysis
        avg_trade_pnl = np.mean([t.pnl for t in self._trades]) if self._trades else 0
        avg_trade_duration = np.mean([t.duration / 3600 for t in self._trades]) if self._trades else 0  # hours

        # Calculate avg bars in trade
        avg_bars_in_trade = 0
        if self._trades and self.portfolio.tick_history:
            tick_times = sorted(self.portfolio.tick_history.keys())
            bars_per_trade = []
            for trade in self._trades:
                bars_in_trade = sum(1 for t in tick_times if trade.entry_time <= t <= trade.exit_time)
                bars_per_trade.append(bars_in_trade)
            avg_bars_in_trade = np.mean(bars_per_trade) if bars_per_trade else 0

        # Bracket order statistics
        bracket_trades_list = [t for t in self._trades if t.is_bracket]
        bracket_trades = len(bracket_trades_list)
        bracket_stop_triggers = len([t for t in bracket_trades_list if t.bracket_exit_type == "STOP"])
        bracket_profit_triggers = len([t for t in bracket_trades_list if t.bracket_exit_type == "PROFIT"])
        bracket_manual_exits = len([t for t in bracket_trades_list if t.bracket_exit_type == "MANUAL"])

        bracket_stop_rate = (bracket_stop_triggers / bracket_trades * 100) if bracket_trades > 0 else 0
        bracket_profit_rate = (bracket_profit_triggers / bracket_trades * 100) if bracket_trades > 0 else 0

        metrics = PerformanceMetrics(
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
            trading_days=trading_days
        )

        self._metrics = metrics
        return metrics

    def get_tick_returns(self) -> pd.Series:
        """
        Calculate returns for each tick in the data
        Returns a pandas Series indexed by datetime with tick-by-tick returns
        """
        if not self.portfolio.tick_history or not self.portfolio.value_history:
            raise ValueError("Portfolio must have keep_history=True to calculate tick returns")

        # Create equity curve from portfolio value history
        equity_curve = pd.Series(self.portfolio.value_history)
        equity_curve.index = pd.to_datetime(list(self.portfolio.tick_history.keys()), utc=True)

        # Calculate tick-by-tick returns
        tick_returns = equity_curve.pct_change().dropna()

        return tick_returns

    def get_daily_returns(self) -> pd.Series:
        """
        Calculate daily returns by resampling tick returns.
        Returns a pandas Series indexed by date with daily returns
        """
        tick_returns = self.get_tick_returns()

        # Resample to daily frequency
        # Use the last value of each day and calculate daily return
        if not self.portfolio.value_history:
            raise ValueError("Portfolio must have keep_history=True")

        equity_curve = pd.Series(self.portfolio.value_history)
        equity_curve.index = pd.to_datetime(list(self.portfolio.tick_history.keys()), utc=True)

        # Resample to daily, taking the last value of each day
        daily_equity = equity_curve.resample('D').last().dropna()

        # Calculate daily returns
        daily_returns = daily_equity.pct_change().dropna()

        return daily_returns

    def get_monthly_returns(self) -> pd.Series:
        """
        Calculate monthly returns by resampling tick returns
        Returns a pandas Series indexed by month with monthly returns
        """
        if not self.portfolio.value_history:
            raise ValueError("Portfolio must have keep_history=True")

        equity_curve = pd.Series(self.portfolio.value_history)
        equity_curve.index = pd.to_datetime(list(self.portfolio.tick_history.keys()), utc=True)

        # Resample to monthly, taking the last value of each month
        monthly_equity = equity_curve.resample('ME').last().dropna()

        # Calculate monthly returns
        monthly_returns = monthly_equity.pct_change().dropna()

        return monthly_returns

    def analyze_bracket_effectiveness(self) -> dict:
        """
        Analyze effectiveness of bracket orders
        Returns dict with detailed bracket order statistics
        """
        if self._trades is None:
            self.extract_trades()

        bracket_trades = [t for t in self._trades if t.is_bracket]

        if not bracket_trades:
            return {
                "total_bracket_trades": 0,
                "message": "No bracket trades found"
            }

        # Separate by exit type
        stop_trades = [t for t in bracket_trades if t.bracket_exit_type == "STOP"]
        profit_trades = [t for t in bracket_trades if t.bracket_exit_type == "PROFIT"]
        manual_trades = [t for t in bracket_trades if t.bracket_exit_type == "MANUAL"]

        # Calculate stats for each type
        def calc_stats(trades):
            if not trades:
                return {}
            return {
                "count": len(trades),
                "avg_pnl": np.mean([t.pnl for t in trades]),
                "total_pnl": sum([t.pnl for t in trades]),
                "avg_pnl_pct": np.mean([t.pnl_pct for t in trades]),
                "avg_duration_hours": np.mean([t.duration / 3600 for t in trades]),
                "win_rate": len([t for t in trades if t.pnl > 0]) / len(trades) * 100
            }

        return {
            "total_bracket_trades": len(bracket_trades),
            "stop_loss_exits": calc_stats(stop_trades),
            "profit_taker_exits": calc_stats(profit_trades),
            "manual_exits": calc_stats(manual_trades),
            "overall": {
                "avg_pnl": np.mean([t.pnl for t in bracket_trades]),
                "total_pnl": sum([t.pnl for t in bracket_trades]),
                "win_rate": len([t for t in bracket_trades if t.pnl > 0]) / len(bracket_trades) * 100,
                "stop_rate": len(stop_trades) / len(bracket_trades) * 100,
                "profit_rate": len(profit_trades) / len(bracket_trades) * 100,
                "manual_rate": len(manual_trades) / len(bracket_trades) * 100
            }
        }

    def plot_equity_curve(self, show: bool = False, save_path: Optional[str] = None):
        """Plot equity curve over time"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for thread safety
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        if not self.portfolio.tick_history:
            raise ValueError("Portfolio must have keep_history=True")

        equity_curve = pd.Series(self.portfolio.value_history)
        equity_curve.index = pd.to_datetime(list(self.portfolio.tick_history.keys()), utc=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(equity_curve.index, equity_curve.values, linewidth=2)
        ax.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_drawdown(self, show: bool = False, save_path: Optional[str] = None):
        """Plot drawdown over time"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for thread safety
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if not self.portfolio.tick_history:
            raise ValueError("Portfolio must have keep_history=True")

        equity_curve = pd.Series(self.portfolio.value_history)
        equity_curve.index = pd.to_datetime(list(self.portfolio.tick_history.keys()), utc=True)

        running_max = equity_curve.expanding().max()
        drawdown_pct = ((equity_curve - running_max) / running_max) * 100

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(drawdown_pct.index, drawdown_pct.values, 0, alpha=0.3, color='red')
        ax.plot(drawdown_pct.index, drawdown_pct.values, color='red', linewidth=2)
        ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_trade_pnl(self, show: bool = False, save_path: Optional[str] = None):
        """Plot P&L for each trade"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for thread safety
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if self._trades is None:
            self.extract_trades()

        if not self._trades:
            raise ValueError("No trades to plot")

        pnls = [t.pnl for t in self._trades]
        colors = ['green' if p > 0 else 'red' for p in pnls]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.6)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('P&L ($)')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_returns_distribution(self, show: bool = False, save_path: Optional[str] = None):
        """Plot distribution of returns"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for thread safety
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        tick_returns = self.get_tick_returns()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(tick_returns * 100, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_title('Returns Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_portfolio_with_trades(self, show: bool = False, save_path: Optional[str] = None):
        """Plot portfolio value over time with BUY/SELL markers"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for thread safety
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if not self.portfolio.tick_history or not self.portfolio.value_history:
            raise ValueError("Portfolio must have keep_history=True")

        # Get equity curve
        equity_curve = pd.Series(self.portfolio.value_history)
        equity_curve.index = pd.to_datetime(list(self.portfolio.tick_history.keys()), utc=True)

        # Get all filled orders (exclude CANCELED and child orders of brackets)
        buy_orders = []
        sell_orders = []

        # Collect all orders (both filled and pending, since bracket orders stay pending)
        all_orders = {**self.order_manager.filled_orders_by_id, **self.order_manager.pending_orders_by_id}

        # First pass: identify all bracket parent IDs
        bracket_parent_ids = set()
        for order in all_orders.values():
            if order.type == OrderType.BRACKET and order.action == OrderAction.BUY:
                bracket_parent_ids.add(order.order_id)

        # Second pass: categorize orders
        for order in all_orders.values():
            # Skip orders with zero or negative quantity
            if order.quantity <= 0:
                continue

            # Get order timestamp - use executed_datetime if available, otherwise placed_datetime
            order_time = order.executed_datetime if order.executed_datetime else order.placed_datetime

            # Handle BRACKET orders (these are the BUY entries)
            # Note: Bracket orders have status PENDING or PENDING_SALE after execution, not FILLED
            # When a bracket order is executed, it goes from PENDING → PENDING_SALE
            if order.type == OrderType.BRACKET and order.action == OrderAction.BUY:
                # Show bracket orders that have been executed (status is PENDING_SALE or FILLED)
                # Also show if status is PENDING but has executed_datetime set
                if order.status in {OrderStatus.PENDING_SALE, OrderStatus.FILLED} or order.executed_datetime:
                    buy_orders.append((order_time, order))
                continue

            # Only show FILLED orders for non-bracket orders
            if order.status != OrderStatus.FILLED:
                continue

            # Handle regular orders (but skip child orders of bracket orders)
            if order.type != OrderType.BRACKET:
                # Skip if this is a child order of a bracket order
                if hasattr(order, 'parent_id') and order.parent_id in bracket_parent_ids:
                    # Only add the actual exit order (skip the canceled sibling)
                    if order.action == OrderAction.SELL:
                        sell_orders.append((order_time, order))
                # Regular non-bracket orders
                elif not hasattr(order, 'parent_id'):
                    if order.action == OrderAction.BUY:
                        buy_orders.append((order_time, order))
                    elif order.action == OrderAction.SELL:
                        sell_orders.append((order_time, order))

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot equity curve
        ax.plot(equity_curve.index, equity_curve.values, linewidth=2, color='blue', label='Portfolio Value')

        # Plot BUY markers
        if buy_orders:
            buy_times = [t for t, _ in buy_orders]
            # Find portfolio value at buy times (use nearest value)
            buy_values = [equity_curve.asof(t) for t in buy_times]
            ax.scatter(buy_times, buy_values, marker='x', s=150, c='green',
                      linewidths=3, label='BUY', zorder=5)

        # Plot SELL markers
        if sell_orders:
            sell_times = [t for t, _ in sell_orders]
            # Find portfolio value at sell times (use nearest value)
            sell_values = [equity_curve.asof(t) for t in sell_times]
            ax.scatter(sell_times, sell_values, marker='x', s=150, c='red',
                      linewidths=3, label='SELL', zorder=5)

        ax.set_title('Portfolio Value with Trade Markers', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_stock_performance(self, show: bool = False, save_path: Optional[str] = None):
        """
        Plot stock prices and returns over time for each position
        Shows how individual stocks performed during the backtest period
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for thread safety
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if not self.portfolio.tick_history:
            raise ValueError("Portfolio must have keep_history=True")

        # Extract price data for each symbol from tick_history
        symbol_prices = defaultdict(list)
        symbol_times = defaultdict(list)

        for timestamp, tick_data in self.portfolio.tick_history.items():
            for price_data in tick_data:
                symbol = price_data.symbol
                symbol_times[symbol].append(timestamp)
                symbol_prices[symbol].append(price_data.close)

        if not symbol_prices:
            raise ValueError("No price data found in tick history")

        # Create subplots: one for prices, one for returns
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Stock prices over time
        for symbol in sorted(symbol_prices.keys()):
            times = symbol_times[symbol]
            prices = symbol_prices[symbol]
            ax1.plot(times, prices, linewidth=2, label=symbol, marker='o', markersize=3)

        ax1.set_title('Stock Prices Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))

        # Plot 2: Stock returns (normalized to 100 at start)
        for symbol in sorted(symbol_prices.keys()):
            prices = symbol_prices[symbol]
            times = symbol_times[symbol]

            if len(prices) > 1:
                # Normalize to 100 at the start
                initial_price = prices[0]
                normalized_values = [(p / initial_price) * 100 for p in prices]
                ax2.plot(times, normalized_values, linewidth=2, label=symbol, marker='o', markersize=3)

                # Calculate total return
                total_return = ((prices[-1] - prices[0]) / prices[0]) * 100
                print(f"{symbol} Total Return: {total_return:.2f}%")

        ax2.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Initial Value')
        ax2.set_title('Stock Returns (Normalized to 100)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Normalized Value')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_interactive_portfolio(self, show: bool = False, save_path: Optional[str] = None):
        """
        Create interactive Plotly chart with portfolio value, cash, trades, and stock prices

        Features:
        - Zoomable timeline
        - Portfolio value line
        - Cash balance line
        - Buy/Sell markers (green/red X)
        - Individual stock price lines
        - Hide/unhide lines via legend clicks
        - Hover tooltips with detailed info

        Args:
            show: Whether to display the chart in browser
            save_path: Path to save HTML file (e.g., "portfolio.html")

        Returns:
            Plotly figure object
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("plotly is required for interactive charts. Install with: pip install plotly")

        if not self.portfolio.tick_history or not self.portfolio.value_history:
            raise ValueError("Portfolio must have keep_history=True")

        # Create figure with secondary y-axis for better scaling
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Extract portfolio value history
        timestamps = list(self.portfolio.tick_history.keys())
        portfolio_values = [self.portfolio.value_history[ts] for ts in timestamps]
        cash_values = [self.portfolio.cash_history[ts] for ts in timestamps]

        # Add portfolio value line
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=portfolio_values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2.5),
                hovertemplate='<b>Portfolio Value</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ),
            secondary_y=False
        )

        # Add cash balance line
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=cash_values,
                mode='lines',
                name='Cash Balance',
                line=dict(color='green', width=2, dash='dot'),
                hovertemplate='<b>Cash Balance</b><br>Date: %{x}<br>Cash: $%{y:,.2f}<extra></extra>'
            ),
            secondary_y=False
        )

        # Extract stock prices from tick history
        symbol_prices = defaultdict(list)
        symbol_times = defaultdict(list)

        for timestamp, tick_data in self.portfolio.tick_history.items():
            for price_data in tick_data:
                symbol = price_data.symbol
                symbol_times[symbol].append(timestamp)
                symbol_prices[symbol].append(price_data.close)

        # Add stock price lines (on secondary y-axis for better scaling)
        colors = ['purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'lime']
        for i, symbol in enumerate(sorted(symbol_prices.keys())):
            times = symbol_times[symbol]
            prices = symbol_prices[symbol]
            color = colors[i % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=prices,
                    mode='lines',
                    name=f'{symbol} Price',
                    line=dict(color=color, width=1.5, dash='dash'),
                    visible='legendonly',  # Hidden by default to avoid clutter
                    hovertemplate=f'<b>{symbol} Price</b><br>Date: %{{x}}<br>Price: $%{{y:,.2f}}<extra></extra>'
                ),
                secondary_y=True
            )

        # Get buy and sell orders for markers
        buy_orders = []
        sell_orders = []

        # Collect all orders (both filled and pending, since bracket orders stay pending)
        all_orders = {**self.order_manager.filled_orders_by_id, **self.order_manager.pending_orders_by_id}

        # First pass: identify all bracket parent IDs
        bracket_parent_ids = set()
        for order in all_orders.values():
            if order.type == OrderType.BRACKET and order.action == OrderAction.BUY:
                bracket_parent_ids.add(order.order_id)

        # Second pass: categorize orders
        for order in all_orders.values():
            # Skip orders with zero or negative quantity
            if order.quantity <= 0:
                continue

            # Get order timestamp - use executed_datetime if available, otherwise placed_datetime
            order_time = order.executed_datetime if order.executed_datetime else order.placed_datetime

            # Handle BRACKET orders (BUY entries)
            # Note: Bracket orders have status PENDING or PENDING_SALE after execution, not FILLED
            # When a bracket order is executed, it goes from PENDING → PENDING_SALE
            if order.type == OrderType.BRACKET and order.action == OrderAction.BUY:
                # Show bracket orders that have been executed (status is PENDING_SALE or FILLED)
                # Also show if status is PENDING but has executed_datetime set
                if order.status in {OrderStatus.PENDING_SALE, OrderStatus.FILLED} or order.executed_datetime:
                    buy_orders.append((order_time, order))
                continue

            # Only show FILLED orders for non-bracket orders
            if order.status != OrderStatus.FILLED:
                continue

            # Handle regular orders
            if order.type != OrderType.BRACKET:
                # Skip if this is a child order of a bracket order
                if hasattr(order, 'parent_id') and order.parent_id in bracket_parent_ids:
                    if order.action == OrderAction.SELL:
                        sell_orders.append((order_time, order))
                # Regular non-bracket orders
                elif not hasattr(order, 'parent_id'):
                    if order.action == OrderAction.BUY:
                        buy_orders.append((order_time, order))
                    elif order.action == OrderAction.SELL:
                        sell_orders.append((order_time, order))

        # Add BUY markers
        if buy_orders:
            buy_times = [t for t, _ in buy_orders]
            buy_prices = [o.price for _, o in buy_orders]
            buy_symbols = [o.symbol for _, o in buy_orders]
            buy_quantities = [o.quantity for _, o in buy_orders]

            # Get portfolio value at buy times for marker placement
            buy_portfolio_values = []
            for t in buy_times:
                # Find nearest timestamp
                idx = min(range(len(timestamps)), key=lambda i: abs((timestamps[i] - t).total_seconds()))
                buy_portfolio_values.append(portfolio_values[idx])

            hover_text = [
                f'<b>BUY</b><br>Symbol: {sym}<br>Quantity: {qty}<br>Price: ${price:.2f}<br>Total: ${qty*price:,.2f}'
                for sym, qty, price in zip(buy_symbols, buy_quantities, buy_prices)
            ]

            fig.add_trace(
                go.Scatter(
                    x=buy_times,
                    y=buy_portfolio_values,
                    mode='markers',
                    name='BUY',
                    marker=dict(
                        symbol='x',
                        size=15,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_text
                ),
                secondary_y=False
            )

        # Add SELL markers
        if sell_orders:
            sell_times = [t for t, _ in sell_orders]
            sell_prices = [o.price for _, o in sell_orders]
            sell_symbols = [o.symbol for _, o in sell_orders]
            sell_quantities = [o.quantity for _, o in sell_orders]

            # Get portfolio value at sell times for marker placement
            sell_portfolio_values = []
            for t in sell_times:
                # Find nearest timestamp
                idx = min(range(len(timestamps)), key=lambda i: abs((timestamps[i] - t).total_seconds()))
                sell_portfolio_values.append(portfolio_values[idx])

            hover_text = [
                f'<b>SELL</b><br>Symbol: {sym}<br>Quantity: {qty}<br>Price: ${price:.2f}<br>Total: ${qty*price:,.2f}'
                for sym, qty, price in zip(sell_symbols, sell_quantities, sell_prices)
            ]

            fig.add_trace(
                go.Scatter(
                    x=sell_times,
                    y=sell_portfolio_values,
                    mode='markers',
                    name='SELL',
                    marker=dict(
                        symbol='x',
                        size=15,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_text
                ),
                secondary_y=False
            )

        # Update layout for interactivity
        fig.update_layout(
            title={
                'text': 'Interactive Portfolio Analysis',
                'font': {'size': 20, 'color': 'darkblue'}
            },
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            height=700,
            template='plotly_white',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1
            ),
            xaxis=dict(
                rangeslider=dict(visible=True),  # Add range slider for zooming
                type='date'
            )
        )

        # Update y-axes titles
        fig.update_yaxes(title_text="Portfolio Value / Cash ($)", secondary_y=False)
        fig.update_yaxes(title_text="Stock Price ($)", secondary_y=True)

        # Add annotation about interactivity
        fig.add_annotation(
            text="💡 Click legend items to hide/show • Use range slider to zoom • Hover for details",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=11, color="gray"),
            xanchor='center'
        )

        # Save to HTML if requested
        if save_path:
            if not save_path.endswith('.html'):
                save_path += '.html'
            fig.write_html(save_path)
            logger.debug(f"Interactive chart saved to: {save_path}")

        # Show in browser if requested
        if show:
            fig.show()

        return fig

    def plot_comprehensive_dashboard(self, show: bool = False, save_path: Optional[str] = None):
        """Create a comprehensive dashboard with multiple plots"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for thread safety
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if self._metrics is None:
            self.calculate_metrics()

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        equity_curve = pd.Series(self.portfolio.value_history)
        equity_curve.index = pd.to_datetime(list(self.portfolio.tick_history.keys()), utc=True)
        ax1.plot(equity_curve.index, equity_curve.values, linewidth=2, color='blue')
        ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        running_max = equity_curve.expanding().max()
        drawdown_pct = ((equity_curve - running_max) / running_max) * 100
        ax2.fill_between(drawdown_pct.index, drawdown_pct.values, 0, alpha=0.3, color='red')
        ax2.plot(drawdown_pct.index, drawdown_pct.values, color='red', linewidth=2)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # 3. Trade P&L
        ax3 = fig.add_subplot(gs[2, 0])
        if self._trades:
            pnls = [t.pnl for t in self._trades]
            colors = ['green' if p > 0 else 'red' for p in pnls]
            ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.6)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('Trade P&L', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('P&L ($)')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Performance Metrics Text
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.axis('off')

        metrics_text = f"""
        PERFORMANCE METRICS

        Total Return: ${self._metrics.total_return:,.2f} ({self._metrics.total_return_pct:.2f}%)
        Annualized Return: {self._metrics.annualized_return:.2f}%
        Sharpe Ratio: {self._metrics.sharpe_ratio:.2f}
        Sortino Ratio: {self._metrics.sortino_ratio:.2f}
        Max Drawdown: {self._metrics.max_drawdown_pct:.2f}%

        Total Trades: {self._metrics.total_trades}
        Win Rate: {self._metrics.win_rate:.1f}%
        Profit Factor: {self._metrics.profit_factor:.2f}
        Avg Trade P&L: ${self._metrics.avg_trade_pnl:,.2f}

        Bracket Trades: {self._metrics.bracket_trades}
        Stop Loss Rate: {self._metrics.bracket_stop_rate:.1f}%
        Profit Taker Rate: {self._metrics.bracket_profit_rate:.1f}%
        """

        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.suptitle('Trading Performance Dashboard', fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def calculate_benchmark_comparison(self) -> Dict[str, Any]:
        """
        Calculate buy-and-hold benchmark metrics for comparison

        Returns:
            Dict with benchmark metrics for each symbol and comparative analysis
        """
        if not self.portfolio.tick_history:
            raise ValueError("Portfolio must have keep_history=True")

        # Extract price data for each symbol from tick_history
        symbol_prices = defaultdict(list)
        symbol_times = defaultdict(list)

        for timestamp, tick_data in self.portfolio.tick_history.items():
            for price_data in tick_data:
                symbol = price_data.symbol
                symbol_times[symbol].append(timestamp)
                symbol_prices[symbol].append(price_data.close)

        if not symbol_prices:
            return {}

        # Get portfolio returns for comparison
        portfolio_returns = self.get_tick_returns()
        initial_equity = self._metrics.initial_equity if self._metrics else self.portfolio.value_history[0]

        benchmark_results = {}

        for symbol in symbol_prices:
            prices = np.array(symbol_prices[symbol])

            # Calculate buy-and-hold returns
            initial_price = prices[0]
            final_price = prices[-1]

            # Buy-and-hold metrics
            total_return_pct = ((final_price - initial_price) / initial_price) * 100

            # Calculate hypothetical shares if all capital was invested in this symbol
            shares = initial_equity / initial_price
            final_value = shares * final_price
            total_return_dollars = final_value - initial_equity

            # Calculate returns series for this symbol
            symbol_returns = pd.Series(prices).pct_change().fillna(0)

            # Risk metrics for buy-and-hold
            volatility = symbol_returns.std() * np.sqrt(252 * 78) * 100  # Annualized (5min bars)

            # Calculate max drawdown
            cumulative = (1 + symbol_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown_pct = drawdown.min() * 100

            # Sharpe ratio (assuming 0% risk-free rate for simplicity)
            if volatility > 0:
                sharpe = (symbol_returns.mean() * 252 * 78) / (symbol_returns.std() * np.sqrt(252 * 78))
            else:
                sharpe = 0

            benchmark_results[symbol] = {
                'total_return_pct': total_return_pct,
                'total_return_dollars': total_return_dollars,
                'final_value': final_value,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown_pct': max_drawdown_pct,
                'initial_price': initial_price,
                'final_price': final_price,
                'price_change_pct': total_return_pct
            }

        # Calculate comparative metrics vs portfolio
        if self._metrics and benchmark_results:
            # Use equal-weighted benchmark if multiple symbols
            avg_benchmark_return = np.mean([b['total_return_pct'] for b in benchmark_results.values()])
            avg_benchmark_sharpe = np.mean([b['sharpe_ratio'] for b in benchmark_results.values()])
            avg_benchmark_vol = np.mean([b['volatility'] for b in benchmark_results.values()])

            comparison = {
                'portfolio_return_pct': self._metrics.total_return_pct,
                'benchmark_return_pct': avg_benchmark_return,
                'alpha': self._metrics.total_return_pct - avg_benchmark_return,
                'portfolio_sharpe': self._metrics.sharpe_ratio,
                'benchmark_sharpe': avg_benchmark_sharpe,
                'portfolio_volatility': self._metrics.volatility,
                'benchmark_volatility': avg_benchmark_vol,
                'outperformance': self._metrics.total_return_pct > avg_benchmark_return
            }

            benchmark_results['_comparison'] = comparison

        return benchmark_results

    def generate_report(self) -> str:
        """Generate a comprehensive text report"""
        if self._metrics is None:
            self.calculate_metrics()

        if self._trades is None:
            self.extract_trades()

        bracket_analysis = self.analyze_bracket_effectiveness()

        report = f"""
{'='*80}
TRADING PERFORMANCE REPORT
{'='*80}

OVERVIEW
--------
Initial Equity:         ${self._metrics.initial_equity:,.2f}
Final Equity:           ${self._metrics.final_equity:,.2f}
Peak Equity:            ${self._metrics.peak_equity:,.2f}
Total Return:           ${self._metrics.total_return:,.2f} ({self._metrics.total_return_pct:.2f}%)
Annualized Return:      {self._metrics.annualized_return:.2f}%

RISK METRICS
------------
Volatility (Ann.):      {self._metrics.volatility:.2f}%
Sharpe Ratio:           {self._metrics.sharpe_ratio:.2f}
Sortino Ratio:          {self._metrics.sortino_ratio:.2f}
Calmar Ratio:           {self._metrics.calmar_ratio:.2f}
Max Drawdown:           ${self._metrics.max_drawdown:,.2f} ({self._metrics.max_drawdown_pct:.2f}%)
Max DD Duration:        {self._metrics.max_drawdown_duration:.1f} days
Ulcer Index:            {self._metrics.ulcer_index:.2f}

TRADE STATISTICS
----------------
Total Trades:           {self._metrics.total_trades}
Winning Trades:         {self._metrics.winning_trades}
Losing Trades:          {self._metrics.losing_trades}
Win Rate:               {self._metrics.win_rate:.2f}%

Average Win:            ${self._metrics.avg_win:,.2f}
Average Loss:           ${self._metrics.avg_loss:,.2f}
Largest Win:            ${self._metrics.largest_win:,.2f}
Largest Loss:           ${self._metrics.largest_loss:,.2f}
Profit Factor:          {self._metrics.profit_factor:.2f}

Avg Trade P&L:          ${self._metrics.avg_trade_pnl:,.2f}
Avg Trade Duration:     {self._metrics.avg_trade_duration:.2f} hours
Avg Bars in Trade:      {self._metrics.avg_bars_in_trade:.1f}

DAILY STATISTICS
----------------
Best Day:               {self._metrics.best_day:.2f}%
Worst Day:              {self._metrics.worst_day:.2f}%
Avg Daily Return:       {self._metrics.avg_daily_return:.4f}%

DISTRIBUTION
------------
Skewness:               {self._metrics.skewness:.2f}
Kurtosis:               {self._metrics.kurtosis:.2f}

BRACKET ORDER ANALYSIS
----------------------
Total Bracket Trades:   {self._metrics.bracket_trades}
Stop Loss Triggers:     {self._metrics.bracket_stop_triggers} ({self._metrics.bracket_stop_rate:.1f}%)
Profit Taker Triggers:  {self._metrics.bracket_profit_triggers} ({self._metrics.bracket_profit_rate:.1f}%)
Manual Exits:           {self._metrics.bracket_manual_exits}
"""

        if bracket_analysis.get("total_bracket_trades", 0) > 0:
            report += f"""
Bracket Performance:
  - Stop Loss Exits:    Avg P&L ${bracket_analysis['stop_loss_exits'].get('avg_pnl', 0):,.2f}, Win Rate {bracket_analysis['stop_loss_exits'].get('win_rate', 0):.1f}%
  - Profit Taker Exits: Avg P&L ${bracket_analysis['profit_taker_exits'].get('avg_pnl', 0):,.2f}, Win Rate {bracket_analysis['profit_taker_exits'].get('win_rate', 0):.1f}%
  - Manual Exits:       Avg P&L ${bracket_analysis['manual_exits'].get('avg_pnl', 0):,.2f}, Win Rate {bracket_analysis['manual_exits'].get('win_rate', 0):.1f}%
"""

        # Add benchmark comparison section
        try:
            benchmark = self.calculate_benchmark_comparison()
            if benchmark:
                report += f"""
BENCHMARK COMPARISON (BUY & HOLD)
----------------------------------
"""
                # Per-symbol benchmarks
                for symbol, metrics in benchmark.items():
                    if symbol == '_comparison':
                        continue
                    report += f"""
{symbol}:
  Initial Price:        ${metrics['initial_price']:,.2f}
  Final Price:          ${metrics['final_price']:,.2f}
  Price Change:         {metrics['price_change_pct']:.2f}%
  B&H Total Return:     ${metrics['total_return_dollars']:,.2f} ({metrics['total_return_pct']:.2f}%)
  B&H Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}
  B&H Volatility:       {metrics['volatility']:.2f}%
  B&H Max Drawdown:     {metrics['max_drawdown_pct']:.2f}%
"""

                # Overall comparison
                if '_comparison' in benchmark:
                    comp = benchmark['_comparison']
                    outperform_symbol = "[+]" if comp['outperformance'] else "[-]"
                    report += f"""
STRATEGY vs. BUY & HOLD COMPARISON
-----------------------------------
Strategy Return:        {comp['portfolio_return_pct']:.2f}%
Buy & Hold Return:      {comp['benchmark_return_pct']:.2f}%
Alpha (Excess Return):  {comp['alpha']:.2f}% {outperform_symbol}

Strategy Sharpe:        {comp['portfolio_sharpe']:.2f}
Buy & Hold Sharpe:      {comp['benchmark_sharpe']:.2f}

Strategy Volatility:    {comp['portfolio_volatility']:.2f}%
Buy & Hold Volatility:  {comp['benchmark_volatility']:.2f}%

Performance:            {"OUTPERFORMED" if comp['outperformance'] else "UNDERPERFORMED"} buy-and-hold by {abs(comp['alpha']):.2f}%
"""
        except Exception as e:
            logger.debug(f"Could not calculate benchmark comparison: {e}")

        report += f"""
TIME PERIOD
-----------
Total Days:             {self._metrics.total_days:.1f}
Trading Days:           {self._metrics.trading_days}

{'='*80}
"""

        return report

    def generate_signals_orders_report(self) -> str:
        """
        Generate a detailed report linking signals to their corresponding orders.

        For each signal that has an 'order_id' in its metadata, shows:
        - Signal details (type, symbol, strength, timestamp, full metadata)
        - Entry order details (price, quantity, cost)
        - Exit order details (price, quantity, proceeds, P&L) if applicable

        Returns:
            Formatted text report as string
        """
        if not hasattr(self.portfolio, 'signals_history') or not self.portfolio.signals_history:
            return "No signals history available.\n"

        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("SIGNALS TO ORDERS REPORT")
        report_lines.append("=" * 100)
        report_lines.append("")

        # Sort signals by timestamp
        sorted_timestamps = sorted(self.portfolio.signals_history.keys())

        signal_count = 0
        matched_count = 0

        for timestamp in sorted_timestamps:
            signals = self.portfolio.signals_history[timestamp]

            for signal in signals:
                signal_count += 1

                # Check if signal has an order_id in metadata
                if 'order_id' not in signal.metadata:
                    continue

                order_id = signal.metadata['order_id']

                # Find the order in order_manager
                order = self.order_manager.all_orders.get(order_id)

                if order is None:
                    logger.warning(f"Order {order_id} referenced in signal not found in order manager")
                    continue

                matched_count += 1

                # Generate entry for this signal-order pair
                report_lines.append("-" * 100)
                report_lines.append(f"SIGNAL #{matched_count}")
                report_lines.append("-" * 100)

                # Signal details
                report_lines.append(f"Timestamp:          {timestamp}")
                report_lines.append(f"Signal Type:        {signal.type.name}")
                report_lines.append(f"Symbol:             {signal.symbol}")
                report_lines.append(f"Strength:           {signal.strength}/100")

                # Metadata (excluding order_id which we already show)
                if signal.metadata:
                    report_lines.append(f"Metadata:")
                    for key, value in signal.metadata.items():
                        if key != 'order_id':
                            # Format datetime objects nicely
                            if isinstance(value, datetime):
                                value = value.strftime("%Y-%m-%d %H:%M:%S")
                            # Format floats to 4 decimal places
                            elif isinstance(value, float):
                                value = f"{value:.4f}"
                            report_lines.append(f"  {key:20s} = {value}")

                report_lines.append("")

                # Entry Order details
                report_lines.append(f"ENTRY ORDER:")
                report_lines.append(f"  Order ID:         {order.order_id}")
                report_lines.append(f"  Order Type:       {order.type.name}")
                report_lines.append(f"  Action:           {order.action.name}")
                report_lines.append(f"  Status:           {order.status.name}")
                report_lines.append(f"  Quantity:         {order.quantity:,}")
                report_lines.append(f"  Entry Price:      ${order.price:,.4f}")
                report_lines.append(f"  Total Cost:       ${order.cash:,.2f}")
                report_lines.append(f"  Transaction Cost: ${order.tx_cost:,.2f}")
                report_lines.append(f"  Placed:           {order.placed_datetime}")
                if order.executed_datetime:
                    report_lines.append(f"  Executed:         {order.executed_datetime}")

                # Check if this is a BracketOrder with a SOLD_ORDER (exit)
                if isinstance(order, BracketOrder) and hasattr(order, 'SOLD_ORDER') and order.SOLD_ORDER is not None:
                    sold_order = order.SOLD_ORDER
                    report_lines.append("")
                    report_lines.append(f"EXIT ORDER:")
                    report_lines.append(f"  Order ID:         {sold_order.order_id}")
                    report_lines.append(f"  Exit Type:        {sold_order.type.name}")
                    report_lines.append(f"  Action:           {sold_order.action.name}")
                    report_lines.append(f"  Status:           {sold_order.status.name}")
                    report_lines.append(f"  Quantity:         {sold_order.quantity:,}")
                    report_lines.append(f"  Exit Price:       ${sold_order.price:,.4f}")
                    report_lines.append(f"  Total Proceeds:   ${sold_order.cash:,.2f}")
                    report_lines.append(f"  Transaction Cost: ${sold_order.tx_cost:,.2f}")
                    if sold_order.executed_datetime:
                        report_lines.append(f"  Executed:         {sold_order.executed_datetime}")

                    # Calculate P&L
                    entry_total = order.cash + order.tx_cost
                    exit_total = sold_order.cash - sold_order.tx_cost
                    pnl = exit_total - entry_total
                    pnl_pct = (pnl / entry_total * 100) if entry_total > 0 else 0

                    # Calculate duration
                    if order.executed_datetime and sold_order.executed_datetime:
                        duration = sold_order.executed_datetime - order.executed_datetime
                        duration_hours = duration.total_seconds() / 3600
                        duration_str = f"{duration_hours:.2f} hours"
                    else:
                        duration_str = "N/A"

                    report_lines.append("")
                    report_lines.append(f"TRADE SUMMARY:")
                    report_lines.append(f"  Entry Total:      ${entry_total:,.2f}")
                    report_lines.append(f"  Exit Total:       ${exit_total:,.2f}")
                    report_lines.append(f"  Net P&L:          ${pnl:,.2f} ({pnl_pct:+.2f}%)")
                    report_lines.append(f"  Duration:         {duration_str}")

                    # Show which exit condition triggered
                    if sold_order.type == OrderType.STOP_LOSS:
                        report_lines.append(f"  Exit Reason:      STOP LOSS triggered")
                    elif sold_order.type == OrderType.PROFIT_TAKER:
                        report_lines.append(f"  Exit Reason:      PROFIT TARGET hit")
                    else:
                        report_lines.append(f"  Exit Reason:      Manual exit")

                report_lines.append("")

        # Summary
        report_lines.append("=" * 100)
        report_lines.append(f"SUMMARY: {matched_count} signals linked to orders (out of {signal_count} total signals)")
        report_lines.append("=" * 100)
        report_lines.append("")

        return "\n".join(report_lines)

    def _is_object_to_explode(self, value: Any) -> bool:
        """
        Check if a metadata value should be exploded into separate columns.

        Returns True for objects with attributes (dataclasses, custom classes, etc.)
        Returns False for primitive types (str, int, float, bool, datetime, None, lists, dicts)

        Args:
            value: The value to check

        Returns:
            True if value should be exploded, False otherwise
        """
        # Primitive types - don't explode
        if value is None:
            return False
        if isinstance(value, (str, int, float, bool, bytes)):
            return False
        if isinstance(value, datetime):
            return False
        # Collections - don't explode (they'll be stored as is or converted to string)
        if isinstance(value, (list, tuple, dict, set)):
            return False

        # Check if it's an object with __dict__ (class instance)
        if hasattr(value, '__dict__'):
            return True

        # Check if it's a dataclass
        try:
            from dataclasses import is_dataclass
            if is_dataclass(value):
                return True
        except ImportError:
            pass

        return False

    def _get_object_attributes(self, obj: Any) -> Dict[str, Any]:
        """
        Extract attributes from an object for DataFrame explosion.

        Handles both regular class instances and dataclasses.
        Skips private attributes (starting with _).

        Args:
            obj: The object to extract attributes from

        Returns:
            Dictionary mapping attribute names to values

        Example:
            >>> rsi = RSI(period=14, rsi=51.37, avg_gain=0.5, avg_loss=0.3)
            >>> attrs = _get_object_attributes(rsi)
            >>> # {'period': 14, 'rsi': 51.37, 'avg_gain': 0.5, 'avg_loss': 0.3}
        """
        attributes = {}

        # Try dataclass first (if available)
        try:
            from dataclasses import is_dataclass, fields
            if is_dataclass(obj):
                for field in fields(obj):
                    if not field.name.startswith('_'):
                        attributes[field.name] = getattr(obj, field.name)
                return attributes
        except ImportError:
            pass

        # Fall back to __dict__ for regular class instances
        if hasattr(obj, '__dict__'):
            for attr_name, attr_value in obj.__dict__.items():
                # Skip private attributes
                if not attr_name.startswith('_'):
                    attributes[attr_name] = attr_value

        return attributes

    def generate_signals_orders_dataframe(self) -> pd.DataFrame:
        """
        Generate a DataFrame linking signals to their corresponding orders for analysis.

        This is useful for analyzing which signal characteristics lead to winning vs losing trades.
        All metadata fields are exploded into separate columns for easy filtering and analysis.

        **Object Explosion:**
        If a metadata value is an object (e.g., RSI, MACD indicator), its attributes are
        automatically exploded into separate columns with dot notation:
            - metadata['rsi'] = RSI(period=14, rsi=51.37, ...) becomes:
              - signal_rsi.period = 14
              - signal_rsi.rsi = 51.37
              - signal_rsi.avg_gain = ...
              - signal_rsi.avg_loss = ...

        Returns:
            pandas DataFrame with columns:
                - timestamp: When signal occurred
                - signal_type: BUY or SELL
                - symbol: Trading symbol
                - signal_strength: 0-100
                - signal_[field]: Primitive metadata values
                - signal_[object].[attribute]: Exploded object attributes (e.g., signal_rsi.period)
                - order_id: Order ID
                - order_type: MARKET, BRACKET, etc.
                - entry_price: Entry price
                - entry_quantity: Number of shares
                - entry_cost: Total entry cost
                - entry_tx_cost: Entry transaction cost
                - entry_total: entry_cost + entry_tx_cost
                - entry_datetime: When entry was executed
                - exit_price: Exit price (if applicable)
                - exit_quantity: Exit quantity (if applicable)
                - exit_proceeds: Total exit proceeds (if applicable)
                - exit_tx_cost: Exit transaction cost (if applicable)
                - exit_total: exit_proceeds - exit_tx_cost (if applicable)
                - exit_datetime: When exit was executed (if applicable)
                - exit_type: STOP_LOSS, PROFIT_TAKER, MANUAL, etc. (if applicable)
                - exit_reason: Human-readable exit reason (if applicable)
                - net_pnl: Profit/loss in dollars
                - net_pnl_pct: Profit/loss percentage
                - duration_seconds: Trade duration in seconds
                - duration_hours: Trade duration in hours
                - duration_days: Trade duration in days
                - is_win: Boolean - True if profitable trade
                - is_bracket_order: Boolean - True if bracket order
                - order_status: Current order status
                - has_exit: Boolean - True if order has been exited

        Example:
            engine = AnalysisEngine(portfolio, order_manager)
            df = engine.generate_signals_orders_dataframe()

            # Analyze winning vs losing trades
            wins = df[df['is_win'] == True]
            losses = df[df['is_win'] == False]
            print(f"Win rate: {len(wins) / len(df) * 100:.1f}%")

            # Find best signal characteristics
            print(wins.groupby('signal_type')['net_pnl'].mean())
        """
        if not hasattr(self.portfolio, 'signals_history') or not self.portfolio.signals_history:
            logger.warning("No signals history available for DataFrame generation")
            return pd.DataFrame()

        records = []

        # Sort signals by timestamp
        sorted_timestamps = sorted(self.portfolio.signals_history.keys())

        for timestamp in sorted_timestamps:
            signals = self.portfolio.signals_history[timestamp]

            for signal in signals:
                # Check if signal has an order_id in metadata
                if 'order_id' not in signal.metadata:
                    continue

                order_id = signal.metadata['order_id']

                # Find the order in order_manager
                order = self.order_manager.all_orders.get(order_id)

                if order is None:
                    logger.warning(f"Order {order_id} referenced in signal not found")
                    continue

                # Base record with signal data
                record = {
                    'timestamp': timestamp,
                    'signal_type': signal.type.name,
                    'symbol': signal.symbol,
                    'signal_strength': signal.strength,
                }

                # Explode metadata fields as separate columns (excluding order_id)
                for key, value in signal.metadata.items():
                    if key != 'order_id':
                        # Check if value is an object with attributes (not a primitive type)
                        if self._is_object_to_explode(value):
                            # Explode object attributes into separate columns
                            for attr_name, attr_value in self._get_object_attributes(value).items():
                                # Use dot notation: signal_rsi.period, signal_rsi.rsi, etc.
                                record[f'signal_{key}.{attr_name}'] = attr_value
                        else:
                            # Primitive value - store directly
                            record[f'signal_{key}'] = value

                # Order identification
                record['order_id'] = order.order_id
                record['order_type'] = order.type.name
                record['order_status'] = order.status.name
                record['is_bracket_order'] = isinstance(order, BracketOrder)

                # Entry order details
                record['entry_price'] = order.price
                record['entry_quantity'] = order.quantity
                record['entry_cost'] = order.cash
                record['entry_tx_cost'] = order.tx_cost
                record['entry_total'] = order.cash + order.tx_cost
                record['entry_datetime'] = order.executed_datetime if order.executed_datetime else order.placed_datetime

                # Exit order details (if applicable)
                has_exit = isinstance(order, BracketOrder) and hasattr(order, 'SOLD_ORDER') and order.SOLD_ORDER is not None
                record['has_exit'] = has_exit

                if has_exit:
                    sold_order = order.SOLD_ORDER
                    record['exit_price'] = sold_order.price
                    record['exit_quantity'] = sold_order.quantity
                    record['exit_proceeds'] = sold_order.cash
                    record['exit_tx_cost'] = sold_order.tx_cost
                    record['exit_total'] = sold_order.cash - sold_order.tx_cost
                    record['exit_datetime'] = sold_order.executed_datetime
                    record['exit_type'] = sold_order.type.name

                    # Human-readable exit reason
                    if sold_order.type == OrderType.STOP_LOSS:
                        record['exit_reason'] = 'STOP_LOSS'
                    elif sold_order.type == OrderType.PROFIT_TAKER:
                        record['exit_reason'] = 'PROFIT_TARGET'
                    else:
                        record['exit_reason'] = 'MANUAL'

                    # Calculate P&L metrics
                    entry_total = order.cash + order.tx_cost
                    exit_total = sold_order.cash - sold_order.tx_cost
                    net_pnl = exit_total - entry_total
                    net_pnl_pct = (net_pnl / entry_total * 100) if entry_total > 0 else 0

                    record['net_pnl'] = net_pnl
                    record['net_pnl_pct'] = net_pnl_pct
                    record['is_win'] = net_pnl > 0

                    # Calculate duration
                    if order.executed_datetime and sold_order.executed_datetime:
                        duration = sold_order.executed_datetime - order.executed_datetime
                        record['duration_seconds'] = duration.total_seconds()
                        record['duration_hours'] = duration.total_seconds() / 3600
                        record['duration_days'] = duration.total_seconds() / 86400

                        # Extract time-based features for analysis
                        record['entry_hour'] = order.executed_datetime.hour
                        record['entry_day_of_week'] = order.executed_datetime.strftime('%A')
                        record['exit_hour'] = sold_order.executed_datetime.hour
                        record['exit_day_of_week'] = sold_order.executed_datetime.strftime('%A')
                    else:
                        record['duration_seconds'] = None
                        record['duration_hours'] = None
                        record['duration_days'] = None
                        record['entry_hour'] = None
                        record['entry_day_of_week'] = None
                        record['exit_hour'] = None
                        record['exit_day_of_week'] = None

                    # Calculate return metrics for comparison
                    record['price_change_pct'] = ((sold_order.price - order.price) / order.price * 100) if order.price > 0 else 0
                    record['total_return'] = exit_total

                    # Risk metrics
                    if isinstance(order, BracketOrder):
                        stop_order = order.get_child_order("STOP")
                        profit_order = order.get_child_order("PROFIT")

                        if stop_order and profit_order:
                            record['stop_loss_price'] = stop_order.price
                            record['profit_target_price'] = profit_order.price
                            record['risk_pct'] = ((order.price - stop_order.price) / order.price * 100) if order.price > 0 else 0
                            record['reward_pct'] = ((profit_order.price - order.price) / order.price * 100) if order.price > 0 else 0
                            record['risk_reward_ratio'] = record['reward_pct'] / record['risk_pct'] if record['risk_pct'] != 0 else 0
                        else:
                            record['stop_loss_price'] = None
                            record['profit_target_price'] = None
                            record['risk_pct'] = None
                            record['reward_pct'] = None
                            record['risk_reward_ratio'] = None
                    else:
                        record['stop_loss_price'] = None
                        record['profit_target_price'] = None
                        record['risk_pct'] = None
                        record['reward_pct'] = None
                        record['risk_reward_ratio'] = None

                else:
                    # No exit yet - populate with None/NaN
                    record['exit_price'] = None
                    record['exit_quantity'] = None
                    record['exit_proceeds'] = None
                    record['exit_tx_cost'] = None
                    record['exit_total'] = None
                    record['exit_datetime'] = None
                    record['exit_type'] = None
                    record['exit_reason'] = None
                    record['net_pnl'] = None
                    record['net_pnl_pct'] = None
                    record['is_win'] = None
                    record['duration_seconds'] = None
                    record['duration_hours'] = None
                    record['duration_days'] = None
                    record['entry_hour'] = None
                    record['entry_day_of_week'] = None
                    record['exit_hour'] = None
                    record['exit_day_of_week'] = None
                    record['price_change_pct'] = None
                    record['total_return'] = None
                    record['stop_loss_price'] = None
                    record['profit_target_price'] = None
                    record['risk_pct'] = None
                    record['reward_pct'] = None
                    record['risk_reward_ratio'] = None

                records.append(record)

        df = pd.DataFrame(records)

        if not df.empty:
            logger.info(f"Generated signals-orders DataFrame with {len(df)} rows and {len(df.columns)} columns")
        else:
            logger.warning("No signals with order_id found in metadata")

        return df

    def extract_all_orders(self) -> Dict[str, list[Order]]:
        """
        Extract ALL orders from order_manager including filled, pending, cancelled,
        and child orders from bracket orders.

        Returns:
            Dictionary with keys:
                - 'all_orders': All orders (flat list)
                - 'filled_orders': Orders with FILLED status
                - 'pending_orders': Orders with PENDING status
                - 'cancelled_orders': Orders with CANCELED status
                - 'bracket_parents': Parent orders from bracket orders
                - 'bracket_children': Child orders from bracket orders (STOP, PROFIT, MANUAL)
        """
        all_orders = []
        filled_orders = []
        pending_orders = []
        cancelled_orders = []
        bracket_parents = []
        bracket_children = []

        # Iterate through all orders in order_manager
        for order_id, order in self.order_manager.all_orders.items():
            all_orders.append(order)

            # Categorize by status
            if order.status == OrderStatus.FILLED:
                filled_orders.append(order)
            elif order.status == OrderStatus.PENDING or order.status == OrderStatus.PENDING_SALE:
                pending_orders.append(order)
            elif order.status == OrderStatus.CANCELED:
                cancelled_orders.append(order)

            # Check if it's a bracket order parent
            if isinstance(order, BracketOrder):
                bracket_parents.append(order)

                # Extract child orders
                if hasattr(order, 'child_orders') and order.child_orders:
                    for child_name in order.child_orders:
                        child_order = order.get_child_order(child_name)
                        if child_order:
                            bracket_children.append(child_order)
                            all_orders.append(child_order)

                            # Categorize child by status
                            if child_order.status == OrderStatus.FILLED:
                                filled_orders.append(child_order)
                            elif child_order.status == OrderStatus.PENDING or child_order.status == OrderStatus.PENDING_SALE:
                                pending_orders.append(child_order)
                            elif child_order.status == OrderStatus.CANCELED:
                                cancelled_orders.append(child_order)

        return {
            'all_orders': all_orders,
            'filled_orders': filled_orders,
            'pending_orders': pending_orders,
            'cancelled_orders': cancelled_orders,
            'bracket_parents': bracket_parents,
            'bracket_children': bracket_children,
        }

    def generate_all_orders_report(self) -> str:
        """
        Generate comprehensive text report showing ALL orders executed,
        including filled, pending, and cancelled orders.

        Organizes bracket orders by parent-child relationships and standalone orders separately.

        Returns:
            Formatted text report string
        """
        orders_dict = self.extract_all_orders()

        all_orders = orders_dict['all_orders']
        filled_orders = orders_dict['filled_orders']
        pending_orders = orders_dict['pending_orders']
        cancelled_orders = orders_dict['cancelled_orders']
        bracket_parents = orders_dict['bracket_parents']
        bracket_children = orders_dict['bracket_children']

        lines = []
        lines.append("=" * 100)
        lines.append("COMPREHENSIVE ORDERS REPORT - ALL ORDERS")
        lines.append("=" * 100)
        lines.append("")

        # Summary statistics
        lines.append("SUMMARY")
        lines.append("-" * 100)
        lines.append(f"Total Orders:           {len(all_orders)}")
        lines.append(f"  Filled:               {len(filled_orders)}")
        lines.append(f"  Pending:              {len(pending_orders)}")
        lines.append(f"  Cancelled:            {len(cancelled_orders)}")
        lines.append(f"Bracket Parents:        {len(bracket_parents)}")
        lines.append(f"Bracket Children:       {len(bracket_children)}")
        lines.append(f"Standalone Orders:      {len(all_orders) - len(bracket_parents) - len(bracket_children)}")
        lines.append("")

        # Bracket Orders Section
        if bracket_parents:
            lines.append("=" * 100)
            lines.append("BRACKET ORDERS (Parent + Children)")
            lines.append("=" * 100)
            lines.append("")

            for parent in bracket_parents:
                lines.append("-" * 100)
                lines.append(f"PARENT ORDER: {parent.order_id}")
                lines.append(f"  Symbol:           {parent.symbol}")
                lines.append(f"  Type:             {parent.type.name}")
                lines.append(f"  Status:           {parent.status.name}")
                lines.append(f"  Action:           {parent.action.name}")
                lines.append(f"  Quantity:         {parent.quantity}")
                lines.append(f"  Price:            ${parent.price:.2f}")
                lines.append(f"  Cash:             ${parent.cash:.2f}")
                lines.append(f"  Tx Cost:          ${parent.tx_cost:.2f}")
                lines.append(f"  Placed:           {parent.placed_datetime}")
                if parent.executed_datetime:
                    lines.append(f"  Executed:         {parent.executed_datetime}")

                # Child orders
                if hasattr(parent, 'child_orders') and parent.child_orders:
                    lines.append("")
                    lines.append("  CHILD ORDERS:")
                    for child_name in parent.child_orders:
                        child_order = parent.get_child_order(child_name)
                        if child_order:
                            lines.append(f"    {child_name}:")
                            lines.append(f"      Order ID:       {child_order.order_id}")
                            lines.append(f"      Type:           {child_order.type.name}")
                            lines.append(f"      Status:         {child_order.status.name}")
                            lines.append(f"      Action:         {child_order.action.name}")
                            lines.append(f"      Quantity:       {child_order.quantity}")
                            lines.append(f"      Price:          ${child_order.price:.2f}")
                            lines.append(f"      Placed:         {child_order.placed_datetime}")
                            if child_order.executed_datetime:
                                lines.append(f"      Executed:       {child_order.executed_datetime}")

                # Show which child was sold
                if hasattr(parent, 'SOLD_ORDER') and parent.SOLD_ORDER:
                    lines.append("")
                    lines.append(f"  SOLD VIA:         {parent.SOLD_ORDER.type.name}")

                lines.append("")

        # Standalone Orders Section
        standalone_orders = [o for o in all_orders
                             if not isinstance(o, BracketOrder)
                             and not hasattr(o, 'parent_id')]

        if standalone_orders:
            lines.append("=" * 100)
            lines.append("STANDALONE ORDERS")
            lines.append("=" * 100)
            lines.append("")

            for order in standalone_orders:
                lines.append("-" * 100)
                lines.append(f"ORDER: {order.order_id}")
                lines.append(f"  Symbol:           {order.symbol}")
                lines.append(f"  Type:             {order.type.name}")
                lines.append(f"  Status:           {order.status.name}")
                lines.append(f"  Action:           {order.action.name}")
                lines.append(f"  Quantity:         {order.quantity}")
                lines.append(f"  Price:            ${order.price:.2f}")
                lines.append(f"  Cash:             ${order.cash:.2f}")
                lines.append(f"  Tx Cost:          ${order.tx_cost:.2f}")
                lines.append(f"  Placed:           {order.placed_datetime}")
                if order.executed_datetime:
                    lines.append(f"  Executed:         {order.executed_datetime}")
                lines.append("")

        lines.append("=" * 100)
        lines.append("END OF COMPREHENSIVE ORDERS REPORT")
        lines.append("=" * 100)

        return "\n".join(lines)

    def plot_orders_timeline(self, show: bool = False, save_path: Optional[str] = None):
        """
        Create timeline visualization showing ALL orders (filled, pending, cancelled)
        plotted at their execution prices.

        Args:
            show: Whether to display the plot
            save_path: Path to save the plot (optional)

        Returns:
            matplotlib figure object
        """
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for thread safety
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        orders_dict = self.extract_all_orders()
        all_orders = orders_dict['all_orders']

        if not all_orders:
            logger.warning("No orders to plot")
            return None

        fig, ax = plt.subplots(figsize=(16, 8))

        # Prepare data - categorize by action and status
        buy_filled = {'x': [], 'y': [], 'labels': [], 'symbols': []}
        sell_filled = {'x': [], 'y': [], 'labels': [], 'symbols': []}
        buy_pending = {'x': [], 'y': [], 'labels': [], 'symbols': []}
        sell_pending = {'x': [], 'y': [], 'labels': [], 'symbols': []}
        buy_cancelled = {'x': [], 'y': [], 'labels': [], 'symbols': []}
        sell_cancelled = {'x': [], 'y': [], 'labels': [], 'symbols': []}

        for order in all_orders:
            # Use executed datetime if available, else placed datetime
            order_time = order.executed_datetime if order.executed_datetime else order.placed_datetime
            # Y-axis is now the execution price
            price = order.price

            label = f"{order.symbol} {order.action.name} {order.quantity}@${order.price:.2f} ({order.status.name})"

            # Categorize by status and action
            if order.status == OrderStatus.FILLED:
                if order.action == OrderAction.BUY:
                    buy_filled['x'].append(order_time)
                    buy_filled['y'].append(price)
                    buy_filled['labels'].append(label)
                    buy_filled['symbols'].append(order.symbol)
                else:
                    sell_filled['x'].append(order_time)
                    sell_filled['y'].append(price)
                    sell_filled['labels'].append(label)
                    sell_filled['symbols'].append(order.symbol)
            elif order.status in [OrderStatus.PENDING, OrderStatus.PENDING_SALE]:
                if order.action == OrderAction.BUY:
                    buy_pending['x'].append(order_time)
                    buy_pending['y'].append(price)
                    buy_pending['labels'].append(label)
                    buy_pending['symbols'].append(order.symbol)
                else:
                    sell_pending['x'].append(order_time)
                    sell_pending['y'].append(price)
                    sell_pending['labels'].append(label)
                    sell_pending['symbols'].append(order.symbol)
            elif order.status == OrderStatus.CANCELED:
                if order.action == OrderAction.BUY:
                    buy_cancelled['x'].append(order_time)
                    buy_cancelled['y'].append(price)
                    buy_cancelled['labels'].append(label)
                    buy_cancelled['symbols'].append(order.symbol)
                else:
                    sell_cancelled['x'].append(order_time)
                    sell_cancelled['y'].append(price)
                    sell_cancelled['labels'].append(label)
                    sell_cancelled['symbols'].append(order.symbol)

        # Plot each category with distinct colors
        if buy_filled['x']:
            ax.scatter(buy_filled['x'], buy_filled['y'], c='green', marker='^',
                       s=120, alpha=0.8, label='BUY - Filled', edgecolors='black', linewidths=1.5)

        if sell_filled['x']:
            ax.scatter(sell_filled['x'], sell_filled['y'], c='red', marker='v',
                       s=120, alpha=0.8, label='SELL - Filled', edgecolors='black', linewidths=1.5)

        if buy_pending['x']:
            ax.scatter(buy_pending['x'], buy_pending['y'], c='blue', marker='^',
                       s=120, alpha=0.7, label='BUY - Pending', edgecolors='black', linewidths=1.5)

        if sell_pending['x']:
            ax.scatter(sell_pending['x'], sell_pending['y'], c='orange', marker='v',
                       s=120, alpha=0.7, label='SELL - Pending', edgecolors='black', linewidths=1.5)

        if buy_cancelled['x']:
            ax.scatter(buy_cancelled['x'], buy_cancelled['y'], c='lightgray', marker='^',
                       s=100, alpha=0.5, label='BUY - Cancelled', edgecolors='black', linewidths=1)

        if sell_cancelled['x']:
            ax.scatter(sell_cancelled['x'], sell_cancelled['y'], c='darkgray', marker='v',
                       s=100, alpha=0.5, label='SELL - Cancelled', edgecolors='black', linewidths=1)

        # Formatting
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
        ax.set_title('Orders Timeline - All Orders by Execution Price', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.2f}'))

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Orders timeline saved to {save_path}")

        if show:
            plt.show()

        return fig

    def log_to_mlflow(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        log_charts: bool = True,
        log_trades: bool = True,
        log_signals: bool = True,
        log_report: bool = True,
        chart_dpi: int = 150
    ):
        """
        Log all analysis results to MLflow

        Args:
            experiment_name: Name for the MLflow experiment
            run_name: Name for the MLflow run
            description: Description of the run
            tags: Additional tags to attach to the run
            parameters: Strategy/algorithm parameters to log
            log_charts: Whether to log chart visualizations
            log_trades: Whether to log individual trades as JSON
            log_signals: Whether to log signals history as JSON (default: True)
            log_report: Whether to log text report
            chart_dpi: DPI for chart images

        Example:
            engine = AnalysisEngine(portfolio, order_manager)
            engine.log_to_mlflow(
                run_name="SMA Strategy Test",
                description="Testing 5/20 SMA crossover",
                parameters={"sma_short": 5, "sma_long": 20, "symbol": "AAPL"}
            )
        """
        try:
            from utils.mlflow_client import MLflowClient
        except ImportError:
            logger.error("MLflow not available. Install with: pip install mlflow")
            return

        # Initialize MLflow client with experiment name
        mlflow = MLflowClient.from_config(experiment_name=experiment_name)

        if not mlflow.enabled:
            logger.info("MLflow tracking is disabled in config")
            return

        # Start run
        with mlflow.start_run(run_name=run_name, description=description, tags=tags):
            logger.info("Logging analysis results to MLflow...")

            # Ensure metrics are calculated
            if self._metrics is None:
                self.calculate_metrics()

            # Ensure trades are extracted
            if self._trades is None:
                self.extract_trades()

            # Log parameters if provided
            if parameters:
                mlflow.log_params(parameters)
                logger.debug(f"Logged {len(parameters)} parameters")

            # Log performance metrics
            logger.debug("Logging performance metrics...")
            mlflow.log_metrics({
                # Returns
                "total_return": self._metrics.total_return,
                "total_return_pct": self._metrics.total_return_pct,
                "annualized_return": self._metrics.annualized_return,

                # Risk metrics
                "sharpe_ratio": self._metrics.sharpe_ratio,
                "sortino_ratio": self._metrics.sortino_ratio,
                "max_drawdown": self._metrics.max_drawdown,
                "max_drawdown_pct": self._metrics.max_drawdown_pct,
                "max_drawdown_duration": self._metrics.max_drawdown_duration,
                "calmar_ratio": self._metrics.calmar_ratio,
                "ulcer_index": self._metrics.ulcer_index,
                "volatility": self._metrics.volatility,

                # Trade statistics
                "total_trades": float(self._metrics.total_trades),
                "winning_trades": float(self._metrics.winning_trades),
                "losing_trades": float(self._metrics.losing_trades),
                "win_rate": self._metrics.win_rate,

                # P&L statistics
                "avg_win": self._metrics.avg_win,
                "avg_loss": self._metrics.avg_loss,
                "largest_win": self._metrics.largest_win,
                "largest_loss": self._metrics.largest_loss,
                "profit_factor": self._metrics.profit_factor,
                "avg_trade_pnl": self._metrics.avg_trade_pnl,
                "avg_trade_duration_hours": self._metrics.avg_trade_duration,
                "avg_bars_in_trade": self._metrics.avg_bars_in_trade,

                # Bracket order metrics
                "bracket_trades": float(self._metrics.bracket_trades),
                "bracket_stop_triggers": float(self._metrics.bracket_stop_triggers),
                "bracket_profit_triggers": float(self._metrics.bracket_profit_triggers),
                "bracket_manual_exits": float(self._metrics.bracket_manual_exits),
                "bracket_stop_rate": self._metrics.bracket_stop_rate,
                "bracket_profit_rate": self._metrics.bracket_profit_rate,

                # Daily statistics
                "best_day": self._metrics.best_day,
                "worst_day": self._metrics.worst_day,
                "avg_daily_return": self._metrics.avg_daily_return,

                # Distribution
                "skewness": self._metrics.skewness,
                "kurtosis": self._metrics.kurtosis,

                # Equity metrics
                "final_equity": self._metrics.final_equity,
                "initial_equity": self._metrics.initial_equity,
                "peak_equity": self._metrics.peak_equity,

                # Time metrics
                "total_days": self._metrics.total_days,
                "trading_days": float(self._metrics.trading_days),
            })

            # Log charts if enabled
            if log_charts:
                try:
                    logger.debug("Generating and logging charts...")

                    # Equity curve
                    fig = self.plot_equity_curve(show=False)
                    mlflow.log_chart(fig, "equity_curve", format="png", dpi=chart_dpi)

                    # Portfolio with trades
                    fig = self.plot_portfolio_with_trades(show=False)
                    mlflow.log_chart(fig, "portfolio_with_trades", format="png", dpi=chart_dpi)

                    # Drawdown
                    fig = self.plot_drawdown(show=False)
                    mlflow.log_chart(fig, "drawdown", format="png", dpi=chart_dpi)

                    # Trade P&L
                    if self._trades:
                        fig = self.plot_trade_pnl(show=False)
                        mlflow.log_chart(fig, "trade_pnl", format="png", dpi=chart_dpi)

                    # Returns distribution
                    fig = self.plot_returns_distribution(show=False)
                    mlflow.log_chart(fig, "returns_distribution", format="png", dpi=chart_dpi)

                    # Stock performance
                    fig = self.plot_stock_performance(show=False)
                    mlflow.log_chart(fig, "stock_performance", format="png", dpi=chart_dpi)

                    # Comprehensive dashboard
                    fig = self.plot_comprehensive_dashboard(save_path=None)
                    mlflow.log_chart(fig, "dashboard", format="png", dpi=chart_dpi*1.33)

                    # Interactive Plotly chart (saved as HTML)
                    try:
                        import plotly.graph_objects as go
                        logger.info("Generating interactive Plotly chart...")
                        fig = self.plot_interactive_portfolio(show=False)

                        # Convert to HTML string and log using MLflow's log_html
                        logger.debug("Converting chart to HTML...")
                        html_string = fig.to_html(
                            include_plotlyjs='cdn',  # Use CDN for smaller file size
                            config={'displayModeBar': True, 'responsive': True}
                        )
                        logger.debug(f"HTML generated, size: {len(html_string)} bytes")

                        logger.debug("Logging HTML to MLflow...")
                        mlflow.log_html(html_string, "interactive_portfolio")
                        logger.info("Successfully logged interactive chart: interactive_portfolio.html")
                    except ImportError as e:
                        logger.warning(f"⚠Plotly not installed, skipping interactive chart. Install with: pip install plotly")
                    except Exception as e:
                        logger.error(f"Failed to log interactive chart: {e}", exc_info=True)

                    logger.debug("Logged 7 static chart artifacts + 1 interactive HTML chart")

                except Exception as e:
                    logger.warning(f"Failed to log some charts: {e}")

            # Log trades as JSON if enabled
            if log_trades and self._trades:
                logger.debug("Logging trades data...")
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
                        "duration_seconds": t.duration,
                        "duration_hours": t.duration / 3600,
                        "is_bracket": t.is_bracket,
                        "bracket_exit_type": t.bracket_exit_type,
                    }
                    for t in self._trades
                ]
                mlflow.log_json(trades_data, "trades.json")

            # Log bracket analysis if applicable
            if self._metrics.bracket_trades > 0:
                bracket_analysis = self.analyze_bracket_effectiveness()
                mlflow.log_json(bracket_analysis, "bracket_analysis.json")

            # Log signals history as JSON if enabled
            if log_signals and hasattr(self.portfolio, 'signals_history') and self.portfolio.signals_history:
                logger.debug("Logging signals history...")
                from utils.utils import serialize_signals_history, serialize_signals_history_flat

                # Convert dict[datetime, list[MarketSignal]] to list[list[MarketSignal]]
                # Sort by timestamp to maintain chronological order
                sorted_timestamps = sorted(self.portfolio.signals_history.keys())
                signals_list = [self.portfolio.signals_history[ts] for ts in sorted_timestamps]

                # Log nested format (preserves tick structure)
                signals_nested = serialize_signals_history(signals_list)
                mlflow.log_json(signals_nested, "signals_history.json")

                # Log flat format (easier for pandas analysis)
                signals_flat = serialize_signals_history_flat(signals_list)
                mlflow.log_json(signals_flat, "signals_history_flat.json")

                # Calculate signal statistics
                total_signals = sum(len(signals) for signals in signals_list)
                buy_signals = sum(1 for signals in signals_list for s in signals if s.type.name == "BUY")
                sell_signals = sum(1 for signals in signals_list for s in signals if s.type.name == "SELL")
                avg_strength = sum(s.strength for signals in signals_list for s in signals) / total_signals if total_signals > 0 else 0

                logger.info(f"Logged {total_signals} signals across {len(signals_list)} ticks "
                           f"({buy_signals} BUY, {sell_signals} SELL, avg strength: {avg_strength:.1f})")

                # Generate and log signals-to-orders report (with error handling)
                try:
                    logger.debug("Generating signals-to-orders report...")
                    signals_orders_report = self.generate_signals_orders_report()
                    if signals_orders_report and "No signals history" not in signals_orders_report:
                        mlflow.log_text(signals_orders_report, "signals_to_orders_report.txt")
                        logger.debug("Signals-to-orders report logged successfully")
                except Exception as e:
                    logger.warning(f"Failed to generate signals-to-orders report: {e}")
                    logger.debug("Continuing without signals-to-orders report", exc_info=True)

                # Generate and log signals-to-orders DataFrame (with error handling)
                try:
                    logger.debug("Generating signals-to-orders DataFrame...")
                    signals_df = self.generate_signals_orders_dataframe()
                    if not signals_df.empty:
                        # Save DataFrame as CSV for easy analysis
                        import tempfile
                        import os

                        # Create temp directory to ensure proper filenames
                        temp_dir = tempfile.mkdtemp()

                        # Save CSV with proper filename
                        csv_path = os.path.join(temp_dir, "signals_orders_dataframe.csv")
                        signals_df.to_csv(csv_path, index=False)
                        mlflow.log_artifact(csv_path)
                        logger.debug(f"Signals-to-orders DataFrame CSV logged ({len(signals_df)} rows, {len(signals_df.columns)} columns)")

                        # Also save as Parquet for efficient storage (if pyarrow available)
                        try:
                            parquet_path = os.path.join(temp_dir, "signals_orders_dataframe.parquet")
                            signals_df.to_parquet(parquet_path, index=False)
                            mlflow.log_artifact(parquet_path)
                            logger.debug("Signals-to-orders DataFrame Parquet logged")
                        except ImportError:
                            logger.debug("Parquet export skipped (pyarrow not available)")
                        except Exception as e:
                            logger.debug(f"Parquet export failed: {e}")

                        # Clean up temp directory
                        try:
                            if os.path.exists(csv_path):
                                os.unlink(csv_path)
                            # Check if parquet file was created
                            parquet_path = os.path.join(temp_dir, "signals_orders_dataframe.parquet")
                            if os.path.exists(parquet_path):
                                os.unlink(parquet_path)
                            os.rmdir(temp_dir)
                        except:
                            pass  # Best effort cleanup

                        # Log basic statistics about the DataFrame
                        if 'is_win' in signals_df.columns:
                            wins = signals_df['is_win'].sum()
                            total = signals_df['is_win'].notna().sum()
                            win_rate = (wins / total * 100) if total > 0 else 0
                            logger.info(f"Signals-orders analysis: {wins}/{total} wins ({win_rate:.1f}% win rate)")
                    else:
                        logger.debug("No signal-order correlations found")
                except Exception as e:
                    logger.warning(f"Failed to generate signals-to-orders DataFrame: {e}")
                    logger.debug("Continuing without signals-to-orders DataFrame", exc_info=True)

            # Log text report if enabled
            if log_report:
                logger.debug("Logging text report...")
                report = self.generate_report()
                mlflow.log_text(report, "performance_report.txt")

            # Log comprehensive orders report (all orders including cancelled)
            try:
                logger.debug("Generating comprehensive orders report...")
                all_orders_report = self.generate_all_orders_report()
                if all_orders_report:
                    mlflow.log_text(all_orders_report, "all_orders_report.txt")
                    logger.debug("Comprehensive orders report logged successfully")
            except Exception as e:
                logger.warning(f"Failed to generate comprehensive orders report: {e}")
                logger.debug("Continuing without comprehensive orders report", exc_info=True)

            # Log orders timeline chart if charts are enabled
            if log_charts:
                try:
                    logger.debug("Generating orders timeline chart...")
                    fig = self.plot_orders_timeline(show=False)
                    if fig:
                        mlflow.log_chart(fig, "orders_timeline", format="png", dpi=chart_dpi)
                        logger.debug("Orders timeline chart logged successfully")
                except Exception as e:
                    logger.warning(f"Failed to generate orders timeline chart: {e}")
                    logger.debug("Continuing without orders timeline chart", exc_info=True)

            # Log summary as markdown
            logger.debug("Logging markdown summary...")
            markdown_summary = f"""# Backtest Analysis Summary

## Performance Overview
- **Total Return:** ${self._metrics.total_return:,.2f} ({self._metrics.total_return_pct:.2f}%)
- **Annualized Return:** {self._metrics.annualized_return:.2f}%
- **Initial Equity:** ${self._metrics.initial_equity:,.2f}
- **Final Equity:** ${self._metrics.final_equity:,.2f}
- **Peak Equity:** ${self._metrics.peak_equity:,.2f}

## Risk Metrics
- **Sharpe Ratio:** {self._metrics.sharpe_ratio:.2f}
- **Sortino Ratio:** {self._metrics.sortino_ratio:.2f}
- **Max Drawdown:** {self._metrics.max_drawdown_pct:.2f}%
- **Volatility (Ann.):** {self._metrics.volatility:.2f}%
- **Calmar Ratio:** {self._metrics.calmar_ratio:.2f}

## Trade Statistics
- **Total Trades:** {self._metrics.total_trades}
- **Winning Trades:** {self._metrics.winning_trades}
- **Losing Trades:** {self._metrics.losing_trades}
- **Win Rate:** {self._metrics.win_rate:.2f}%
- **Profit Factor:** {self._metrics.profit_factor:.2f}

## P&L Statistics
- **Average Win:** ${self._metrics.avg_win:,.2f}
- **Average Loss:** ${self._metrics.avg_loss:,.2f}
- **Largest Win:** ${self._metrics.largest_win:,.2f}
- **Largest Loss:** ${self._metrics.largest_loss:,.2f}
- **Avg Trade P&L:** ${self._metrics.avg_trade_pnl:,.2f}

## Bracket Orders
- **Bracket Trades:** {self._metrics.bracket_trades}
- **Stop Loss Triggers:** {self._metrics.bracket_stop_triggers} ({self._metrics.bracket_stop_rate:.1f}%)
- **Profit Taker Triggers:** {self._metrics.bracket_profit_triggers} ({self._metrics.bracket_profit_rate:.1f}%)
- **Manual Exits:** {self._metrics.bracket_manual_exits}

## Time Period
- **Total Days:** {self._metrics.total_days:.1f}
- **Trading Days:** {self._metrics.trading_days}
"""
            mlflow.log_markdown(markdown_summary, "summary.md")

            # Get MLflow UI URL
            run_url = mlflow.get_run_url()
            logger.info(f"Analysis logged to MLflow successfully!")
            if run_url:
                logger.info(f"View results: {run_url}")

    def run_full_analysis(
        self,
        log_to_mlflow: bool = True,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        save_charts_locally: bool = False,
        save_report_locally: bool = False,
        output_dir: str = ".",
        show_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete analysis and optionally log to MLflow

        This method performs all analysis steps from analysis_example.py:
        - Extract trades
        - Calculate metrics
        - Get returns (tick, daily, monthly)
        - Analyze bracket effectiveness
        - Generate report
        - Generate all visualizations
        - Optionally log everything to MLflow

        Args:
            log_to_mlflow: Whether to log results to MLflow (default: True)
            experiment_name: Name for MLflow experiment
            run_name: Name for MLflow run
            description: Description for MLflow run
            tags: Tags for MLflow run
            parameters: Parameters to log to MLflow
            save_charts_locally: Whether to save charts to local files
            save_report_locally: Whether to save report to local file
            output_dir: Directory for local file outputs
            show_summary: Whether to print summary to console

        Returns:
            Dictionary containing all analysis results:
                - trades: List of Trade objects
                - metrics: PerformanceMetrics object
                - tick_returns: pandas Series
                - daily_returns: pandas Series
                - monthly_returns: pandas Series
                - bracket_analysis: dict
                - report: str

        Example:
            engine = AnalysisEngine(portfolio, order_manager)
            results = engine.run_full_analysis(
                run_name="SMA Strategy",
                parameters={"sma_short": 5, "sma_long": 20},
                save_charts_locally=True
            )
        """
        logger.info("Starting full analysis...")

        # Extract trades
        logger.debug("Extracting trades...")
        trades = self.extract_trades()
        logger.info(f"Extracted {len(trades)} trades")

        # Calculate metrics
        logger.debug("Calculating performance metrics...")
        metrics = self.calculate_metrics()
        logger.info("Metrics calculated")

        # Get returns at different time scales
        logger.debug("Calculating returns...")
        tick_returns = self.get_tick_returns()
        daily_returns = self.get_daily_returns()
        monthly_returns = self.get_monthly_returns()

        # Analyze bracket effectiveness
        bracket_analysis = None
        if metrics.bracket_trades > 0:
            logger.debug("Analyzing bracket order effectiveness...")
            bracket_analysis = self.analyze_bracket_effectiveness()

        # Generate report
        logger.debug("Generating text report...")
        report = self.generate_report()

        # Print summary if requested
        if show_summary:
            print("\n" + "="*80)
            print("ANALYSIS COMPLETE")
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
            print("="*80 + "\n")

        # Save report locally if requested
        if save_report_locally:
            import os
            report_path = os.path.join(output_dir, "backtest_report.txt")
            with open(report_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to: {report_path}")

        # Save charts locally if requested
        if save_charts_locally:
            try:
                import os
                logger.info("Generating and saving charts...")

                self.plot_equity_curve(show=False, save_path=os.path.join(output_dir, "equity_curve.png"))
                self.plot_portfolio_with_trades(show=False, save_path=os.path.join(output_dir, "portfolio_with_trades.png"))
                self.plot_drawdown(show=False, save_path=os.path.join(output_dir, "drawdown.png"))
                self.plot_trade_pnl(show=False, save_path=os.path.join(output_dir, "trade_pnl.png"))
                self.plot_returns_distribution(show=False, save_path=os.path.join(output_dir, "returns_distribution.png"))
                self.plot_stock_performance(show=False, save_path=os.path.join(output_dir, "stock_performance.png"))
                self.plot_comprehensive_dashboard(save_path=os.path.join(output_dir, "dashboard.png"))

                # Save interactive Plotly chart as HTML
                try:
                    self.plot_interactive_portfolio(show=False, save_path=os.path.join(output_dir, "interactive_portfolio.html"))
                    logger.info(f"Interactive chart saved to: {output_dir}/interactive_portfolio.html")
                except ImportError:
                    logger.debug("Plotly not available, skipping interactive chart")
                except Exception as e:
                    logger.warning(f"Failed to save interactive chart: {e}")

                logger.info(f"Charts saved to: {output_dir}/")
            except Exception as e:
                logger.warning(f"Failed to save some charts: {e}")

        # Log to MLflow if requested
        if log_to_mlflow:
            logger.info("Logging results to MLflow...")
            if experiment_name is None:
                self.log_to_mlflow(
                    run_name=run_name,
                    description=description,
                    tags=tags,
                    parameters=parameters
                )
            else:
                self.log_to_mlflow(
                    experiment_name=experiment_name,
                    run_name=run_name,
                    description=description,
                    tags=tags,
                    parameters=parameters
                )

        # Return all results
        return {
            "trades": trades,
            "metrics": metrics,
            "tick_returns": tick_returns,
            "daily_returns": daily_returns,
            "monthly_returns": monthly_returns,
            "bracket_analysis": bracket_analysis,
            "report": report,
        }
