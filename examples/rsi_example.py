"""
RSI (Relative Strength Index) Algorithm Example

This example demonstrates how to use the RSI indicator in a trading algorithm.
RSI is a momentum oscillator that measures the speed and magnitude of price
changes, oscillating between 0 and 100.

Typical interpretation:
- RSI > 70: Overbought (potential sell signal)
- RSI < 30: Oversold (potential buy signal)
- RSI = 50: Neutral (no clear momentum)

NOTE: There are two ways to use RSI:
1. TechnicalAnalyzer.calculate_rsi() - Returns RSI dataclass with additional info
2. calculate_rsi() from utils.indicators - Returns just the RSI value (wrapper)
"""

from trading.core.algorithm import Algorithm
from trading.core.classes import MarketSignal, PriceData, SignalType
from trading.core.ta import TechnicalAnalyzer  # Recommended: Use TechnicalAnalyzer directly
from utils.indicators import calculate_rsi, calculate_rsi_series  # Backward compatibility wrappers


class RSIAlgorithm(Algorithm):
    """
    Simple RSI-based trading algorithm.

    Strategy:
    - BUY when RSI crosses below 30 (oversold)
    - SELL when RSI crosses above 70 (overbought)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.prev_rsi = {}  # Store previous RSI values by symbol
        self.rsi_period = config.get('rsi_period', 14)
        self.oversold_threshold = config.get('oversold', 30)
        self.overbought_threshold = config.get('overbought', 70)

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        signals = []

        for pd in data:
            # Access price history (automatically maintained by Algorithm base class)
            prices = list(self.price_history[pd.symbol])

            # Need at least period + 1 prices to calculate RSI
            if len(prices) < self.rsi_period + 1:
                continue

            # Calculate current RSI
            current_rsi = calculate_rsi(prices, period=self.rsi_period)

            if current_rsi is None:
                continue

            # Check for crossover signals (need previous RSI)
            if pd.symbol in self.prev_rsi:
                prev_rsi = self.prev_rsi[pd.symbol]

                # Buy signal: RSI crosses below oversold threshold
                if prev_rsi >= self.oversold_threshold and current_rsi < self.oversold_threshold:
                    signals.append(MarketSignal(
                        type=SignalType.BUY,
                        symbol=pd.symbol,
                        strength=int((self.oversold_threshold - current_rsi) / self.oversold_threshold * 100)
                    ))

                # Sell signal: RSI crosses above overbought threshold
                elif prev_rsi <= self.overbought_threshold and current_rsi > self.overbought_threshold:
                    signals.append(MarketSignal(
                        type=SignalType.SELL,
                        symbol=pd.symbol,
                        strength=int((current_rsi - self.overbought_threshold) / (100 - self.overbought_threshold) * 100)
                    ))

            # Store current RSI for next iteration
            self.prev_rsi[pd.symbol] = current_rsi

        return signals


class RSIMeanReversionAlgorithm(Algorithm):
    """
    RSI mean reversion strategy.

    Strategy:
    - BUY when RSI is below 30 (oversold condition)
    - SELL when RSI returns to 50 or above (mean reversion)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.rsi_period = config.get('rsi_period', 14)
        self.oversold = config.get('oversold', 30)
        self.exit_rsi = config.get('exit_rsi', 50)
        self.in_position = {}  # Track if we're in a position

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        signals = []

        for pd in data:
            prices = list(self.price_history[pd.symbol])

            if len(prices) < self.rsi_period + 1:
                continue

            rsi = calculate_rsi(prices, period=self.rsi_period)

            if rsi is None:
                continue

            # Initialize position tracking
            if pd.symbol not in self.in_position:
                self.in_position[pd.symbol] = False

            # Entry: RSI is oversold
            if not self.in_position[pd.symbol] and rsi < self.oversold:
                signals.append(MarketSignal(
                    type=SignalType.BUY,
                    symbol=pd.symbol,
                    strength=75
                ))
                self.in_position[pd.symbol] = True

            # Exit: RSI returns to neutral or above
            elif self.in_position[pd.symbol] and rsi >= self.exit_rsi:
                signals.append(MarketSignal(
                    type=SignalType.SELL,
                    symbol=pd.symbol,
                    strength=75
                ))
                self.in_position[pd.symbol] = False

        return signals


class RSIWithTechnicalAnalyzer(Algorithm):
    """
    RSI algorithm using TechnicalAnalyzer directly.

    This provides access to additional RSI data:
    - rsi: The RSI value (0-100)
    - avg_gain: Average gain over period
    - avg_loss: Average loss over period
    - rs: Relative Strength (avg_gain / avg_loss)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.rsi_period = config.get('rsi_period', 14)
        self.oversold = config.get('oversold', 30)
        self.overbought = config.get('overbought', 70)

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        signals = []

        for pd in data:
            prices = list(self.price_history[pd.symbol])

            if len(prices) < self.rsi_period + 1:
                continue

            # Use TechnicalAnalyzer directly - returns RSI dataclass
            rsi_data = TechnicalAnalyzer.calculate_rsi(prices, period=self.rsi_period)

            if rsi_data is None:
                continue

            # Access RSI value
            rsi_value = rsi_data.rsi

            # Can also access additional data for advanced logic
            avg_gain = rsi_data.avg_gain
            avg_loss = rsi_data.avg_loss
            rs = rsi_data.rs  # None if avg_loss is 0

            # Example: Use RS for additional signal strength
            if rsi_value < self.oversold:
                # Stronger buy signal if RS is very low
                strength = 90 if rs is not None and rs < 0.5 else 75
                signals.append(MarketSignal(
                    type=SignalType.BUY,
                    symbol=pd.symbol,
                    strength=strength
                ))

            elif rsi_value > self.overbought:
                # Stronger sell signal if RS is very high
                strength = 90 if rs is not None and rs > 2.0 else 75
                signals.append(MarketSignal(
                    type=SignalType.SELL,
                    symbol=pd.symbol,
                    strength=strength
                ))

        return signals


class RSIDivergenceAlgorithm(Algorithm):
    """
    RSI divergence strategy (advanced).

    Looks for divergence between price and RSI:
    - Bullish divergence: Price makes lower low, but RSI makes higher low
    - Bearish divergence: Price makes higher high, but RSI makes lower high
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.rsi_period = config.get('rsi_period', 14)
        self.lookback = config.get('lookback', 5)  # How far back to look for divergence

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        signals = []

        for pd in data:
            prices = list(self.price_history[pd.symbol])

            # Need enough data for RSI series and lookback
            if len(prices) < self.rsi_period + self.lookback + 1:
                continue

            # Calculate RSI series
            rsi_series = calculate_rsi_series(prices, period=self.rsi_period)

            # Get recent prices and RSI values (non-None)
            recent_prices = prices[-self.lookback:]
            recent_rsi = [r for r in rsi_series[-self.lookback:] if r is not None]

            if len(recent_rsi) < self.lookback:
                continue

            # Find local extrema
            price_min_idx = recent_prices.index(min(recent_prices))
            price_max_idx = recent_prices.index(max(recent_prices))
            rsi_min_idx = recent_rsi.index(min(recent_rsi))
            rsi_max_idx = recent_rsi.index(max(recent_rsi))

            # Bullish divergence: Price lower low, RSI higher low
            if price_min_idx == len(recent_prices) - 1:  # Recent price is lowest
                if rsi_min_idx < price_min_idx:  # But RSI low was earlier
                    signals.append(MarketSignal(
                        type=SignalType.BUY,
                        symbol=pd.symbol,
                        strength=70
                    ))

            # Bearish divergence: Price higher high, RSI lower high
            if price_max_idx == len(recent_prices) - 1:  # Recent price is highest
                if rsi_max_idx < price_max_idx:  # But RSI high was earlier
                    signals.append(MarketSignal(
                        type=SignalType.SELL,
                        symbol=pd.symbol,
                        strength=70
                    ))

        return signals


# Example configuration for config.yaml
"""
simulator:
  data_provider:
    provider: "data_providers.test_data_provider.TestDataProvider"
    path: "data/AAPL.csv"
    truncate: 0
  algorithm:
    algorithm: "examples.rsi_example.RSIAlgorithm"
    history_length: 20  # Need enough for RSI calculation
    full_history: false
    rsi_period: 14
    oversold: 30
    overbought: 70
  order_manager:
    order_manager: "core.om.backtesting_om.BacktestingOM"
  portfolio:
    portfolio: "core.pf.single_symbol_portfolio.SingleSymbolPortfolio"
    symbol: "AAPL"
    cash: 100000
    keep_history: true
"""


# Example usage in a script
if __name__ == "__main__":
    from engines.simulator import Simulator
    from trading.engines.analysis_engine import AnalysisEngine

    # Create algorithm with custom config
    algo_config = {
        'history_length': 20,
        'rsi_period': 14,
        'oversold': 30,
        'overbought': 70
    }
    algo = RSIAlgorithm(algo_config)

    # Run backtest (assuming config.yaml is properly set up)
    sim = Simulator(cfg_section_to_use="simulator")
    sim.run()

    # Analyze results
    engine = AnalysisEngine(sim.portfolio, sim.order_manager)
    results = engine.run_full_analysis(
        run_name="RSI Strategy Test",
        description="14-period RSI with 30/70 thresholds",
        parameters={
            "rsi_period": 14,
            "oversold": 30,
            "overbought": 70
        },
        log_to_mlflow=True,
        save_charts_locally=True
    )

    # Print summary
    print(f"\nTotal Return: {results['metrics'].total_return_pct:.2f}%")
    print(f"Sharpe Ratio: {results['metrics'].sharpe_ratio:.2f}")
    print(f"Win Rate: {results['metrics'].win_rate:.1f}%")
    print(f"Total Trades: {results['metrics'].total_trades}")
