"""
Exception Handling Examples

This file demonstrates how to properly handle exceptions in different
components of the trading framework.

Run: python examples/exception_handling_example.py
"""

from collections import deque
from datetime import datetime
from typing import Optional

from core.exceptions import (
    TradingException,
    ConfigurationError,
    DataProviderError,
    InvalidDataError,
    AlgorithmError,
    IndicatorCalculationError,
    InsufficientFundsError,
    wrap_exception
)
from core.classes import PriceData, MarketSignal, SignalType
from core.algorithm import Algorithm
from utils.logger import Logger

logger = Logger().get_logger(__name__)


# =============================================================================
# EXAMPLE 1: DataProvider with Robust Error Handling
# =============================================================================

class RobustDataProvider:
    """Example of proper error handling in a data provider."""

    def __init__(self, path: str):
        self.path = path
        self.data = None

    def load_data(self):
        """Load data with proper error handling."""
        import pandas as pd

        try:
            logger.info(f"Loading data from {self.path}")
            self.data = pd.read_csv(self.path)

        except FileNotFoundError as e:
            # FATAL: Can't proceed without data file
            raise wrap_exception(
                e,
                "Data file not found - cannot proceed",
                DataProviderError,
                path=self.path,
                cwd=os.getcwd()
            ) from e

        except pd.errors.ParserError as e:
            # FATAL: Corrupted data file
            raise wrap_exception(
                e,
                "Failed to parse CSV file",
                DataProviderError,
                path=self.path
            ) from e

        except Exception as e:
            # Unexpected error
            raise wrap_exception(
                e,
                "Unexpected error loading data",
                DataProviderError,
                path=self.path
            ) from e

        # Validate data structure
        self._validate_data()
        logger.info(f"Successfully loaded {len(self.data)} rows")

    def _validate_data(self):
        """Validate data structure."""
        required_columns = ['timestamp', 'symbol', 'close']
        missing = set(required_columns) - set(self.data.columns)

        if missing:
            raise InvalidDataError(
                "CSV missing required columns",
                context={
                    'required_columns': required_columns,
                    'actual_columns': list(self.data.columns),
                    'missing_columns': list(missing),
                    'path': self.path
                }
            )

    def iterate(self):
        """Iterate over data with graceful error handling."""
        if self.data is None:
            raise DataProviderError("Data not loaded - call load_data() first")

        for idx, row in self.data.iterrows():
            try:
                # Validate row data
                if pd.isna(row['close']) or row['close'] <= 0:
                    logger.warning(
                        f"Invalid close price at row {idx}, skipping",
                        extra={
                            'row_index': idx,
                            'close': row.get('close'),
                            'symbol': row.get('symbol')
                        }
                    )
                    continue

                # Create and yield PriceData
                yield self._create_price_data(row)

            except KeyError as e:
                logger.error(
                    f"Missing required column at row {idx}: {e}",
                    extra={'row_index': idx, 'available_columns': list(row.index)}
                )
                continue

            except Exception as e:
                logger.error(
                    f"Unexpected error processing row {idx}",
                    exc_info=True,
                    extra={'row_index': idx}
                )
                continue


# =============================================================================
# EXAMPLE 2: Algorithm with Graceful Degradation
# =============================================================================

class RobustAlgorithm(Algorithm):
    """Example of proper error handling in an algorithm."""

    def __init__(self, cfg=None, history_length=20):
        super().__init__(cfg, history_length)
        self.error_count = 0
        self.max_errors_per_run = 100

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        """Process data with robust error handling."""
        signals = []

        for price_data in data:
            try:
                # Calculate indicators with fallback
                signal = self._generate_signal_with_fallback(price_data)
                if signal:
                    signals.append(signal)

            except IndicatorCalculationError as e:
                # Expected error (insufficient data) - log at debug level
                logger.debug(
                    f"Cannot generate signal for {price_data.symbol}: {e.message}",
                    extra=e.context
                )
                continue

            except AlgorithmError as e:
                # Algorithm-specific error - log warning
                logger.warning(
                    f"Algorithm error for {price_data.symbol}: {e.message}",
                    extra=e.context
                )
                self.error_count += 1
                continue

            except Exception as e:
                # Unexpected error - log with full traceback
                logger.error(
                    f"Unexpected error processing {price_data.symbol}",
                    exc_info=True,
                    extra={
                        'symbol': price_data.symbol,
                        'timestamp': price_data.timestamp,
                        'price': price_data.close
                    }
                )
                self.error_count += 1
                continue

        # Check if too many errors
        if self.error_count > self.max_errors_per_run:
            raise AlgorithmError(
                "Too many errors in algorithm execution",
                context={
                    'error_count': self.error_count,
                    'max_allowed': self.max_errors_per_run
                }
            )

        return signals

    def _generate_signal_with_fallback(
        self,
        price_data: PriceData
    ) -> Optional[MarketSignal]:
        """Generate signal with fallback strategy."""
        prices = self.price_history[price_data.symbol]

        # Try primary method
        try:
            return self._calculate_rsi_signal(prices, price_data)
        except IndicatorCalculationError as e:
            logger.debug(f"RSI calculation failed for {price_data.symbol}, trying SMA")

            # Try fallback method
            try:
                return self._calculate_sma_signal(prices, price_data)
            except IndicatorCalculationError:
                logger.debug(f"SMA calculation also failed for {price_data.symbol}")
                raise  # Re-raise original error

    def _calculate_rsi_signal(
        self,
        prices: deque,
        price_data: PriceData
    ) -> MarketSignal:
        """Calculate signal using RSI."""
        if len(prices) < 14:
            raise IndicatorCalculationError(
                "Insufficient history for RSI",
                context={
                    'required': 14,
                    'available': len(prices),
                    'symbol': price_data.symbol
                }
            )

        # Simplified RSI logic
        avg_gain = sum(max(0, prices[i] - prices[i-1]) for i in range(1, 15)) / 14
        avg_loss = sum(max(0, prices[i-1] - prices[i]) for i in range(1, 15)) / 14

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # Generate signal
        if rsi < 30:
            return MarketSignal(SignalType.BUY, price_data.symbol, strength=80)
        elif rsi > 70:
            return MarketSignal(SignalType.SELL, price_data.symbol, strength=80)
        else:
            return None


# =============================================================================
# EXAMPLE 3: Portfolio with State Rollback
# =============================================================================

class RobustPortfolio:
    """Example of proper error handling in portfolio management."""

    def __init__(self, initial_cash: float = 100000):
        self.cash = initial_cash
        self.positions = {}
        self.transaction_log = []

    def execute_order(self, symbol: str, quantity: int, price: float):
        """Execute order with state rollback on error."""
        # Save state for potential rollback
        original_cash = self.cash
        original_positions = self.positions.copy()

        try:
            # Validate order
            cost = quantity * price
            if cost > self.cash:
                raise InsufficientFundsError(
                    "Not enough cash to execute order",
                    context={
                        'symbol': symbol,
                        'required': cost,
                        'available': self.cash,
                        'quantity': quantity,
                        'price': price
                    }
                )

            # Execute order
            self.cash -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity

            # Validate state
            if self.cash < 0:
                raise Exception("Cash went negative - should never happen")

            # Log transaction
            self.transaction_log.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'cost': cost
            })

            logger.info(
                f"Order executed: {quantity} shares of {symbol} at ${price:.2f}",
                extra={
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price,
                    'remaining_cash': self.cash
                }
            )

        except InsufficientFundsError:
            # Expected error - no rollback needed (no state changed)
            raise

        except Exception as e:
            # Unexpected error - rollback state
            logger.error(
                "Error executing order, rolling back state",
                exc_info=True,
                extra={
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price
                }
            )

            # Rollback
            self.cash = original_cash
            self.positions = original_positions

            # Re-raise
            raise


# =============================================================================
# EXAMPLE 4: Demonstrating Exception Hierarchy
# =============================================================================

def demonstrate_exception_handling():
    """Demonstrate exception handling patterns."""
    import os

    print("=" * 70)
    print("Exception Handling Examples")
    print("=" * 70)

    # Example 1: Configuration Error (Fatal)
    print("\n1. Configuration Error (Fatal - would stop execution)")
    try:
        if not os.path.exists('/fake/path/data.csv'):
            raise ConfigurationError(
                "Data file not found in configuration",
                context={'path': '/fake/path/data.csv'}
            )
    except ConfigurationError as e:
        print(f"   FATAL ERROR: {e}")
        print(f"   Context: {e.context}")
        # In real code: sys.exit(1)

    # Example 2: Data Error (Recoverable)
    print("\n2. Data Error (Recoverable - skip and continue)")
    try:
        raise InvalidDataError(
            "Invalid price in row 145",
            context={'row': 145, 'price': -10.5, 'symbol': 'AAPL'}
        )
    except InvalidDataError as e:
        print(f"   WARNING: {e}")
        print(f"   Context: {e.context}")
        print("   Action: Skip this row and continue")

    # Example 3: Algorithm Error (Recoverable)
    print("\n3. Algorithm Error (Recoverable - return empty signal)")
    try:
        raise IndicatorCalculationError(
            "Insufficient data for RSI calculation",
            context={'required': 14, 'available': 5, 'symbol': 'TSLA'}
        )
    except IndicatorCalculationError as e:
        print(f"   DEBUG: {e}")
        print(f"   Context: {e.context}")
        print("   Action: Return empty signal list")

    # Example 4: Order Error (Recoverable)
    print("\n4. Order Error (Recoverable - cancel order)")
    try:
        raise InsufficientFundsError(
            "Cannot buy 100 shares",
            context={
                'symbol': 'NVDA',
                'required': 50000,
                'available': 10000,
                'quantity': 100
            }
        )
    except InsufficientFundsError as e:
        print(f"   INFO: {e}")
        print(f"   Context: {e.context}")
        print("   Action: Cancel order and continue")

    # Example 5: Wrapped Exception
    print("\n5. Wrapped Exception (preserves original error)")
    try:
        try:
            # Simulate pandas error
            import pandas as pd
            pd.read_csv('/fake/file.csv')
        except FileNotFoundError as e:
            raise wrap_exception(
                e,
                "Failed to load price data",
                DataProviderError,
                path='/fake/file.csv',
                operation='load_historical_data'
            ) from e
    except DataProviderError as e:
        print(f"   ERROR: {e}")
        print(f"   Original error: {e.original_exception}")
        print(f"   Context: {e.context}")

    print("\n" + "=" * 70)


# =============================================================================
# EXAMPLE 5: Testing Recovery Strategies
# =============================================================================

def test_retry_pattern():
    """Demonstrate retry pattern."""
    from time import sleep

    print("\n" + "=" * 70)
    print("Retry Pattern Example")
    print("=" * 70)

    attempt_count = [0]  # Use list to modify in closure

    def flaky_operation():
        """Operation that fails first 2 times, succeeds on 3rd."""
        attempt_count[0] += 1
        print(f"   Attempt {attempt_count[0]}...")

        if attempt_count[0] < 3:
            raise ConnectionError("Temporary network error")

        return "Success!"

    # Retry logic
    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            result = flaky_operation()
            print(f"   ✓ Operation succeeded: {result}")
            break
        except ConnectionError as e:
            if attempt == max_retries:
                print(f"   ✗ Failed after {max_retries} retries")
                raise
            print(f"   ⚠ Retrying after error: {e}")
            sleep(0.1)  # Wait before retry

    print("=" * 70)


if __name__ == "__main__":
    demonstrate_exception_handling()
    test_retry_pattern()

    print("\n✓ All examples completed successfully!")
    print("\nKey Takeaways:")
    print("  1. Use specific exception types")
    print("  2. Always include context in exceptions")
    print("  3. Log before raising or handling")
    print("  4. Different errors need different strategies:")
    print("     - Fatal: Stop immediately")
    print("     - Recoverable: Skip/retry and continue")
    print("  5. Wrap generic exceptions in domain-specific ones")
    print("  6. Implement retry logic for transient errors")
    print("  7. Save state before modifications, rollback on error")
