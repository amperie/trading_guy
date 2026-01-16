# Exception Handling Strategy

## Table of Contents
1. [Overview](#overview)
2. [Exception Hierarchy](#exception-hierarchy)
3. [Layer-Specific Handling](#layer-specific-handling)
4. [Best Practices by Component](#best-practices-by-component)
5. [Logging Integration](#logging-integration)
6. [Recovery Strategies](#recovery-strategies)
7. [Future: Distributed Architecture](#future-distributed-architecture)

---

## Overview

### Design Principles

1. **Fail Fast for Fatal Errors**: Configuration errors should crash immediately
2. **Recover Gracefully for Transient Errors**: Skip bad ticks, retry operations
3. **Never Lose Context**: Every exception carries symbol, timestamp, order_id, etc.
4. **Log Everything**: All exceptions logged with full context
5. **Type Safety**: Use specific exception types, not generic `Exception`
6. **Serializable**: Exceptions must be JSON-serializable for future distributed architecture

### Exception Categories

| Category | Severity | Action | Examples |
|----------|----------|--------|----------|
| **Configuration** | Fatal | Stop immediately | Missing config, invalid class path |
| **Data** | Recoverable | Skip tick or use fallback | Missing CSV, bad timestamp |
| **Algorithm** | Recoverable | Skip tick, log warning | Indicator calc error |
| **Order** | Recoverable | Cancel order, continue | Insufficient funds |
| **Portfolio** | Recoverable | Revert change, continue | Position error |

---

## Exception Hierarchy

```
TradingException (base)
├── ConfigurationError          # FATAL - stop execution
├── DataError                   # May be recoverable
│   ├── DataProviderError
│   ├── InvalidDataError
│   └── MissingDataError
├── AlgorithmError              # Recoverable - skip tick
│   ├── IndicatorCalculationError
│   └── SignalGenerationError
├── OrderError                  # Recoverable - cancel order
│   ├── InvalidOrderError
│   ├── InsufficientFundsError
│   └── OrderRejectedError
└── PortfolioError              # Recoverable - skip action
    ├── PositionError
    └── RiskLimitError
```

All exceptions in `core/exceptions.py`.

---

## Layer-Specific Handling

### 1. DataProvider Layer

**Strategy**: Raise early, fail fast for fatal errors. Allow graceful degradation for missing data.

```python
from trading.core.exceptions import DataProviderError, MissingDataError, wrap_exception
from utils.logger import Logger

logger = Logger().get_logger(__name__)


class TestDataProvider(DataProvider):
    def load_data(self):
        try:
            self.data = pd.read_csv(self.path)
        except FileNotFoundError as e:
            # FATAL: Can't proceed without data
            raise wrap_exception(
                e,
                "Data file not found",
                DataProviderError,
                path=self.path
            ) from e
        except pd.errors.ParserError as e:
            # FATAL: Corrupted data
            raise wrap_exception(
                e,
                "Failed to parse CSV data",
                DataProviderError,
                path=self.path
            ) from e

        # Validate required columns
        required = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing = set(required) - set(self.data.columns)
        if missing:
            raise InvalidDataError(
                "CSV missing required columns",
                context={'missing_columns': list(missing), 'path': self.path}
            )

    def iterate(self):
        """Generate price data, handling gaps gracefully."""
        for idx, row in self.data.iterrows():
            try:
                # Validate each row
                if pd.isna(row['close']) or row['close'] <= 0:
                    logger.warning(
                        f"Invalid price data at index {idx}, skipping",
                        extra={'row': row.to_dict()}
                    )
                    continue

                yield self._create_price_data(row)

            except Exception as e:
                # Log and skip bad row, don't crash entire iteration
                logger.error(
                    f"Error processing row {idx}: {e}",
                    exc_info=True,
                    extra={'row': row.to_dict() if hasattr(row, 'to_dict') else str(row)}
                )
                continue
```

**Key Points**:
- ✅ Raise `DataProviderError` for fatal issues (file not found, parse error)
- ✅ Skip individual bad rows with warning, don't crash iteration
- ✅ Validate data structure early
- ✅ Use `wrap_exception()` to preserve original exception

---

### 2. Algorithm Layer

**Strategy**: Handle errors gracefully, return empty signals on failure. Never crash the backtest.

```python
from trading.core.exceptions import AlgorithmError, IndicatorCalculationError
from utils.logger import Logger

logger = Logger().get_logger(__name__)


class MyAlgorithm(Algorithm):
    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        signals = []

        for price_data in data:
            try:
                # Get price history
                prices = self.price_history[price_data.symbol]

                # Calculate indicators (may fail)
                rsi = self._calculate_rsi(prices)
                macd = self._calculate_macd(prices)

                # Generate signal
                signal = self._generate_signal(price_data, rsi, macd)
                if signal:
                    signals.append(signal)

            except IndicatorCalculationError as e:
                # Log and skip this symbol - insufficient data is normal
                logger.debug(
                    f"Skipping {price_data.symbol}: {e.message}",
                    extra={'context': e.context}
                )
                continue

            except Exception as e:
                # Unexpected error - log with full traceback
                logger.error(
                    f"Unexpected error processing {price_data.symbol}",
                    exc_info=True,
                    extra={
                        'symbol': price_data.symbol,
                        'timestamp': price_data.timestamp,
                        'error_type': type(e).__name__
                    }
                )
                continue

        return signals

    def _calculate_rsi(self, prices):
        """Calculate RSI, raise specific exception on error."""
        if len(prices) < 14:
            raise IndicatorCalculationError(
                "Insufficient history for RSI calculation",
                context={'required': 14, 'available': len(prices)}
            )

        try:
            return TechnicalAnalyzer.calculate_rsi(prices, period=14)
        except Exception as e:
            raise wrap_exception(
                e,
                "RSI calculation failed",
                IndicatorCalculationError,
                period=14,
                price_count=len(prices)
            ) from e
```

**Key Points**:
- ✅ Use `try-except` inside the loop - one symbol failure doesn't crash others
- ✅ Distinguish between expected (insufficient data) and unexpected errors
- ✅ Return empty list on failure - let portfolio handle lack of signals
- ✅ Never let algorithm crash the backtest

---

### 3. Portfolio Layer

**Strategy**: Validate before acting, revert on error, maintain consistency.

```python
from trading.core.exceptions import PortfolioError, InsufficientFundsError, PositionError
from utils.logger import Logger

logger = Logger().get_logger(__name__)


class Portfolio:
    def process_tick_market_signals(self, signals, tick):
        """Process signals with error handling."""
        # Update pending orders first
        try:
            self._process_pending_orders(tick)
        except Exception as e:
            logger.error(
                "Error processing pending orders",
                exc_info=True,
                extra={'tick_count': len(tick)}
            )
            # Continue - don't let this stop new signal processing

        # Process new signals
        orders = []
        for signal in signals:
            try:
                order = self._process_single_signal(signal, tick)
                if order:
                    orders.append(order)
            except InsufficientFundsError as e:
                # Expected - log at info level
                logger.info(
                    f"Insufficient funds for {signal.symbol}",
                    extra=e.context
                )
            except PortfolioError as e:
                # Portfolio-specific error
                logger.warning(
                    f"Portfolio error processing {signal.symbol}: {e.message}",
                    extra=e.context
                )
            except Exception as e:
                # Unexpected error
                logger.error(
                    f"Unexpected error processing signal for {signal.symbol}",
                    exc_info=True,
                    extra={'signal': signal.__dict__}
                )

        return orders

    def _process_filled_order(self, order):
        """Update portfolio state with error recovery."""
        # Save state for rollback
        original_cash = self.cash
        original_positions = self.positions.copy()

        try:
            # Update portfolio state
            if order.action == OrderAction.BUY:
                cost = order.quantity * order.filled_price + order.tx_cost
                self.cash -= cost
                # ... update positions ...

            # Validate state
            if self.cash < 0:
                raise PositionError(
                    "Cash balance went negative",
                    context={'cash': self.cash, 'order_id': order.order_id}
                )

        except Exception as e:
            # Rollback on any error
            logger.error(
                f"Error processing filled order {order.order_id}, rolling back",
                exc_info=True
            )
            self.cash = original_cash
            self.positions = original_positions
            raise  # Re-raise after rollback
```

**Key Points**:
- ✅ Save state before modifying, rollback on error
- ✅ Validate state after updates
- ✅ Different log levels for expected vs unexpected errors
- ✅ One signal failure doesn't stop others

---

### 4. OrderManager Layer

**Strategy**: Validate orders strictly, reject invalid orders gracefully.

```python
from trading.core.exceptions import InvalidOrderError, OrderRejectedError
from utils.logger import Logger

logger = Logger().get_logger(__name__)


class BacktestingOM(OrderManager):
    def submit_order(self, order, tick, positions, pf_cash):
        """Submit order with validation."""
        try:
            # Validate order
            self._validate_order(order)

            # Submit to backend
            return self._submit_order_to_backend(order, tick, positions, pf_cash)

        except InvalidOrderError as e:
            # Log and return rejected order
            logger.warning(
                f"Order rejected: {e.message}",
                extra=e.context
            )
            order.status = OrderStatus.CANCELED
            return order

        except Exception as e:
            # Unexpected error
            logger.error(
                f"Unexpected error submitting order {order.order_id}",
                exc_info=True,
                extra={'order': order.__dict__}
            )
            order.status = OrderStatus.CANCELED
            return order

    def _validate_order(self, order):
        """Validate order parameters."""
        if order.quantity <= 0:
            raise InvalidOrderError(
                "Order quantity must be positive",
                context={
                    'order_id': order.order_id,
                    'quantity': order.quantity,
                    'symbol': order.symbol
                }
            )

        if order.price <= 0:
            raise InvalidOrderError(
                "Order price must be positive",
                context={
                    'order_id': order.order_id,
                    'price': order.price,
                    'symbol': order.symbol
                }
            )
```

**Key Points**:
- ✅ Validate before executing
- ✅ Return canceled order instead of raising
- ✅ Log with structured context

---

### 5. Simulator/Engine Layer

**Strategy**: Catch everything at top level, decide whether to continue or stop.

```python
from trading.core.exceptions import TradingException, ConfigurationError
from utils.logger import Logger

logger = Logger().get_logger(__name__)


class Simulator:
    def run(self):
        """Run simulation with top-level error handling."""
        try:
            # Configuration errors should stop immediately
            self._validate_configuration()

            logger.info("Starting backtest simulation")
            tick_count = 0
            error_count = 0
            max_consecutive_errors = 10

            for tick in self.data_provider.iterate():
                try:
                    tick_count += 1

                    # Run algorithm
                    signals = self.algorithm.on_data(tick)

                    # Process through portfolio
                    orders = self.portfolio.process_tick_market_signals(signals, tick)

                    # Update order statuses
                    self.order_manager.update_pending_orders(
                        self.portfolio.pending_orders_by_id.values(),
                        tick,
                        self.portfolio.positions,
                        self.portfolio.cash
                    )

                    # Reset error counter on success
                    error_count = 0

                except TradingException as e:
                    # Known error type - log and continue
                    error_count += 1
                    logger.warning(
                        f"Error on tick {tick_count}: {e.message}",
                        extra={'context': e.context, 'error_count': error_count}
                    )

                    if error_count >= max_consecutive_errors:
                        logger.error(
                            f"Too many consecutive errors ({error_count}), stopping"
                        )
                        break

                except Exception as e:
                    # Unexpected error - log with full traceback
                    error_count += 1
                    logger.error(
                        f"Unexpected error on tick {tick_count}",
                        exc_info=True,
                        extra={'tick_count': tick_count, 'error_count': error_count}
                    )

                    if error_count >= max_consecutive_errors:
                        logger.error("Too many errors, stopping simulation")
                        raise  # Re-raise to stop execution

            logger.info(
                f"Simulation completed: {tick_count} ticks processed, "
                f"{error_count} errors in final stretch"
            )

        except ConfigurationError as e:
            # Fatal error - log and re-raise
            logger.critical(f"Configuration error: {e.message}", extra=e.context)
            raise

        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
            raise

        except Exception as e:
            logger.critical("Fatal error in simulation", exc_info=True)
            raise

    def _validate_configuration(self):
        """Validate configuration, raise ConfigurationError on failure."""
        if not self.data_provider:
            raise ConfigurationError(
                "Data provider not configured",
                context={'config': self.cfg}
            )

        if not self.algorithm:
            raise ConfigurationError(
                "Algorithm not configured",
                context={'config': self.cfg}
            )
```

**Key Points**:
- ✅ Top-level try-except catches all errors
- ✅ Circuit breaker pattern (max consecutive errors)
- ✅ Distinguish fatal (configuration) from recoverable errors
- ✅ Always log before stopping
- ✅ Let KeyboardInterrupt propagate

---

## Logging Integration

### Structured Logging with Context

```python
from utils.logger import Logger

logger = Logger().get_logger(__name__)

# DON'T: Generic logging
logger.error("Order failed")

# DO: Structured logging with context
logger.error(
    "Order validation failed",
    extra={
        'order_id': order.order_id,
        'symbol': order.symbol,
        'quantity': order.quantity,
        'reason': 'insufficient_funds',
        'available_cash': portfolio.cash,
        'required_cash': required
    }
)
```

### Exception Logging Pattern

```python
try:
    risky_operation()
except SpecificError as e:
    logger.warning(f"Expected error: {e.message}", extra=e.context)
except Exception as e:
    logger.error(
        "Unexpected error",
        exc_info=True,  # Include full traceback
        extra={
            'operation': 'risky_operation',
            'error_type': type(e).__name__
        }
    )
```

---

## Recovery Strategies

### 1. Retry Pattern (for transient errors)

```python
from typing import Callable, TypeVar, Optional
import time

T = TypeVar('T')

def retry_on_error(
    func: Callable[[], T],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Optional[T]:
    """
    Retry function on specified exceptions.

    Args:
        func: Function to retry
        max_retries: Maximum retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch

    Returns:
        Function result or None if all retries failed
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            if attempt == max_retries:
                logger.error(
                    f"Failed after {max_retries} retries",
                    exc_info=True,
                    extra={'function': func.__name__, 'attempts': attempt + 1}
                )
                return None

            wait_time = delay * (backoff ** attempt)
            logger.warning(
                f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}",
                extra={'function': func.__name__}
            )
            time.sleep(wait_time)

# Usage
data = retry_on_error(
    lambda: self._fetch_from_api(),
    max_retries=3,
    exceptions=(ConnectionError, TimeoutError)
)
```

### 2. Circuit Breaker Pattern

```python
class CircuitBreaker:
    """Prevent cascading failures by stopping after too many errors."""

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable[[], T]) -> Optional[T]:
        """Execute function with circuit breaker protection."""
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                logger.warning("Circuit breaker is OPEN, rejecting call")
                return None

        try:
            result = func()
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
                logger.info("Circuit breaker reset to CLOSED")
            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
                logger.error(
                    f"Circuit breaker opened after {self.failure_count} failures",
                    extra={'threshold': self.failure_threshold}
                )

            raise

# Usage in DataProvider
circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60.0)

def fetch_data(self):
    return circuit_breaker.call(lambda: self._do_fetch())
```

### 3. Fallback Pattern

```python
def calculate_indicator_with_fallback(
    prices: deque,
    primary_method: Callable,
    fallback_method: Callable,
    default_value: Optional[float] = None
):
    """Try primary method, fallback to secondary, finally return default."""
    try:
        return primary_method(prices)
    except IndicatorCalculationError as e:
        logger.debug(f"Primary method failed: {e.message}, trying fallback")
        try:
            return fallback_method(prices)
        except Exception as e2:
            logger.warning(
                f"Both methods failed, using default",
                extra={'primary_error': str(e), 'fallback_error': str(e2)}
            )
            return default_value

# Usage
rsi = calculate_indicator_with_fallback(
    prices,
    primary_method=lambda p: TechnicalAnalyzer.calculate_rsi(p, period=14),
    fallback_method=lambda p: TechnicalAnalyzer.calculate_rsi(p, period=10),
    default_value=50.0  # Neutral RSI
)
```

---

## Future: Distributed Architecture

When moving to a service bus with distributed components:

### 1. Serializable Exceptions

```python
import json
from typing import Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class SerializableException:
    """Exception that can be sent across process boundaries."""
    exception_type: str
    message: str
    context: Dict[str, Any]
    timestamp: str
    correlation_id: str  # For distributed tracing
    service_name: str

    def to_json(self) -> str:
        """Serialize to JSON for message queue."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'SerializableException':
        """Deserialize from JSON."""
        return cls(**json.loads(json_str))

    @classmethod
    def from_exception(
        cls,
        exc: TradingException,
        correlation_id: str,
        service_name: str
    ) -> 'SerializableException':
        """Create from TradingException."""
        return cls(
            exception_type=type(exc).__name__,
            message=exc.message,
            context=exc.context,
            timestamp=exc.timestamp.isoformat(),
            correlation_id=correlation_id,
            service_name=service_name
        )
```

### 2. Dead Letter Queue Pattern

```python
class ErrorQueue:
    """Queue for failed messages requiring manual intervention."""

    def __init__(self, queue_name: str = "trading_errors"):
        self.queue_name = queue_name
        self.errors = []

    def push(self, error: SerializableException, original_message: Dict):
        """Add failed message to error queue."""
        self.errors.append({
            'error': error,
            'message': original_message,
            'queued_at': datetime.now().isoformat()
        })

        # In production: send to actual message queue
        logger.error(
            f"Message moved to dead letter queue",
            extra={
                'queue': self.queue_name,
                'error_type': error.exception_type,
                'correlation_id': error.correlation_id
            }
        )

    def retry_all(self):
        """Retry all messages in error queue."""
        while self.errors:
            item = self.errors.pop(0)
            yield item['message']
```

### 3. Correlation IDs for Distributed Tracing

```python
import uuid
from contextvars import ContextVar

# Thread-safe correlation ID storage
correlation_id: ContextVar[str] = ContextVar('correlation_id', default=None)

def set_correlation_id(corr_id: str = None):
    """Set correlation ID for current context."""
    if corr_id is None:
        corr_id = str(uuid.uuid4())
    correlation_id.set(corr_id)
    return corr_id

def get_correlation_id() -> str:
    """Get correlation ID for current context."""
    corr_id = correlation_id.get()
    if corr_id is None:
        corr_id = set_correlation_id()
    return corr_id

# Usage in distributed services
class AlgorithmService:
    def process_message(self, message: Dict):
        # Extract or create correlation ID
        corr_id = message.get('correlation_id') or set_correlation_id()

        try:
            # Process message
            result = self.algorithm.on_data(message['data'])

            # Include correlation ID in response
            return {
                'result': result,
                'correlation_id': corr_id
            }

        except TradingException as e:
            # Convert to serializable exception with correlation ID
            error = SerializableException.from_exception(
                e,
                correlation_id=corr_id,
                service_name='algorithm_service'
            )

            # Send to error queue
            self.error_queue.push(error, message)

            # Log with correlation ID
            logger.error(
                f"Algorithm service error",
                extra={
                    'correlation_id': corr_id,
                    'error': error.to_json()
                }
            )
```

### 4. Service Health Checks

```python
from enum import Enum
from typing import List, Optional

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthCheck:
    """Health check for distributed service."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.error_count = 0
        self.last_error_time = None
        self.status = ServiceStatus.HEALTHY

    def record_error(self, error: Exception):
        """Record error and update health status."""
        self.error_count += 1
        self.last_error_time = datetime.now()

        # Update status based on error rate
        if self.error_count > 50:
            self.status = ServiceStatus.UNHEALTHY
        elif self.error_count > 10:
            self.status = ServiceStatus.DEGRADED

        logger.warning(
            f"Service health degraded",
            extra={
                'service': self.service_name,
                'status': self.status.value,
                'error_count': self.error_count
            }
        )

    def record_success(self):
        """Record successful operation."""
        # Gradually recover
        if self.error_count > 0:
            self.error_count = max(0, self.error_count - 1)

        if self.error_count == 0:
            self.status = ServiceStatus.HEALTHY
        elif self.error_count < 10:
            self.status = ServiceStatus.DEGRADED

    def get_status(self) -> Dict:
        """Get current health status."""
        return {
            'service': self.service_name,
            'status': self.status.value,
            'error_count': self.error_count,
            'last_error': self.last_error_time.isoformat() if self.last_error_time else None
        }
```

---

## Summary: Best Practices Checklist

### For Current Monolithic Architecture:

- [ ] Use specific exception types from `core.exceptions`
- [ ] Add context to all exceptions (symbol, timestamp, etc.)
- [ ] Log before raising or re-raising
- [ ] Use structured logging with `extra` parameter
- [ ] Catch specific exceptions, not bare `except:`
- [ ] Implement circuit breakers for external dependencies
- [ ] Save state before modifications, rollback on error
- [ ] Never let one symbol failure crash the whole system
- [ ] Use different log levels: DEBUG for expected, ERROR for unexpected
- [ ] Validate early, fail fast for configuration errors

### For Future Distributed Architecture:

- [ ] Make all exceptions JSON-serializable
- [ ] Use correlation IDs for distributed tracing
- [ ] Implement dead letter queues for failed messages
- [ ] Add service health checks and status endpoints
- [ ] Use message acknowledgment patterns
- [ ] Implement idempotency for retry safety
- [ ] Add timeout and retry logic to all service calls
- [ ] Monitor error rates across services
- [ ] Use distributed tracing (OpenTelemetry/Jaeger)
- [ ] Implement graceful degradation patterns

---

## See Also

- `core/exceptions.py` - Exception class definitions
- `utils/logger.py` - Logging configuration
- `engines/simulator.py` - Top-level error handling example
