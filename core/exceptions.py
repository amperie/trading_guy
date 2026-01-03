"""
Custom exceptions for the trading framework.

Exception Hierarchy:
-------------------
TradingException (base)
├── ConfigurationError (fatal - stop immediately)
├── DataError (may be recoverable)
│   ├── DataProviderError
│   ├── InvalidDataError
│   └── MissingDataError
├── AlgorithmError (recoverable - skip tick)
│   ├── IndicatorCalculationError
│   └── SignalGenerationError
├── OrderError (recoverable - cancel order)
│   ├── InvalidOrderError
│   ├── InsufficientFundsError
│   └── OrderRejectedError
└── PortfolioError (recoverable - skip action)
    ├── PositionError
    └── RiskLimitError

Design Principles:
-----------------
1. Fatal errors inherit directly from TradingException
2. Recoverable errors have specific subtypes
3. All exceptions carry context (timestamp, symbol, etc.)
4. Exceptions are logged automatically via __init__
"""

from typing import Optional, Dict, Any
from datetime import datetime


class TradingException(Exception):
    """
    Base exception for all trading-related errors.

    All custom exceptions should inherit from this.
    Provides automatic logging and context tracking.
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize trading exception with context.

        Args:
            message: Human-readable error description
            context: Additional context (symbol, timestamp, order_id, etc.)
            original_exception: Original exception if wrapping another error
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.original_exception = original_exception
        self.timestamp = datetime.now()

        # Add original exception info to context
        if original_exception:
            self.context['original_error'] = str(original_exception)
            self.context['original_type'] = type(original_exception).__name__

    def __str__(self):
        """Format exception with context for logging."""
        parts = [self.message]

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"[{context_str}]")

        if self.original_exception:
            parts.append(f"(caused by: {type(self.original_exception).__name__})")

        return " ".join(parts)


# =============================================================================
# FATAL ERRORS - Should stop execution immediately
# =============================================================================

class ConfigurationError(TradingException):
    """
    Critical configuration error that prevents system startup.

    Examples:
    - Missing required config keys
    - Invalid class paths
    - Conflicting settings
    """
    pass


# =============================================================================
# DATA ERRORS - May be recoverable depending on context
# =============================================================================

class DataError(TradingException):
    """Base class for data-related errors."""
    pass


class DataProviderError(DataError):
    """
    Error loading or processing data from data provider.

    Examples:
    - CSV file not found
    - API connection failure
    - Corrupted data file
    """
    pass


class InvalidDataError(DataError):
    """
    Data validation failed (wrong format, missing columns, etc.).

    Examples:
    - Missing required columns
    - Invalid timestamp format
    - Negative prices or volumes
    """
    pass


class MissingDataError(DataError):
    """
    Required data is missing (gaps in time series, etc.).

    Examples:
    - No data for requested symbol
    - Gaps in historical data
    - Timestamp not found
    """
    pass


# =============================================================================
# ALGORITHM ERRORS - Usually recoverable (skip tick, use default)
# =============================================================================

class AlgorithmError(TradingException):
    """Base class for algorithm execution errors."""
    pass


class IndicatorCalculationError(AlgorithmError):
    """
    Error calculating technical indicator.

    Examples:
    - Insufficient history for calculation
    - Division by zero in indicator formula
    - Invalid parameter values
    """
    pass


class SignalGenerationError(AlgorithmError):
    """
    Error generating trading signals.

    Examples:
    - Invalid signal strength value
    - Unable to determine signal type
    - Inconsistent signal data
    """
    pass


# =============================================================================
# ORDER ERRORS - Recoverable (cancel/reject order, log, continue)
# =============================================================================

class OrderError(TradingException):
    """Base class for order-related errors."""
    pass


class InvalidOrderError(OrderError):
    """
    Order validation failed.

    Examples:
    - Negative quantity
    - Invalid order type
    - Missing required fields
    """
    pass


class InsufficientFundsError(OrderError):
    """
    Not enough cash to execute order.

    This is usually expected behavior, not a bug.
    """
    pass


class OrderRejectedError(OrderError):
    """
    Order was rejected by order manager or broker.

    Examples:
    - Risk limits exceeded
    - Market closed
    - Symbol not tradeable
    """
    pass


# =============================================================================
# PORTFOLIO ERRORS - Recoverable (skip action, revert changes)
# =============================================================================

class PortfolioError(TradingException):
    """Base class for portfolio management errors."""
    pass


class PositionError(PortfolioError):
    """
    Error managing positions.

    Examples:
    - Trying to sell more shares than owned
    - Position state inconsistency
    - Invalid position update
    """
    pass


class RiskLimitError(PortfolioError):
    """
    Risk management limit exceeded.

    Examples:
    - Max position size exceeded
    - Portfolio concentration too high
    - Drawdown limit reached
    """
    pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def wrap_exception(
    original: Exception,
    message: str,
    exception_class: type = TradingException,
    **context
) -> TradingException:
    """
    Wrap a generic exception in a trading-specific exception.

    Preserves the original exception and adds context.

    Args:
        original: Original exception to wrap
        message: New descriptive message
        exception_class: Which TradingException subclass to use
        **context: Additional context to attach

    Returns:
        Trading exception with original exception preserved

    Example:
        try:
            df = pd.read_csv(path)
        except FileNotFoundError as e:
            raise wrap_exception(
                e,
                "Failed to load price data",
                DataProviderError,
                path=path,
                symbol=symbol
            ) from e
    """
    return exception_class(
        message=message,
        context=context,
        original_exception=original
    )
