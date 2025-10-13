"""
Pytest configuration and shared fixtures for all tests
"""
import pytest
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from core.classes import (
    PriceData, Order, Position, MarketSignal,
    SignalType, OrderType, OrderAction, OrderStatus
)


@pytest.fixture
def sample_timestamp():
    """Returns a fixed timestamp for testing"""
    return datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_pricedata(sample_timestamp):
    """Returns a sample PriceData object"""
    return PriceData(
        symbol="AAPL",
        timestamp=sample_timestamp,
        open=150.0,
        high=152.0,
        low=149.0,
        close=151.0,
        volume=1000000.0,
        trade_count=5000,
        vwap=150.5,
        exchange=1.0
    )


@pytest.fixture
def sample_pricedata_list(sample_timestamp):
    """Returns a list of PriceData objects for multiple symbols"""
    return [
        PriceData(
            symbol="AAPL",
            timestamp=sample_timestamp,
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000.0
        ),
        PriceData(
            symbol="GOOGL",
            timestamp=sample_timestamp,
            open=2800.0,
            high=2820.0,
            low=2790.0,
            close=2810.0,
            volume=500000.0
        ),
        PriceData(
            symbol="MSFT",
            timestamp=sample_timestamp,
            open=380.0,
            high=385.0,
            low=378.0,
            close=383.0,
            volume=800000.0
        )
    ]


@pytest.fixture
def sample_position():
    """Returns a sample Position"""
    return Position(symbol="AAPL", quantity=100)


@pytest.fixture
def sample_market_signal():
    """Returns a sample MarketSignal"""
    return MarketSignal(
        type=SignalType.BUY,
        symbol="AAPL",
        strength=75
    )


@pytest.fixture
def sample_market_signals():
    """Returns a list of MarketSignals"""
    return [
        MarketSignal(type=SignalType.BUY, symbol="AAPL", strength=80),
        MarketSignal(type=SignalType.SELL, symbol="GOOGL", strength=60),
        MarketSignal(type=SignalType.BUY, symbol="MSFT", strength=90)
    ]


@pytest.fixture
def sample_order(sample_timestamp):
    """Returns a sample Order"""
    return Order(
        placed_datetime=sample_timestamp,
        executed_datetime=sample_timestamp,
        action=OrderAction.BUY,
        type=OrderType.MARKET,
        symbol="AAPL",
        price=151.0,
        quantity=100,
        cash=15100.0,
        status=OrderStatus.FILLED,
        tx_cost=1.0
    )


@pytest.fixture
def sample_pending_order(sample_timestamp):
    """Returns a pending Order"""
    return Order(
        placed_datetime=sample_timestamp,
        executed_datetime=None,
        action=OrderAction.BUY,
        type=OrderType.STOP,
        symbol="AAPL",
        price=155.0,
        quantity=100,
        cash=15500.0,
        status=OrderStatus.PENDING,
        tx_cost=1.0
    )


@pytest.fixture
def sample_csv_data(tmp_path):
    """Creates a temporary CSV file with sample market data"""
    data = {
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'GOOGL', 'GOOGL', 'GOOGL'],
        'timestamp': [
            '2024-01-15 10:00:00+00:00',
            '2024-01-15 10:01:00+00:00',
            '2024-01-15 10:02:00+00:00',
            '2024-01-15 10:00:00+00:00',
            '2024-01-15 10:01:00+00:00',
            '2024-01-15 10:02:00+00:00'
        ],
        'open': [150.0, 151.0, 152.0, 2800.0, 2810.0, 2820.0],
        'high': [151.0, 152.0, 153.0, 2810.0, 2820.0, 2830.0],
        'low': [149.0, 150.0, 151.0, 2790.0, 2800.0, 2810.0],
        'close': [151.0, 152.0, 153.0, 2810.0, 2820.0, 2830.0],
        'volume': [100000, 110000, 120000, 50000, 55000, 60000]
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def algorithm_config():
    """Returns a sample algorithm configuration"""
    return {
        "history_length": 10,
        "full_history": True
    }


@pytest.fixture
def portfolio_config():
    """Returns a sample portfolio configuration"""
    return {
        "cash": 100000.0,
        "keep_history": True,
        "symbol": "AAPL"
    }


@pytest.fixture
def data_provider_config(sample_csv_data):
    """Returns a sample data provider configuration"""
    return {
        "provider": "data_providers.test_data_provider.TestDataProvider",
        "path": str(sample_csv_data),
        "truncate": 0
    }