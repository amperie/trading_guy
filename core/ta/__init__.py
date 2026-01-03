"""
Technical Analysis module for trading algorithms.

Provides dataclasses and calculation methods for common technical indicators
including MACD, RSI, Bollinger Bands, Stochastic, ATR, ADX, CCI, and Williams %R.
"""

from core.ta.analyzer import (
    TechnicalAnalyzer,
    MACD,
    RSI,
    BollingerBands,
    Stochastic,
    ATR,
    ADX,
    CCI,
    WilliamsR,
    EMA,
    SMA
)

__all__ = [
    'TechnicalAnalyzer',
    'MACD',
    'RSI',
    'BollingerBands',
    'Stochastic',
    'ATR',
    'ADX',
    'CCI',
    'WilliamsR',
    'EMA',
    'SMA'
]
