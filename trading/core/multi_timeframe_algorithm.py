"""
Base class for multi-timeframe trading algorithms.

Raw 1-min ticks are aggregated in-process into N user-defined timeframes.
Subclasses implement on_mtf_data() and receive both the raw tick and a dict
of newly-completed bars keyed by period (minutes).
"""
from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, final

from trading.core.algorithm import Algorithm
from trading.core.classes import MarketSignal, PriceData
from utils.logger import Logger

logger = Logger().get_logger(__name__)


class MultiTimeframeAlgorithm(Algorithm):
    """Base class for algorithms that reason across multiple bar timeframes.

    Extends Algorithm to maintain separate completed-bar histories for N
    registered timeframes.  Raw ticks are aggregated in-process; no external
    aggregation engine is required.

    Config keys (in addition to Algorithm base-class keys):
        timeframes (list[int]): Aggregation periods in minutes, e.g. [5, 15, 60].
            Required.
        market_open_hour (int): Hour of market open for window alignment. Default 9.
        market_open_minute (int): Minute of market open. Default 30.
        use_market_open (bool): Align windows to market open. Default True.

    Attributes (in addition to Algorithm base):
        bar_history (dict[int, dict[str, deque[PriceData]]]): Completed bars per
            timeframe per symbol. e.g. self.bar_history[60]["SPY"].
        timeframes (list[int]): Registered aggregation periods in minutes, sorted.

    Warmup:
        required_warmup_bars defaults to history_length * max(timeframes) so the
        slowest timeframe's bar history is fully populated before signals fire.
        Override required_warmup_bars in your subclass to adjust this threshold.

    Subclassing:
        Implement on_mtf_data() with your strategy logic.
        Do NOT override on_data_logic() — it is @final in this class.

    Example::

        class MyDualTFAlgo(MultiTimeframeAlgorithm):
            def on_mtf_data(
                self,
                tick: list[PriceData],
                new_bars: dict[int, list[PriceData]],
            ) -> list[MarketSignal]:
                # Act only when a new 15-min bar closes
                if 15 not in new_bars:
                    return []
                fast = list(self.bar_history[5]["SPY"])
                slow = list(self.bar_history[15]["SPY"])
                if not fast or not slow:
                    return []
                if fast[-1].close > slow[-1].close:
                    return [MarketSignal(SignalType.BUY, "SPY", 75)]
                return []
    """

    def __init__(self, cfg: dict[str, Any] | None = None, history_length: int = 0):
        super().__init__(cfg, history_length)

        timeframes_raw = self.cfg.get("timeframes")
        if not timeframes_raw:
            raise ValueError(
                "MultiTimeframeAlgorithm requires 'timeframes' in config "
                "(list of int minutes, e.g. [5, 15, 60])"
            )
        self.timeframes: list[int] = sorted(int(tf) for tf in timeframes_raw)

        self.market_open_hour: int = int(self.cfg.get("market_open_hour", 9))
        self.market_open_minute: int = int(self.cfg.get("market_open_minute", 30))
        self.use_market_open: bool = bool(self.cfg.get("use_market_open", True))

        hl = self.history_length
        if hl > 0:
            self.bar_history: dict[int, dict[str, deque]] = {
                tf: defaultdict(lambda bound=hl: deque(maxlen=bound))
                for tf in self.timeframes
            }
        else:
            self.bar_history: dict[int, dict[str, deque]] = {
                tf: defaultdict(deque) for tf in self.timeframes
            }

        # Per-timeframe aggregation state: in-progress OHLCV bar and window boundary
        self._mtf_agg: dict[int, dict[str, PriceData]] = {tf: {} for tf in self.timeframes}
        self._mtf_window_end: dict[int, dict[str, datetime]] = {tf: {} for tf in self.timeframes}

        logger.debug(
            f"{self.__class__.__name__} initialized: timeframes={self.timeframes} "
            f"history_length={self.history_length}"
        )

    @property
    def required_warmup_bars(self) -> int:
        """Defaults to history_length × max(timeframes) raw ticks.

        This ensures the slowest timeframe's bar history is fully populated
        before any signals are emitted.  Override to use a different threshold.
        """
        if self._required_warmup_bars_override is not None:
            return self._required_warmup_bars_override
        if not self.timeframes or self.history_length == 0:
            return self.history_length
        return self.history_length * max(self.timeframes)

    @final
    def on_data_logic(self, tick: list[PriceData]) -> list[MarketSignal]:
        """Aggregate raw tick into all registered timeframes, then call on_mtf_data."""
        new_bars: dict[int, list[PriceData]] = {}
        for tf in self.timeframes:
            completed = self._aggregate_tick_for_period(tf, tick)
            if completed:
                new_bars[tf] = completed
                for bar in completed:
                    self.bar_history[tf][bar.symbol].append(bar)
        return self.on_mtf_data(tick, new_bars)

    @abstractmethod
    def on_mtf_data(
        self,
        tick: list[PriceData],
        new_bars: dict[int, list[PriceData]],
    ) -> list[MarketSignal]:
        """Strategy logic implementation.  Override this in your subclass.

        Called once per raw tick after history and bar aggregations are updated.

        Args:
            tick: Raw PriceData for this tick (one entry per symbol in the stream).
            new_bars: Dict mapping timeframe period (minutes) to the list of
                newly-completed PriceData bars for that period.  A key is present
                only when at least one bar completed on this tick; multiple symbols
                may appear in the list.

        Returns:
            List of MarketSignal objects.  Return [] for no signal.
        """
        ...

    # ── Aggregation internals ────────────────────────────────────────────────

    def _aggregate_tick_for_period(self, period: int, tick: list[PriceData]) -> list[PriceData]:
        """Accumulate raw PriceData into the given period; return any completed bars."""
        completed: list[PriceData] = []
        for pd in tick:
            symbol = pd.symbol
            window_end = self._get_window_end(pd.timestamp, period)

            if symbol not in self._mtf_agg[period]:
                self._mtf_agg[period][symbol] = self._make_agg_bar(pd, window_end)
                self._mtf_window_end[period][symbol] = window_end
                # A tick that lands exactly on a boundary is its own window.
                if pd.timestamp == window_end:
                    bar = self._flush_agg(period, symbol)
                    if bar is not None:
                        completed.append(bar)
                continue

            current_end = self._mtf_window_end[period][symbol]
            if pd.timestamp < current_end:
                self._update_agg_bar(period, symbol, pd)
            elif pd.timestamp == current_end:
                self._update_agg_bar(period, symbol, pd)
                bar = self._flush_agg(period, symbol)
                if bar is not None:
                    completed.append(bar)
            else:
                # Tick belongs to a new window — flush the old one first.
                bar = self._flush_agg(period, symbol)
                if bar is not None:
                    completed.append(bar)
                self._mtf_agg[period][symbol] = self._make_agg_bar(pd, window_end)
                self._mtf_window_end[period][symbol] = window_end
                if pd.timestamp == window_end:
                    bar = self._flush_agg(period, symbol)
                    if bar is not None:
                        completed.append(bar)

        return completed

    def _get_window_end(self, ts: datetime, period: int) -> datetime:
        """Return the end timestamp of the aggregation window containing ts."""
        if self.use_market_open:
            anchor = datetime(
                ts.year, ts.month, ts.day,
                self.market_open_hour, self.market_open_minute, 0,
                tzinfo=ts.tzinfo,
            )
            if ts < anchor:
                anchor = anchor - timedelta(days=1)
            delta = ts - anchor
            period_seconds = period * 60
            if delta.total_seconds() % period_seconds == 0 and ts.second == 0:
                return ts
            steps = int(delta.total_seconds() // period_seconds) + 1
            return anchor + timedelta(seconds=steps * period_seconds)

        # Wall-clock alignment (no market-open anchor)
        minute_of_day = ts.hour * 60 + ts.minute
        offset = minute_of_day % period
        if offset == 0 and ts.second == 0:
            end_minute = minute_of_day
        else:
            end_minute = minute_of_day + (period - offset)

        end_date = ts.date()
        if end_minute >= 1440:
            end_minute -= 1440
            end_date = end_date + timedelta(days=1)

        return datetime(
            end_date.year, end_date.month, end_date.day,
            end_minute // 60, end_minute % 60, 0,
            tzinfo=ts.tzinfo,
        )

    def _make_agg_bar(self, pd: PriceData, window_end: datetime) -> PriceData:
        return PriceData(
            symbol=pd.symbol, timestamp=window_end,
            open=pd.open, high=pd.high, low=pd.low, close=pd.close,
            volume=pd.volume, trade_count=pd.trade_count,
            vwap=pd.vwap, exchange=pd.exchange,
        )

    def _update_agg_bar(self, period: int, symbol: str, pd: PriceData) -> None:
        agg = self._mtf_agg[period][symbol]
        agg.high = max(agg.high, pd.high)
        agg.low = min(agg.low, pd.low)
        agg.close = pd.close
        agg.volume += pd.volume

    def _flush_agg(self, period: int, symbol: str) -> PriceData | None:
        agg = self._mtf_agg[period].pop(symbol, None)
        self._mtf_window_end[period].pop(symbol, None)
        return agg
