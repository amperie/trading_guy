from datetime import datetime, timedelta

from trading.engines.base_engine import BaseEngine
from trading.core.classes import PriceData, TickResults


class TickAggregationPassthroughEngine(BaseEngine):
    """
    This engine can be called by a data receiving engine
    It aggregates the ticks according to the settings and
    passes an aggregated tick to the downstream engine
    ie: if set to aggregate ticks into 5 minute ticks, it will
    run five times to aggregate the incoming ticks and when the aggregated
    5 minute tick is ready, it will call the downstream engine with it.
    It will not call the downstream engine without fully aggregated ticks

    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.downstream_engine = cfg['downstream_engine']
        # Aggregation period in minutes
        self.aggregation_period_minutes = cfg.get('aggregation_period_minutes', 5)
        # Minute when the aggregation starts (ie: 1 -> agg starts at 9:01
        self.aggregation_start_minutes = cfg.get('aggregation_start_minutes', None)
        self.use_market_open = cfg.get('use_market_open', True)
        self.market_open_hour = cfg.get('market_open_hour', 9)
        self.market_open_minute = cfg.get('market_open_minute', 30)
        self._window_end_by_symbol: dict[str, datetime] = {}
        self._agg_by_symbol: dict[str, PriceData] = {}

    def finalize(self):
        pass

    def run(self):
        for tick in self.dp.iterate():
            self.on_tick(tick)

    def on_tick(self, tick: list[PriceData]) -> TickResults:
        aggregated: list[PriceData] = []
        for pd in tick:
            symbol = pd.symbol
            window_end = self._get_window_end(pd.timestamp)

            if symbol not in self._agg_by_symbol:
                self._agg_by_symbol[symbol] = self._init_agg(pd, window_end)
                self._window_end_by_symbol[symbol] = window_end
                if pd.timestamp == window_end:
                    aggregated.append(self._flush_symbol(symbol, window_end))
                continue

            current_end = self._window_end_by_symbol[symbol]
            if pd.timestamp < current_end:
                self._update_agg(symbol, pd)
            elif pd.timestamp == current_end:
                self._update_agg(symbol, pd)
                aggregated.append(self._flush_symbol(symbol, current_end))
            else:
                aggregated.append(self._flush_symbol(symbol, current_end))
                self._agg_by_symbol[symbol] = self._init_agg(pd, window_end)
                self._window_end_by_symbol[symbol] = window_end
                if pd.timestamp == window_end:
                    aggregated.append(self._flush_symbol(symbol, window_end))

        aggregated = [a for a in aggregated if a is not None]
        if aggregated:
            return self.downstream_engine.on_tick(aggregated)
        return TickResults(orders=[])

    def _get_window_end(self, ts: datetime) -> datetime:
        if self.use_market_open:
            anchor = datetime(
                ts.year, ts.month, ts.day,
                self.market_open_hour, self.market_open_minute, 0,
                tzinfo=ts.tzinfo
            )
            if ts < anchor:
                anchor = anchor - timedelta(days=1)
            delta = ts - anchor
            period_seconds = int(self.aggregation_period_minutes) * 60
            if delta.total_seconds() % period_seconds == 0 and ts.second == 0:
                return ts
            steps = int(delta.total_seconds() // period_seconds) + 1
            return anchor + timedelta(seconds=steps * period_seconds)

        start_minute = int(self.aggregation_start_minutes or 0)
        period = int(self.aggregation_period_minutes)

        minute_of_day = ts.hour * 60 + ts.minute
        offset = (minute_of_day - start_minute) % period
        if offset == 0 and ts.second == 0:
            end_minute_of_day = minute_of_day
        else:
            end_minute_of_day = minute_of_day + (period - offset)

        end_date = ts.date()
        if end_minute_of_day >= 1440:
            end_minute_of_day -= 1440
            end_date = end_date + timedelta(days=1)

        end_dt = datetime(
            end_date.year, end_date.month, end_date.day,
            end_minute_of_day // 60, end_minute_of_day % 60,
            0, tzinfo=ts.tzinfo
        )
        return end_dt

    def _init_agg(self, pd: PriceData, window_end: datetime) -> PriceData:
        return PriceData(
            symbol=pd.symbol,
            timestamp=window_end,
            open=pd.open,
            high=pd.high,
            low=pd.low,
            close=pd.close,
            volume=pd.volume,
            trade_count=pd.trade_count,
            vwap=pd.vwap,
            exchange=pd.exchange,
        )

    def _update_agg(self, symbol: str, pd: PriceData):
        agg = self._agg_by_symbol[symbol]
        agg.high = max(agg.high, pd.high)
        agg.low = min(agg.low, pd.low)
        agg.close = pd.close
        agg.volume += pd.volume

    def _flush_symbol(self, symbol: str, window_end: datetime) -> PriceData | None:
        agg = self._agg_by_symbol.get(symbol)
        if agg is None:
            return None
        if self._window_end_by_symbol.get(symbol) != window_end:
            return None
        self._agg_by_symbol.pop(symbol, None)
        self._window_end_by_symbol.pop(symbol, None)
        return agg
