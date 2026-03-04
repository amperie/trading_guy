"""
Tests for TickAggregationPassthroughEngine and its integration with
BacktestingEngine (backtest path) and AlpacaRealTimeEngine (live path).

Coverage:
  - Window boundary alignment (_get_window_end)
  - OHLCV accumulation within a window
  - Downstream call timing (when to flush / when to hold)
  - Multi-symbol independent aggregation
  - batch_symbols mode
  - BacktestingEngine integration (algo sees aggregated bars, not raw 1-min)
  - AlpacaRealTimeEngine routing (_agg_engine None vs set, _run_pipeline shim)
"""
import types
from datetime import datetime
from unittest.mock import MagicMock, call

from trading.core.classes import PriceData, TickResults
from trading.engines.tick_aggregation_passthrough_engine import TickAggregationPassthroughEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pd(symbol, hour, minute, *, open_=100.0, high=101.0, low=99.0, close=100.0, volume=1000):
    """Create a timezone-naive PriceData at 2024-01-02 HH:MM:00."""
    return PriceData(
        symbol=symbol,
        timestamp=datetime(2024, 1, 2, hour, minute, 0),
        open=open_, high=high, low=low, close=close, volume=volume,
    )


def _make_agg(period=5, *, batch_symbols=False, expected_symbols=None):
    """Return (engine, mock_downstream). Downstream records all calls."""
    downstream = MagicMock()
    downstream.on_tick.return_value = TickResults(orders=[])
    cfg = {
        "downstream_engine": downstream,
        "aggregation_period_minutes": period,
        "use_market_open": True,
        "market_open_hour": 9,
        "market_open_minute": 30,
        "batch_symbols": batch_symbols,
        "expected_symbols": expected_symbols or [],
    }
    return TickAggregationPassthroughEngine(cfg=cfg), downstream


# ---------------------------------------------------------------------------
# 1. Window-end alignment
# ---------------------------------------------------------------------------

class TestWindowEndAlignment:
    """_get_window_end computes the correct boundary for each timestamp."""

    def _eng(self, period=5):
        eng, _ = _make_agg(period)
        return eng

    def test_market_open_tick_is_its_own_boundary(self):
        """9:30:00 is exactly on a 5-min boundary — window_end == ts."""
        ts = datetime(2024, 1, 2, 9, 30, 0)
        assert self._eng(5)._get_window_end(ts) == ts

    def test_9_35_is_its_own_boundary(self):
        ts = datetime(2024, 1, 2, 9, 35, 0)
        assert self._eng(5)._get_window_end(ts) == ts

    def test_first_minute_after_open_ends_at_9_35(self):
        ts = datetime(2024, 1, 2, 9, 31, 0)
        assert self._eng(5)._get_window_end(ts) == datetime(2024, 1, 2, 9, 35, 0)

    def test_mid_window_ends_at_same_boundary(self):
        eng = self._eng(5)
        for minute in [31, 32, 33, 34]:
            ts = datetime(2024, 1, 2, 9, minute, 0)
            assert eng._get_window_end(ts) == datetime(2024, 1, 2, 9, 35, 0)

    def test_tick_after_boundary_starts_next_window(self):
        ts = datetime(2024, 1, 2, 9, 36, 0)
        assert self._eng(5)._get_window_end(ts) == datetime(2024, 1, 2, 9, 40, 0)

    def test_1min_period_every_tick_is_own_boundary(self):
        eng = self._eng(1)
        for minute in [30, 31, 32, 33, 34]:
            ts = datetime(2024, 1, 2, 9, minute, 0)
            assert eng._get_window_end(ts) == ts

    def test_10min_period_alignment(self):
        # 9:30 anchor → windows end at 9:40, 9:50, 10:00, …
        eng = self._eng(10)
        for minute in [31, 32, 33, 34, 35, 36, 37, 38, 39]:
            ts = datetime(2024, 1, 2, 9, minute, 0)
            assert eng._get_window_end(ts) == datetime(2024, 1, 2, 9, 40, 0)

    def test_10min_boundary_tick(self):
        ts = datetime(2024, 1, 2, 9, 40, 0)
        assert self._eng(10)._get_window_end(ts) == ts

    def test_3min_period(self):
        # 9:30 → 9:33; 9:31 → 9:33; 9:33 → 9:33; 9:34 → 9:36
        eng = self._eng(3)
        assert eng._get_window_end(datetime(2024, 1, 2, 9, 30)) == datetime(2024, 1, 2, 9, 30)
        assert eng._get_window_end(datetime(2024, 1, 2, 9, 31)) == datetime(2024, 1, 2, 9, 33)
        assert eng._get_window_end(datetime(2024, 1, 2, 9, 33)) == datetime(2024, 1, 2, 9, 33)
        assert eng._get_window_end(datetime(2024, 1, 2, 9, 34)) == datetime(2024, 1, 2, 9, 36)


# ---------------------------------------------------------------------------
# 2. OHLCV accumulation
# ---------------------------------------------------------------------------

class TestOHLCVAggregation:
    """Aggregated bar fields are computed correctly from raw ticks."""

    def _flush(self, ticks_per_field, boundary_tick):
        """Feed ticks then boundary tick, return the aggregated PriceData."""
        eng, ds = _make_agg(5)
        for t in ticks_per_field:
            eng.on_tick([t])
        eng.on_tick([boundary_tick])
        return ds.on_tick.call_args[0][0][0]

    def test_open_is_from_first_tick(self):
        agg = self._flush(
            [_pd("SPY", 9, 31, open_=111.0), _pd("SPY", 9, 32, open_=222.0)],
            _pd("SPY", 9, 35, open_=333.0),
        )
        assert agg.open == 111.0

    def test_high_is_maximum(self):
        agg = self._flush(
            [_pd("SPY", 9, 31, high=102.0), _pd("SPY", 9, 32, high=108.0), _pd("SPY", 9, 33, high=104.0)],
            _pd("SPY", 9, 35, high=103.0),
        )
        assert agg.high == 108.0

    def test_low_is_minimum(self):
        agg = self._flush(
            [_pd("SPY", 9, 31, low=99.0), _pd("SPY", 9, 32, low=95.0), _pd("SPY", 9, 33, low=97.0)],
            _pd("SPY", 9, 35, low=98.0),
        )
        assert agg.low == 95.0

    def test_close_is_from_boundary_tick(self):
        agg = self._flush(
            [_pd("SPY", 9, 31, close=100.0), _pd("SPY", 9, 32, close=105.0)],
            _pd("SPY", 9, 35, close=999.0),
        )
        assert agg.close == 999.0

    def test_volume_accumulates(self):
        agg = self._flush(
            [_pd("SPY", 9, 31, volume=1000), _pd("SPY", 9, 32, volume=2000)],
            _pd("SPY", 9, 35, volume=500),
        )
        assert agg.volume == 3500

    def test_timestamp_is_window_end_not_tick_time(self):
        agg = self._flush(
            [_pd("SPY", 9, 31)],
            _pd("SPY", 9, 35),
        )
        assert agg.timestamp == datetime(2024, 1, 2, 9, 35, 0)

    def test_symbol_is_preserved(self):
        agg = self._flush(
            [_pd("UPRO", 9, 31)],
            _pd("UPRO", 9, 35),
        )
        assert agg.symbol == "UPRO"

    def test_single_tick_at_boundary_passes_through_unchanged(self):
        """A lone tick that lands exactly on a boundary forms a 1-tick bar."""
        eng, ds = _make_agg(5)
        eng.on_tick([_pd("SPY", 9, 30, open_=50.0, high=52.0, low=48.0, close=51.0, volume=777)])
        agg = ds.on_tick.call_args[0][0][0]
        assert agg.open == 50.0
        assert agg.high == 52.0
        assert agg.low == 48.0
        assert agg.close == 51.0
        assert agg.volume == 777
        assert agg.timestamp == datetime(2024, 1, 2, 9, 30, 0)


# ---------------------------------------------------------------------------
# 3. Downstream call timing
# ---------------------------------------------------------------------------

class TestDownstreamCallTiming:
    """Downstream engine.on_tick() called at the right times."""

    def test_no_call_for_mid_window_ticks(self):
        eng, ds = _make_agg(5)
        for m in [31, 32, 33, 34]:
            eng.on_tick([_pd("SPY", 9, m)])
        ds.on_tick.assert_not_called()

    def test_called_once_at_window_boundary(self):
        eng, ds = _make_agg(5)
        for m in [31, 32, 33, 34, 35]:
            eng.on_tick([_pd("SPY", 9, m)])
        ds.on_tick.assert_called_once()

    def test_called_once_per_window_over_multiple_windows(self):
        eng, ds = _make_agg(5)
        for m in range(31, 51):   # 9:31–9:50 = 4 complete windows (…35, …40, …45, …50)
            eng.on_tick([_pd("SPY", 9, m)])
        assert ds.on_tick.call_count == 4

    def test_boundary_only_tick_flushes_immediately(self):
        """9:30 is its own window; downstream called right away."""
        eng, ds = _make_agg(5)
        eng.on_tick([_pd("SPY", 9, 30)])
        ds.on_tick.assert_called_once()

    def test_skipped_window_flushes_on_next_arrival(self):
        """Jump from 9:31 to 9:36 (past window end): old window flushed immediately."""
        eng, ds = _make_agg(5)
        eng.on_tick([_pd("SPY", 9, 31, close=50.0)])
        eng.on_tick([_pd("SPY", 9, 36)])          # crosses the 9:35 boundary
        ds.on_tick.assert_called_once()
        agg = ds.on_tick.call_args[0][0][0]
        # Flushed bar is from the OLD window (ends 9:35), not the new tick
        assert agg.timestamp == datetime(2024, 1, 2, 9, 35, 0)
        assert agg.close == 50.0                  # only tick in old window

    def test_1min_period_downstream_called_every_tick(self):
        eng, ds = _make_agg(1)
        for m in [30, 31, 32, 33, 34]:
            eng.on_tick([_pd("SPY", 9, m)])
        assert ds.on_tick.call_count == 5

    def test_downstream_receives_list_of_pricedata(self):
        eng, ds = _make_agg(5)
        eng.on_tick([_pd("SPY", 9, 35)])
        args = ds.on_tick.call_args[0][0]
        assert isinstance(args, list)
        assert isinstance(args[0], PriceData)


# ---------------------------------------------------------------------------
# 4. Multi-symbol
# ---------------------------------------------------------------------------

class TestMultiSymbol:
    """Each symbol aggregates independently; batch_symbols waits for all."""

    def test_two_symbols_aggregate_independently(self):
        eng, ds = _make_agg(5)
        # Feed mixed-symbol ticks across the window
        eng.on_tick([_pd("SPY", 9, 31, close=100.0), _pd("UPRO", 9, 31, close=50.0)])
        eng.on_tick([_pd("SPY", 9, 35, close=102.0), _pd("UPRO", 9, 35, close=52.0)])
        agg_list = ds.on_tick.call_args[0][0]
        by_symbol = {pd.symbol: pd for pd in agg_list}
        assert "SPY" in by_symbol and "UPRO" in by_symbol
        assert by_symbol["SPY"].close == 102.0
        assert by_symbol["UPRO"].close == 52.0

    def test_symbol_windows_align_independently_when_ticks_staggered(self):
        """SPY arrives mid-window; UPRO hasn't arrived yet for that window."""
        eng, ds = _make_agg(5)
        eng.on_tick([_pd("SPY", 9, 31)])
        eng.on_tick([_pd("SPY", 9, 35)])    # SPY window flushes
        # Only one symbol in the flush
        flushed = ds.on_tick.call_args[0][0]
        assert all(pd.symbol == "SPY" for pd in flushed)

    def test_batch_symbols_delays_until_all_ready(self):
        """batch_symbols=True: downstream not called until every expected symbol completes."""
        eng, ds = _make_agg(5, batch_symbols=True, expected_symbols=["SPY", "UPRO"])
        eng.on_tick([_pd("SPY", 9, 35)])    # SPY ready, UPRO not yet
        ds.on_tick.assert_not_called()
        eng.on_tick([_pd("UPRO", 9, 35)])   # now both ready → flush
        ds.on_tick.assert_called_once()

    def test_batch_symbols_flush_contains_all_symbols(self):
        eng, ds = _make_agg(5, batch_symbols=True, expected_symbols=["SPY", "UPRO"])
        eng.on_tick([_pd("SPY", 9, 31), _pd("UPRO", 9, 31)])
        eng.on_tick([_pd("SPY", 9, 35)])
        eng.on_tick([_pd("UPRO", 9, 35)])
        flushed = ds.on_tick.call_args[0][0]
        symbols = {pd.symbol for pd in flushed}
        assert symbols == {"SPY", "UPRO"}

    def test_without_batch_symbols_flushes_per_symbol(self):
        """batch_symbols=False (default): each symbol flushes as soon as its own window closes."""
        eng, ds = _make_agg(5, batch_symbols=False)
        eng.on_tick([_pd("SPY", 9, 35)])    # SPY window done
        eng.on_tick([_pd("UPRO", 9, 35)])   # UPRO window done (separate call)
        assert ds.on_tick.call_count == 2


# ---------------------------------------------------------------------------
# 5. BacktestingEngine integration
# ---------------------------------------------------------------------------

class TestBacktestEngineIntegration:
    """BacktestingEngine used as downstream: algo sees aggregated bars, not raw ticks."""

    def _make_tracking_algo(self):
        """Minimal algorithm that records every on_data call."""
        from trading.core.algorithm import Algorithm

        class TrackingAlgo(Algorithm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.calls = 0
                self.received = []   # list[list[PriceData]] per call

            def on_data_logic(self, data):
                self.calls += 1
                self.received.append(list(data))
                return []

        return TrackingAlgo(cfg={}, history_length=0)

    def _make_stack(self, period):
        """Return (agg_engine, tracking_algo) with fresh pf/om wired together."""
        from trading.core.om.backtesting_om import BacktestingOrderManager
        from trading.core.pf.single_symbol_portfolio import SingleSymbolPortfolio
        from trading.engines.backtest_engine import BacktestingEngine

        al = self._make_tracking_algo()
        om = BacktestingOrderManager()
        pf = SingleSymbolPortfolio(
            {"symbol": "SPY", "stop_pct": 5.0, "profit_pct": 10.0},
            om, 10_000.0, {}, False,
        )
        bt = BacktestingEngine(cfg={"state_store": {"enabled": False}}, dp=None, al=al, om=om, pf=pf)

        agg_cfg = {
            "downstream_engine": bt,
            "aggregation_period_minutes": period,
            "use_market_open": True,
            "market_open_hour": 9,
            "market_open_minute": 30,
        }
        agg = TickAggregationPassthroughEngine(cfg=agg_cfg)
        return agg, al

    def test_algo_called_once_per_completed_window(self):
        """10 one-minute ticks → 2 five-minute windows → algo called twice."""
        agg, al = self._make_stack(5)
        for m in range(31, 41):            # 9:31–9:40 inclusive
            agg.on_tick([_pd("SPY", 9, m)])
        assert al.calls == 2

    def test_algo_sees_aggregated_close_not_raw(self):
        """The close in the aggregated bar equals the last raw tick's close."""
        agg, al = self._make_stack(5)
        for m in [31, 32, 33, 34]:
            agg.on_tick([_pd("SPY", 9, m, close=100.0 + m)])
        agg.on_tick([_pd("SPY", 9, 35, close=999.0)])   # boundary tick
        assert al.calls == 1
        assert al.received[0][0].close == 999.0

    def test_algo_sees_correct_high_and_low(self):
        agg, al = self._make_stack(5)
        highs = [102.0, 108.0, 104.0, 103.0, 105.0]
        lows  = [ 99.0,  95.0,  97.0,  98.0,  96.0]
        minutes = [31, 32, 33, 34, 35]
        for m, h, l in zip(minutes, highs, lows):
            agg.on_tick([_pd("SPY", 9, m, high=h, low=l)])
        assert al.received[0][0].high == 108.0
        assert al.received[0][0].low  == 95.0

    def test_algo_sees_accumulated_volume(self):
        agg, al = self._make_stack(5)
        volumes = [1000, 2000, 500, 300, 700]
        for m, v in zip([31, 32, 33, 34, 35], volumes):
            agg.on_tick([_pd("SPY", 9, m, volume=v)])
        assert al.received[0][0].volume == sum(volumes)

    def test_algo_not_called_between_windows(self):
        """Ticks mid-window must not trigger the algo at all."""
        agg, al = self._make_stack(5)
        for m in [31, 32, 33, 34]:   # window not yet closed
            agg.on_tick([_pd("SPY", 9, m)])
        assert al.calls == 0

    def test_1min_aggregation_is_passthrough(self):
        """Period=1 means every tick reaches the algo as-is."""
        agg, al = self._make_stack(1)
        for m in [30, 31, 32]:
            agg.on_tick([_pd("SPY", 9, m)])
        assert al.calls == 3


# ---------------------------------------------------------------------------
# 6. AlpacaRealTimeEngine routing
# ---------------------------------------------------------------------------

class TestAlpacaEngineRouting:
    """AlpacaRealTimeEngine routes through _agg_engine when set."""

    def _make_engine(self):
        """Return a live engine with mocked al / pf / om (no real credentials needed)."""
        from trading.engines.alpaca_engine import AlpacaRealTimeEngine

        al = MagicMock()
        al.on_data.return_value = []

        pf = MagicMock()
        pf.process_market_signals_for_tick.return_value = TickResults(orders=[])

        om = MagicMock()

        cfg = {
            "api_key": "fake-key",
            "secret_key": "fake-secret",
            "symbols_to_subscribe": ["SPY"],
            "state_store": {"enabled": False},
        }
        engine = AlpacaRealTimeEngine(cfg=cfg, al=al, om=om, pf=pf)
        return engine, al, pf

    def test_no_agg_engine_on_tick_calls_run_pipeline(self):
        """Without _agg_engine, on_tick() runs the standard pipeline."""
        engine, al, pf = self._make_engine()
        tick = [_pd("SPY", 9, 31)]
        engine.on_tick(tick)
        al.on_data.assert_called_once_with(tick)
        pf.process_market_signals_for_tick.assert_called_once()

    def test_with_agg_engine_on_tick_routes_to_agg(self):
        """When _agg_engine is set, on_tick() delegates to it — algo NOT called directly."""
        engine, al, pf = self._make_engine()
        mock_agg = MagicMock()
        mock_agg.on_tick.return_value = TickResults(orders=[])
        engine._agg_engine = mock_agg

        tick = [_pd("SPY", 9, 31)]
        engine.on_tick(tick)

        mock_agg.on_tick.assert_called_once_with(tick)
        al.on_data.assert_not_called()

    def test_run_pipeline_calls_algo_then_portfolio(self):
        engine, al, pf = self._make_engine()
        tick = [_pd("SPY", 9, 31)]
        engine._run_pipeline(tick)
        al.on_data.assert_called_once_with(tick)
        pf.process_market_signals_for_tick.assert_called_once()

    def test_shim_routes_agg_output_to_run_pipeline(self):
        """SimpleNamespace shim (as wired in cmd_live) delivers agg output to _run_pipeline."""
        engine, al, pf = self._make_engine()

        shim = types.SimpleNamespace()
        shim.on_tick = engine._run_pipeline

        # Shim used as downstream of a real agg engine
        ds_calls = []
        original_run_pipeline = engine._run_pipeline

        def tracking_pipeline(tick):
            ds_calls.append(tick)
            return original_run_pipeline(tick)

        shim.on_tick = tracking_pipeline

        # Wire a real 1-min agg engine so we can drive it with raw ticks
        agg_cfg = {
            "downstream_engine": shim,
            "aggregation_period_minutes": 1,
            "use_market_open": True,
            "market_open_hour": 9,
            "market_open_minute": 30,
        }
        engine._agg_engine = TickAggregationPassthroughEngine(cfg=agg_cfg)

        tick = [_pd("SPY", 9, 30)]          # 1-min boundary → immediate flush
        engine.on_tick(tick)

        # Pipeline was called (via shim), and algo received the tick
        assert len(ds_calls) == 1
        al.on_data.assert_called_once()

    def test_agg_engine_replaced_without_modifying_run_pipeline(self):
        """Swapping _agg_engine does not affect _run_pipeline."""
        engine, al, pf = self._make_engine()

        agg1 = MagicMock()
        agg1.on_tick.return_value = TickResults(orders=[])
        agg2 = MagicMock()
        agg2.on_tick.return_value = TickResults(orders=[])

        tick = [_pd("SPY", 9, 31)]

        engine._agg_engine = agg1
        engine.on_tick(tick)
        engine._agg_engine = agg2
        engine.on_tick(tick)

        agg1.on_tick.assert_called_once()
        agg2.on_tick.assert_called_once()
        al.on_data.assert_not_called()  # _run_pipeline bypassed both times
