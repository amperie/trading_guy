"""
Unit tests for session replay infrastructure.

Covers:
    - compute_warmup_start_date() — warmup lookback calculation
    - SessionReplayDataProvider._resolve_mongo_params() — MongoDB param fallback
    - PortfolioAnalyzer._contribute_to_run() — in-run metric/artifact logging
    - SessionReplayDataProvider.load_data() — end-to-end with mocked MongoDB
      and a stubbed AlpacaDataProvider (requires mongomock)
"""
import math
import os
import types
import tempfile
import dataclasses
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import pandas as pd

from utils.utils import compute_warmup_start_date


# ---------------------------------------------------------------------------
# compute_warmup_start_date
# ---------------------------------------------------------------------------

class TestComputeWarmupStartDate:

    def _days_back(self, warmup_bars, timeframe, ref=None):
        ref = ref or datetime(2026, 3, 8, 9, 30)
        result = compute_warmup_start_date(warmup_bars, timeframe, ref)
        return (ref - result).days

    def test_minute_single_day(self):
        # 390 bars = 1 trading day; expect 1 + padding ≈ a few calendar days
        days = self._days_back(390, "Minute")
        assert days >= 2, "1 trading day of minutes should look back at least 2 calendar days"

    def test_minute_many_bars(self):
        # 3900 bars = 10 trading days; should look back ≥ 14 calendar days
        days = self._days_back(3900, "Minute")
        assert days >= 14

    def test_day_bars(self):
        # 20 day bars; should look back ≥ 20 calendar days (trading day × 7/5 ratio)
        days = self._days_back(20, "Day")
        assert days >= 20

    def test_hour_bars(self):
        # 13 bars = 2 trading days; should look back ≥ 3 calendar days
        days = self._days_back(13, "Hour")
        assert days >= 3

    def test_week_bars(self):
        # 4 weeks ≈ 20 trading days; should look back ≥ 25 calendar days
        days = self._days_back(4, "Week")
        assert days >= 25

    def test_month_bars(self):
        # 3 months ≈ 66 trading days; should look back substantially
        days = self._days_back(3, "Month")
        assert days >= 50

    def test_unknown_timeframe_falls_back(self):
        # Unknown timeframe: trading_days = warmup_bars (fallback)
        ref = datetime(2026, 3, 8)
        result = compute_warmup_start_date(10, "Tick", ref)
        assert result < ref

    def test_zero_bars_returns_reference(self):
        # 0 warmup bars → at most a few days buffer (due to calendar_days = int(0*7/5)+5)
        ref = datetime(2026, 3, 8)
        result = compute_warmup_start_date(0, "Minute", ref)
        assert result <= ref

    def test_result_is_before_reference(self):
        ref = datetime(2026, 3, 8)
        result = compute_warmup_start_date(100, "Minute", ref)
        assert result < ref

    def test_more_bars_means_earlier_date(self):
        ref = datetime(2026, 3, 8)
        small = compute_warmup_start_date(50, "Minute", ref)
        large = compute_warmup_start_date(500, "Minute", ref)
        assert large < small, "more bars should require an earlier start date"


# ---------------------------------------------------------------------------
# SessionReplayDataProvider._resolve_mongo_params
# ---------------------------------------------------------------------------

class TestResolveMongoParams:
    """Test MongoDB param resolution without actual database connections."""

    def test_explicit_params_take_priority(self):
        from trading.data_providers.session_replay_data_provider import SessionReplayDataProvider
        uri, db = SessionReplayDataProvider._resolve_mongo_params(
            "mongodb://custom:27017", "mydb"
        )
        assert uri == "mongodb://custom:27017"
        assert db == "mydb"

    def test_falls_back_to_defaults_when_none(self):
        from trading.data_providers.session_replay_data_provider import SessionReplayDataProvider
        # ConfigManager may not find state_store in test environment, so falls back
        uri, db = SessionReplayDataProvider._resolve_mongo_params(None, None)
        # Should return some non-empty string
        assert uri
        assert db

    def test_partial_override_uri_only(self):
        from trading.data_providers.session_replay_data_provider import SessionReplayDataProvider
        uri, db = SessionReplayDataProvider._resolve_mongo_params(
            "mongodb://override:27017", None
        )
        assert uri == "mongodb://override:27017"
        assert db  # db falls back to config/default


# ---------------------------------------------------------------------------
# PortfolioAnalyzer._contribute_to_run
# ---------------------------------------------------------------------------

class TestContributeToRun:
    """Test _contribute_to_run() with a mock MLflow client."""

    def _make_analyzer(self, n_ticks=200, initial=10000.0, final=11000.0):
        """Build a minimal PortfolioAnalyzer with enough history to compute metrics.

        Uses day-spaced timestamps (not minutes) so annualized-return maths don't
        overflow when the test window is very short.
        """
        from trading.analysis.portfolio_analyzer import PortfolioAnalyzer

        base = datetime(2026, 1, 2, 9, 30)
        value_history = {}
        cash_history = {}
        tick_history = {}

        step = (final - initial) / n_ticks
        for i in range(n_ticks):
            ts = base + timedelta(days=i)   # one tick per day → sensible annualisation
            value_history[ts] = initial + step * i
            cash_history[ts] = initial + step * i
            tick_history[ts] = []

        pf = types.SimpleNamespace(
            keep_history=True,
            tick_history=tick_history,
            value_history=value_history,
            cash_history=cash_history,
            signals_history={},
            total_value=final,
            cash=final,
            positions={},
            om=types.SimpleNamespace(
                filled_orders_by_id={},
                pending_orders_by_id={},
            ),
        )
        pf.om = pf.om
        return PortfolioAnalyzer(pf)

    def test_metrics_logged_with_no_prefix(self):
        analyzer = self._make_analyzer()
        client = MagicMock()

        analyzer._contribute_to_run(client, metric_prefix="", artifact_prefix="",
                                    log_charts=False, log_trades=False,
                                    log_signals=False, log_report=False)

        client.log_metrics.assert_called_once()
        metrics_dict = client.log_metrics.call_args[0][0]
        assert "total_return" in metrics_dict
        assert isinstance(metrics_dict["total_return"], float)

    def test_metrics_logged_with_prefix(self):
        analyzer = self._make_analyzer()
        client = MagicMock()

        analyzer._contribute_to_run(client, metric_prefix="live_", artifact_prefix="live_",
                                    log_charts=False, log_trades=False,
                                    log_signals=False, log_report=False)

        metrics_dict = client.log_metrics.call_args[0][0]
        # All keys should be prefixed
        assert all(k.startswith("live_") for k in metrics_dict)
        assert "live_total_return" in metrics_dict

    def test_nan_metrics_excluded(self):
        """NaN / Inf metrics must not be passed to MLflow."""
        analyzer = self._make_analyzer(n_ticks=1)  # minimal data → many NaN metrics
        client = MagicMock()

        analyzer._contribute_to_run(client, log_charts=False, log_trades=False,
                                    log_signals=False, log_report=False)

        metrics_dict = client.log_metrics.call_args[0][0]
        for v in metrics_dict.values():
            assert not math.isnan(v), f"NaN slipped through: {v}"
            assert not math.isinf(v), f"Inf slipped through: {v}"

    def test_report_logged_when_enabled(self):
        analyzer = self._make_analyzer()
        client = MagicMock()

        analyzer._contribute_to_run(client, artifact_prefix="replay_",
                                    log_charts=False, log_trades=False,
                                    log_signals=False, log_report=True)

        client.log_text.assert_called_once()
        fname = client.log_text.call_args[0][1]
        assert fname == "replay_performance_report.txt"

    def test_no_crash_when_chart_fails(self):
        """Chart failures should be swallowed with a warning, not raised."""
        analyzer = self._make_analyzer()
        client = MagicMock()

        with patch.object(analyzer, "save_equity_curve", side_effect=RuntimeError("no data")):
            # Should not raise
            analyzer._contribute_to_run(client, log_charts=True, log_trades=False,
                                        log_signals=False, log_report=False)

    def test_log_charts_false_skips_artifacts(self):
        analyzer = self._make_analyzer()
        client = MagicMock()

        analyzer._contribute_to_run(client, log_charts=False, log_trades=False,
                                    log_signals=False, log_report=False)

        client.log_artifact.assert_not_called()

    def test_artifact_prefix_applied_to_files(self):
        """Artifact filenames should start with the given prefix."""
        analyzer = self._make_analyzer()
        client = MagicMock()

        with patch.object(analyzer, "save_trades_csv"):
            analyzer._contribute_to_run(client, artifact_prefix="live_",
                                        log_charts=False, log_trades=True,
                                        log_signals=False, log_report=False)

        logged_paths = [call[0][0] for call in client.log_artifact.call_args_list]
        for p in logged_paths:
            assert os.path.basename(p).startswith("live_"), \
                f"expected 'live_' prefix, got '{os.path.basename(p)}'"


# ---------------------------------------------------------------------------
# SessionReplayDataProvider.load_data (with mongomock + stubbed Alpaca)
# ---------------------------------------------------------------------------

try:
    import mongomock
    HAS_MONGOMOCK = True
except ImportError:
    HAS_MONGOMOCK = False


@pytest.mark.skipif(not HAS_MONGOMOCK, reason="mongomock not installed")
class TestSessionReplayDataProviderLoadData:
    """Full load_data() path using an in-memory MongoDB and a fake AlpacaDataProvider."""

    def _make_store(self):
        from utils.trading_state_store import TradingStateStore
        client = mongomock.MongoClient()
        return TradingStateStore(client=client)

    def _make_session(self, store, session_id, symbols, timeframe="Minute", warmup_bars=5):
        """Create a session with metadata and fake tick data."""
        from trading.core.classes import PriceData

        store.create_session(
            session_id=session_id,
            name="test",
            metadata={
                "symbols": symbols,
                "timeframe": timeframe,
                "warmup_bars": warmup_bars,
                "algorithm_class": "trading.algorithms.test_algorithm.TestAlgorithm",
                "algorithm_config": {"history_length": warmup_bars},
                "portfolio_class": "trading.core.pf.single_symbol_portfolio.SingleSymbolPortfolio",
                "portfolio_config": {"symbol": symbols[0], "cash": 10000},
            },
        )

        # Insert fake ticks to give the provider timestamps to infer session range
        base = datetime(2026, 3, 1, 9, 30)
        for minute in range(3):
            ts = base + timedelta(minutes=minute)
            tick = [PriceData(
                symbol=sym,
                timestamp=ts,
                open=100.0, high=101.0, low=99.0, close=100.5,
                volume=1000,
            ) for sym in symbols]
            store.save_tick(session_id, ts, tick, cash=10000.0, total_value=10000.0)

        return base, base + timedelta(minutes=2)  # session_start, session_end

    def test_load_data_sets_session_metadata(self):
        """load_data() populates _session_metadata and leaves self.data empty."""
        store = self._make_store()
        session_id = "replay-test-1"
        self._make_session(store, session_id, ["SPY"])

        from trading.data_providers.session_replay_data_provider import SessionReplayDataProvider

        dp = SessionReplayDataProvider(cfg={
            "session_id": session_id,
            "api_key": "test-key",
            "secret_key": "test-secret",
        })

        with patch("utils.trading_state_store.TradingStateStore", return_value=store):
            dp.load_data()

        assert dp._session_metadata
        assert dp._session_metadata["symbols"] == ["SPY"]
        assert "session_start" in dp._session_metadata
        assert "session_end" in dp._session_metadata
        # self.data is intentionally empty — callers fetch bars separately
        assert dp.data is not None
        assert len(dp.data) == 0

    def test_session_metadata_exposed(self):
        """_session_metadata is populated after load_data()."""
        store = self._make_store()
        session_id = "replay-test-2"
        self._make_session(store, session_id, ["SPY", "UPRO"], warmup_bars=10)

        from trading.data_providers.session_replay_data_provider import SessionReplayDataProvider

        dp = SessionReplayDataProvider(cfg={
            "session_id": session_id,
            "api_key": "k", "secret_key": "s",
        })

        with patch("utils.trading_state_store.TradingStateStore", return_value=store):
            dp.load_data()

        meta = dp._session_metadata
        assert set(["SPY", "UPRO"]) == set(meta["symbols"])
        assert meta["timeframe"] == "Minute"
        assert meta["warmup_bars"] == 10
        assert "session_start" in meta
        assert "session_end" in meta

    def test_missing_session_raises(self):
        """Requesting a non-existent session_id raises ValueError."""
        store = self._make_store()

        from trading.data_providers.session_replay_data_provider import SessionReplayDataProvider

        dp = SessionReplayDataProvider(cfg={
            "session_id": "does-not-exist",
            "api_key": "k", "secret_key": "s",
        })

        with patch("utils.trading_state_store.TradingStateStore", return_value=store):
            with pytest.raises(ValueError, match="not found"):
                dp.load_data()

    def test_timeframe_cli_override(self):
        """--timeframe CLI flag (passed as cfg["timeframe"]) overrides session metadata."""
        store = self._make_store()
        session_id = "replay-test-3"
        self._make_session(store, session_id, ["SPY"], timeframe="Minute")

        from trading.data_providers.session_replay_data_provider import SessionReplayDataProvider

        dp = SessionReplayDataProvider(cfg={
            "session_id": session_id,
            "api_key": "k", "secret_key": "s",
            "timeframe": "Day",   # CLI override
        })

        with patch("utils.trading_state_store.TradingStateStore", return_value=store):
            dp.load_data()

        assert dp._session_metadata["timeframe"] == "Day"
