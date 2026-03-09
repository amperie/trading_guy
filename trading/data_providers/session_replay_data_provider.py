"""
Data provider that reconstructs the bar stream for a stored live session.

Fetches bars from Alpaca covering the full time range of a MongoDB session,
prepending warmup bars so the algorithm can warm up naturally before the live
session period begins.

Data flow:
    MongoDB session → symbols, timeframe, warmup_bars, session_start/end
    Alpaca API      → bars from (session_start - warmup_lookback) to session_end

The algorithm's built-in warmup gate (required_warmup_bars) automatically
suppresses signals during the warmup portion — no extra logic needed.

Config keys:
    session_id (str):      MongoDB session ID (required)
    api_key (str):         Alpaca API key (required)
    secret_key (str):      Alpaca secret key (required)
    connection_uri (str):  MongoDB URI (falls back to config.yaml state_store)
    database (str):        MongoDB database name (falls back to config.yaml)
    timeframe (str):       Override timeframe (default: from session metadata,
                           or "Minute" if metadata is missing)

After load_data(), exposes:
    _session_metadata (dict): Full metadata dict from the session document,
                              plus inferred values for missing fields.
"""
import pandas as pd

from trading.data_providers.data_provider import DataProvider
from utils.logger import Logger

logger = Logger().get_logger(__name__)


class SessionReplayDataProvider(DataProvider):
    """Fetch Alpaca bars covering a stored live session (warmup + live period)."""

    def __init__(self, cfg: dict = None):
        super().__init__(cfg)
        self._session_metadata: dict = {}

    def load_data(self):
        from datetime import datetime
        from utils.trading_state_store import TradingStateStore
        from utils.utils import compute_warmup_start_date

        session_id = self.cfg["session_id"]
        api_key = self.cfg["api_key"]
        secret_key = self.cfg["secret_key"]

        # --- Connect to MongoDB ---
        connection_uri, database = self._resolve_mongo_params(
            self.cfg.get("connection_uri"), self.cfg.get("database")
        )
        store = TradingStateStore(connection_uri=connection_uri, database=database)

        # --- Load session document ---
        session_doc = store.get_session(session_id)
        if session_doc is None:
            raise ValueError(f"Session '{session_id}' not found in MongoDB")

        metadata = session_doc.get("metadata") or {}

        # --- Resolve symbols ---
        symbols = metadata.get("symbols")
        if not symbols:
            # Infer from ticks collection
            distinct = store._ticks.distinct("symbol", {"session_id": session_id})
            symbols = list(distinct)
            if not symbols:
                raise ValueError(f"Cannot determine symbols for session '{session_id}'")
            logger.info(f"Inferred symbols from ticks: {symbols}")
        else:
            symbols = list(symbols)

        # --- Resolve timeframe ---
        timeframe = self.cfg.get("timeframe") or metadata.get("timeframe") or "Minute"

        # --- Resolve warmup_bars ---
        warmup_bars = metadata.get("warmup_bars")
        if warmup_bars is None:
            # Derive from algo class + config stored in session metadata
            al_class_path = metadata.get("algorithm_class")
            al_cfg = metadata.get("algorithm_config") or {}
            if al_class_path:
                try:
                    from utils.utils import instantiate_from_string
                    history_length = al_cfg.pop("history_length", 0) if isinstance(al_cfg, dict) else 0
                    al_cfg_copy = dict(al_cfg) if isinstance(al_cfg, dict) else {}
                    al_instance = instantiate_from_string(
                        al_class_path, cfg=al_cfg_copy, history_length=history_length
                    )
                    warmup_bars = al_instance.required_warmup_bars
                    logger.info(f"Derived warmup_bars={warmup_bars} from {al_class_path}")
                except Exception as e:
                    logger.warning(f"Could not instantiate algo to derive warmup_bars: {e}. Defaulting to 0.")
                    warmup_bars = 0
            else:
                warmup_bars = 0

        # --- Infer session_start / session_end from tick timestamps ---
        pf_data = store.load_portfolio_history(session_id)
        tick_timestamps = sorted(pf_data["tick_history"].keys()) if pf_data["tick_history"] else []
        if not tick_timestamps:
            raise ValueError(f"No tick data found for session '{session_id}'")

        session_start = tick_timestamps[0]
        session_end = tick_timestamps[-1]

        if not isinstance(session_start, datetime):
            session_start = pd.to_datetime(session_start).to_pydatetime()
        if not isinstance(session_end, datetime):
            session_end = pd.to_datetime(session_end).to_pydatetime()

        logger.info(
            f"Session '{session_id[:8]}...': "
            f"symbols={symbols} timeframe={timeframe} warmup_bars={warmup_bars} "
            f"session_start={session_start} session_end={session_end}"
        )

        # --- Compute warmup start date ---
        if warmup_bars > 0:
            warmup_start = compute_warmup_start_date(warmup_bars, timeframe, session_start)
        else:
            warmup_start = session_start

        logger.info(f"Fetching bars from {warmup_start} to {session_end}")

        # --- Fetch bars from Alpaca ---
        from trading.data_providers.alpaca_data_provider import AlpacaDataProvider

        alpaca_dp = AlpacaDataProvider(cfg={
            "api_key": api_key,
            "secret_key": secret_key,
            "symbols": symbols,
            "timeframe": timeframe,
            "start_date": warmup_start.strftime("%Y-%m-%dT%H:%M:%S"),
            "end_date": session_end.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        alpaca_dp.load_data()
        self.data = alpaca_dp.data

        # --- Expose session metadata for run.py ---
        self._session_metadata = {
            **metadata,
            "symbols":      symbols,
            "timeframe":    timeframe,
            "warmup_bars":  warmup_bars,
            "session_start": session_start.isoformat(),
            "session_end":   session_end.isoformat(),
        }

        logger.info(
            f"SessionReplayDataProvider loaded {len(self.data)} bars "
            f"({warmup_bars} warmup + live period)"
        )

    @staticmethod
    def _resolve_mongo_params(connection_uri, database):
        """Return (uri, db) falling back to ConfigManager state_store config."""
        try:
            from utils.config_manager import ConfigManager
            cfg = ConfigManager().get("state_store") or {}
        except Exception:
            cfg = {}
        uri = connection_uri or cfg.get("connection_uri", "mongodb://localhost:27017")
        db  = database       or cfg.get("database",        "trading")
        return uri, db
