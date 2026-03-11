"""
Metadata extractor for stored live sessions.

Connects to MongoDB, loads session document, and resolves:
    symbols, timeframe, warmup_bars, session_start, session_end

Exposes these via ``_session_metadata`` after ``load_data()``.
The DataProvider's ``self.data`` is left empty — callers (run.py) are
responsible for fetching bars separately via AlpacaDataProvider.

Config keys:
    session_id (str):      MongoDB session ID (required)
    api_key (str):         (reserved for future use — not used here)
    secret_key (str):      (reserved for future use — not used here)
    connection_uri (str):  MongoDB URI (falls back to config.yaml state_store)
    database (str):        MongoDB database name (falls back to config.yaml)
    timeframe (str):       Override timeframe (default: from session metadata,
                           or "Minute" if metadata is missing)

After load_data(), exposes:
    _session_metadata (dict): Resolved metadata dict with keys:
        symbols, timeframe, warmup_bars, session_start (ISO str),
        session_end (ISO str), plus any raw session metadata fields.
"""
import pandas as pd

from trading.data_providers.data_provider import DataProvider
from utils.logger import Logger

logger = Logger().get_logger(__name__)


class SessionReplayDataProvider(DataProvider):
    """Load session metadata from MongoDB; leave bar fetching to the caller."""

    def __init__(self, cfg: dict = None):
        super().__init__(cfg)
        self._session_metadata: dict = {}

    def load_data(self):
        from datetime import datetime
        from utils.trading_state_store import TradingStateStore

        session_id = self.cfg["session_id"]

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

        # --- Leave self.data empty (callers fetch bars themselves) ---
        self.data = pd.DataFrame()

        # --- Expose resolved metadata ---
        self._session_metadata = {
            **metadata,
            "symbols":       symbols,
            "timeframe":     timeframe,
            "warmup_bars":   warmup_bars,
            "session_start": session_start.isoformat(),
            "session_end":   session_end.isoformat(),
        }

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
