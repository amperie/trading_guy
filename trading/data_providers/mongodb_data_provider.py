"""
DataProvider that replays stored tick data from a MongoDB session.

Useful for re-running an algorithm against a previously live-traded session
without re-downloading data from Alpaca. Ticks are read from the `ticks`
collection persisted by TradingStateStore.

Config keys:
    session_id      (required) UUID string of the session to replay
    connection_uri  (optional) MongoDB URI; falls back to state_store config
    database        (optional) database name; falls back to state_store config
    start_date      (optional) ISO date string, inclusive lower bound
    end_date        (optional) ISO date string, inclusive upper bound
"""
import pandas as pd
from pymongo import MongoClient, ASCENDING

from trading.data_providers.data_provider import DataProvider
from utils.config_manager import ConfigManager
from utils.logger import Logger

logger = Logger().get_logger(__name__)


class MongoDBDataProvider(DataProvider):

    def __init__(self, cfg: dict = None):
        super().__init__(cfg)
        self.load_data()

    def load_data(self):
        session_id = self.cfg.get("session_id")
        if not session_id:
            raise ValueError("MongoDBDataProvider requires 'session_id' in config")

        connection_uri, database = self._resolve_mongo_params()

        start_date = self.cfg.get("start_date")
        end_date = self.cfg.get("end_date")

        client = MongoClient(connection_uri)
        db = client[database]
        ticks_col = db["ticks"]

        query = {"session_id": session_id}
        if start_date or end_date:
            ts_filter = {}
            if start_date:
                ts_filter["$gte"] = pd.to_datetime(start_date, utc=True).to_pydatetime()
            if end_date:
                ts_filter["$lte"] = pd.to_datetime(end_date, utc=True).to_pydatetime()
            query["timestamp"] = ts_filter

        docs = list(ticks_col.find(query).sort("timestamp", ASCENDING))
        client.close()

        if not docs:
            logger.warning(f"No ticks found for session_id={session_id}")
            self.data = pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
            return

        rows = [{k: v for k, v in doc.items() if k not in ("_id", "session_id")} for doc in docs]
        df = pd.DataFrame(rows)

        # Convert UTC datetime objects from MongoDB to EST timezone-naive
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], utc=True)
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
        )

        # Ensure numeric columns
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        before = len(df)
        df = df.dropna(subset=numeric_columns)
        dropped = before - len(df)
        if dropped:
            logger.warning(f"Dropped {dropped} rows with non-numeric OHLCV data for session {session_id}")

        self.data = df
        logger.info(
            f"MongoDBDataProvider: session={session_id}, "
            f"rows={len(self.data)}, ticks={self.data['timestamp'].nunique()}, "
            f"symbols={list(self.data['symbol'].unique())}"
        )

    def _resolve_mongo_params(self) -> tuple[str, str]:
        """Return (connection_uri, database), falling back to state_store config."""
        ss_cfg = ConfigManager().get("state_store") or {}
        connection_uri = (
            self.cfg.get("connection_uri")
            or ss_cfg.get("connection_uri")
            or "mongodb://localhost:27017"
        )
        database = (
            self.cfg.get("database")
            or ss_cfg.get("database")
            or "trading"
        )
        return connection_uri, database
