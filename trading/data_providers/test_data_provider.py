"""
Data provider that jsut reads the test data file locally
"""
import pandas as pd
from pathlib import Path

from trading.data_providers.data_provider import DataProvider

class TestDataProvider(DataProvider):

    def __init__(self, cfg: dict=None):
        super().__init__(cfg)
        self.load_data()

    def load_data(self):
        fp = self.cfg['path']
        tr = self.cfg['truncate']
        start_date = self.cfg['start_date'] if 'start_date' in self.cfg else None
        end_date = self.cfg['end_date'] if 'end_date' in self.cfg else None

        # Convert to Path and resolve relative to project root if not absolute
        fp_path = Path(fp)
        if not fp_path.is_absolute():
            # Get project root (parent of data_providers directory)
            project_root = Path(__file__).parent.parent
            fp_path = project_root / fp

        if tr>0:
            df = pd.read_csv(fp_path, nrows=tr)
        else:
            df = pd.read_csv(fp_path)

        # Robust timezone handling: detect and convert to EST
        normalized_timestamps, valid_mask = self._normalize_timestamps_to_est(df['timestamp'])

        # Filter out rows with malformed timestamps
        if not valid_mask.all():
            df = df[valid_mask].copy()
            df['timestamp'] = normalized_timestamps
        else:
            df['timestamp'] = normalized_timestamps

        # Ensure numeric columns are properly typed (not strings)
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop any rows where numeric conversion failed (resulted in NaN)
        if df[numeric_columns].isna().any().any():
            before_count = len(df)
            df = df.dropna(subset=numeric_columns)
            after_count = len(df)
            if before_count > after_count:
                print(f"Warning: Dropped {before_count - after_count} rows with non-numeric price/volume data")

        # Filter by date range if specified
        if start_date or end_date:
            df['timestamp_ts'] = pd.to_datetime(df['timestamp'])
            if start_date:
                df = df[df['timestamp_ts'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['timestamp_ts'] <= pd.to_datetime(end_date)]
            df.drop(['timestamp_ts'], axis=1, inplace=True)

        self.data = df

    def _normalize_timestamps_to_est(self, timestamp_series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        Normalize timestamps from any timezone to EST (America/New_York).

        Handles multiple scenarios:
        - UTC timestamps (most common from Alpaca)
        - EST timestamps (already in target timezone)
        - Other timezone formats
        - Timezone-naive timestamps (assumes UTC)
        - Malformed rows (skipped with coerce)

        Args:
            timestamp_series: Pandas Series with timestamp strings

        Returns:
            Tuple of (normalized_timestamps, valid_mask):
                - normalized_timestamps: Series with valid timestamps in EST
                - valid_mask: Boolean series indicating which rows are valid
        """
        # Try parsing with mixed format (handles various ISO8601 formats)
        # Use errors='coerce' to handle malformed data (converts to NaT)
        try:
            parsed = pd.to_datetime(timestamp_series, format='mixed', utc=True, errors='coerce')
        except Exception:
            # Fallback: try parsing as ISO8601
            try:
                parsed = pd.to_datetime(timestamp_series, format='ISO8601', utc=True, errors='coerce')
            except Exception:
                # Last resort: infer format with coerce (slower but handles edge cases)
                parsed = pd.to_datetime(timestamp_series, errors='coerce')

                # If we got NaT values, try one more approach: remove timezone info and parse
                if parsed.isna().any():
                    # Try parsing without timezone first
                    parsed_naive = pd.to_datetime(timestamp_series.str.replace(r'[+-]\d{2}:\d{2}$', '', regex=True),
                                                  errors='coerce')
                    if not parsed_naive.isna().all():
                        parsed = parsed_naive

        # Identify valid timestamps
        valid_mask = ~parsed.isna()

        if not valid_mask.all():
            # Log warning about dropped rows
            dropped_count = (~valid_mask).sum()
            print(f"Warning: Dropping {dropped_count} rows with malformed timestamps")
            # Filter to only valid timestamps
            parsed = parsed[valid_mask]

        # If timezone-naive, assume UTC
        if parsed.dt.tz is None:
            parsed = parsed.dt.tz_localize('UTC')

        # Convert to EST (America/New_York)
        est_timestamps = parsed.dt.tz_convert('America/New_York')

        # Return as timezone-naive EST timestamps (removes timezone info but keeps the correct time)
        normalized = est_timestamps.dt.tz_localize(None)

        return normalized, valid_mask