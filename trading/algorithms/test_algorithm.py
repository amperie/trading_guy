from typing import Dict, Any
import random
import pandas as pd
from trading.core.algorithm import Algorithm
from trading.core.classes import PriceData, MarketSignal, SignalType
from utils.logger import Logger

logger = Logger().get_logger(__name__)


class TestAlgorithm(Algorithm):

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        logger.debug(f"on_data_logic called with {len(data)} price data points")
        r = random.randint(0,100)
        if r > 50:
            signal = SignalType.BUY
        else:
            signal = SignalType.SELL

        retval = [MarketSignal(signal, x.symbol, 100) for x in data]
        symbols = [x.symbol for x in data]
        logger.info(f"Generated {signal.name} signal for {symbols} (r={r})")
        logger.debug(f"Returning {len(retval)} signals: {[(s.type.name, s.symbol, s.strength) for s in retval]}")
        return retval

class ReadSignalsFromFile(Algorithm):

    def __init__(self, cfg: Dict[str, Any]=None, history_length: int=0):
        super().__init__(cfg, history_length)
        self.threshold = cfg.get('threshold', 0.5) if cfg else 0.5

        # Load CSV file path from config
        csv_path = cfg.get('csv_path') if cfg else None
        if not csv_path:
            raise ValueError("csv_path must be specified in config for ReadSignalsFromFile")

        # Load CSV into pandas dataframe
        self.signals_df = pd.read_csv(csv_path)

        # Format timestamp column to pandas datetime
        # Assumes the timestamp column is named 'timestamp'
        timestamp_col = cfg.get('timestamp_column', 'timestamp') if cfg else 'timestamp'
        if timestamp_col in self.signals_df.columns:
            self.signals_df[timestamp_col] = pd.to_datetime(self.signals_df[timestamp_col])
            # Set timestamp as index for faster lookups
            self.signals_df = self.signals_df.set_index(timestamp_col)
        else:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in CSV")

        # Store the name of the signal column (first non-timestamp column)
        self.signal_column = self.signals_df.columns[0]

    def on_data_logic(self, data: list[PriceData]) -> list[MarketSignal]:
        """
        Read signals from the loaded CSV file based on current timestamp.

        The CSV should have:
        - A timestamp column (default: 'timestamp')
        - A signal column (first column after timestamp)
        """
        signals = []

        for price_data in data:
            # Get current timestamp from the data
            current_timestamp = price_data.timestamp

            # Convert to pandas datetime if needed
            if isinstance(current_timestamp, pd.Timestamp):
                lookup_timestamp = current_timestamp
            else:
                lookup_timestamp = pd.to_datetime(current_timestamp)

            # Normalize timezone to match the dataframe index
            # If the index is timezone-naive, make the lookup timestamp timezone-naive
            # If the index is timezone-aware, make the lookup timestamp timezone-aware
            if self.signals_df.index.tz is None:
                # Index is timezone-naive, remove timezone from lookup timestamp if present
                if lookup_timestamp.tz is not None:
                    lookup_timestamp = lookup_timestamp.tz_localize(None)
            else:
                # Index is timezone-aware, add timezone to lookup timestamp if missing
                if lookup_timestamp.tz is None:
                    lookup_timestamp = lookup_timestamp.tz_localize(self.signals_df.index.tz)
                else:
                    # Convert to the same timezone as the index
                    lookup_timestamp = lookup_timestamp.tz_convert(self.signals_df.index.tz)

            try:
                # Retrieve the signal value from the dataframe for this timestamp
                signal_value = self.signals_df.loc[lookup_timestamp, self.signal_column]

                # Convert signal value to SignalType based on threshold
                if signal_value <= self.threshold:
                    signal_type = SignalType.BUY
                    strength = int(min(signal_value * 100, 100))

                    signals.append(MarketSignal(
                        type=signal_type,
                        symbol=price_data.symbol,
                        strength=strength
                    ))
            except KeyError:
                # Timestamp not found in the dataframe - no signal for this timestamp
                pass

        return signals
