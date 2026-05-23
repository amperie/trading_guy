"""
Simulator for backtesting
Iterates and feeds data through the system as if it were coming from a real time source
"""
import time

from trading.core.algorithm import Algorithm
from trading.core.portfolio import Portfolio
from trading.core.om.order_manager import OrderManager
from trading.data_providers.data_provider import DataProvider
from trading.core.classes import PriceData, TickResults
from trading.engines.base_engine import BaseEngine
from utils.logger import Logger
from utils.status_line import StatusLine

logger = Logger().get_logger(__name__)

class BacktestingEngine(BaseEngine):
    """
    Backtesting engine that simulates a live trading loop.

    Responsibilities:
    - Pulls ticks from the configured DataProvider.
    - Sends ticks into the Algorithm to produce MarketSignals.
    - Passes signals into the Portfolio to create Orders.
    - Lets the Portfolio/OrderManager manage fills and positions.

    Typical usage:
        engine = BacktestingEngine(cfg=cfg)
        engine.run()
    """

    def __init__(
            self, cfg:dict=None, dp: DataProvider=None,
            al: Algorithm=None, om: OrderManager=None,
            pf: Portfolio=None
        ):
        super().__init__(cfg=cfg, dp=dp, al=al, om=om, pf=pf)
        self._progress_started_at: float | None = None
        self._progress_total_ticks = 0
        self._progress_processed_ticks = 0
        self._progress_current_timestamp = None
        self._progress_phase = "idle"
        self._status_line = StatusLine(enabled=self.cfg.get("status_line_enabled"))
        self._status_line_update_every = max(1, int(self.cfg.get("status_line_update_every", 100)))

    def finalize(self):
        """Hook for any end-of-run cleanup (optional)."""
        pass

    def get_progress_snapshot(self):
        elapsed_seconds = None
        eta_seconds = None
        if self._progress_started_at is not None:
            elapsed_seconds = max(0.0, time.monotonic() - self._progress_started_at)
            if self._progress_processed_ticks > 0 and self._progress_total_ticks > self._progress_processed_ticks:
                avg_tick_seconds = elapsed_seconds / self._progress_processed_ticks
                eta_seconds = avg_tick_seconds * (self._progress_total_ticks - self._progress_processed_ticks)
        return {
            "phase": self._progress_phase,
            "ticks_processed": self._progress_processed_ticks,
            "ticks_total": self._progress_total_ticks,
            "current_timestamp": self._progress_current_timestamp,
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": eta_seconds,
        }

    @staticmethod
    def _format_duration(seconds):
        if seconds is None:
            return "n/a"
        total = int(max(0, round(seconds)))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _refresh_status_line(self, final: bool = False):
        snapshot = self.get_progress_snapshot()
        if not snapshot:
            return
        text = (
            f"[BT] phase={snapshot.get('phase')} "
            f"ticks={snapshot.get('ticks_processed', 0)}/{snapshot.get('ticks_total', 0)} "
            f"elapsed={self._format_duration(snapshot.get('elapsed_seconds'))} "
            f"eta={self._format_duration(snapshot.get('eta_seconds'))}"
        )
        current_ts = snapshot.get("current_timestamp")
        if current_ts is not None:
            text += f" ts={current_ts}"
        if final:
            text += " completed"
        self._status_line.update(text)

    def run(self):
        """
        Run the full backtest over the DataProvider's ticks.

        Loop:
            tick -> Algorithm -> signals -> Portfolio -> orders -> OrderManager
        """
        iters = 0
        length = self.dp.get_data_length()
        self._progress_started_at = time.monotonic()
        self._progress_total_ticks = int(length or 0)
        self._progress_processed_ticks = 0
        self._progress_current_timestamp = None
        self._progress_phase = "backtest"
        logger.info(f"Starting backtest for {length} timestamps")
        self._refresh_status_line()

        try:
            for tick in self.dp.iterate():
                iters += 1
                if len(tick) > 0:
                    ts = tick[0].timestamp
                else:
                    ts = None
                self._progress_processed_ticks = iters
                self._progress_current_timestamp = ts

                if not self._status_line.enabled and iters % 500 == 0:
                    logger.info(f"Running iteration {iters} of {length} for timestamp {ts or 'None'}")

                self.on_tick(tick)
                if (
                    iters == 1
                    or iters == length
                    or iters % self._status_line_update_every == 0
                ):
                    self._refresh_status_line()
                self._check_debug()
        finally:
            self._progress_phase = "completed"
            self._refresh_status_line(final=True)
            self._status_line.close()

        logger.info(
            f"Backtest complete: {iters} ticks processed"
            + (f" | final_value={self.pf.total_value:.2f} cash={self.pf.cash:.2f} positions={list(self.pf.positions.keys())}" if self.pf else "")
        )

    def on_tick(self, tick: list[PriceData]) -> TickResults:
        """
        Process one tick and return the portfolio's results.
        """
        market_signals = self.al.on_data(tick)
        ret_val = self.pf.process_market_signals_for_tick(market_signals, tick)
        self._persist_tick(tick, market_signals, ret_val)
        return ret_val
