from __future__ import annotations

from datetime import time

from trading.core.classes import MarketSignal, Order, OrderAction, PriceData, TickResults
from trading.core.portfolio import Portfolio
from utils.logger import Logger

logger = Logger().get_logger(__name__)


class DayBoundaryPortfolio(Portfolio):
    """
    Single-symbol portfolio that trades once at configured day boundaries.

    Default behavior is overnight: buy near the close, sell near the next open.
    Set flip_behavior=true to hold intraday: buy near the open, sell near the
    close. Signals are ignored.

    Config keys:
        symbol: required ticker.
        flip_behavior: false = close->open, true = open->close.
        market_open_hour/minute: default 9:30.
        market_close_hour/minute: default 16:00.
        tx_cost: default 0.0.
    """

    def __init__(
        self,
        cfg: dict = None,
        order_manager=None,
        starting_cash: float = 0.0,
        starting_positions=None,
        keep_history: bool = False,
    ):
        super().__init__(cfg, order_manager, starting_cash, starting_positions, keep_history)
        self.symbol = self.cfg["symbol"]
        self.flip_behavior = bool(self.cfg.get("flip_behavior", False))
        self.tx_cost = float(self.cfg.get("tx_cost", 0.0))
        self.market_open = time(
            int(self.cfg.get("market_open_hour", 9)),
            int(self.cfg.get("market_open_minute", 30)),
        )
        self.market_close = time(
            int(self.cfg.get("market_close_hour", 16)),
            int(self.cfg.get("market_close_minute", 0)),
        )
        self._last_buy_date = None
        self._last_sell_date = None

    def _is_open_window(self, tick_time: time) -> bool:
        return self.market_open <= tick_time < self.market_close

    def _is_close_window(self, tick_time: time) -> bool:
        return tick_time >= self.market_close

    def _buy_order(self, tick: list[PriceData]) -> Order | None:
        price = self.get_price(self.symbol, tick)
        if price is None:
            logger.debug("DayBoundaryPortfolio: no price for %s, skipping buy", self.symbol)
            return None
        available = self.buying_power if self.buying_power is not None else self.cash
        quantity = int(available / price)
        if quantity <= 0 or self.symbol in self.positions:
            return None
        return Order.create_market_order(self.symbol, OrderAction.BUY, quantity, self.tx_cost, tick)

    def _sell_order(self, tick: list[PriceData]) -> Order | None:
        position = self.positions.get(self.symbol)
        if position is None or position.quantity <= 0:
            return None
        return Order.create_market_order(self.symbol, OrderAction.SELL, position.quantity, self.tx_cost, tick)

    def process_tick_market_signals_logic(
        self,
        signals: list[MarketSignal],
        tick: list[PriceData],
    ) -> TickResults:
        if not tick:
            return TickResults(orders=[])

        ts = tick[0].timestamp
        day = ts.date()
        tick_time = ts.time()

        if self.flip_behavior:
            if self._is_open_window(tick_time) and self._last_buy_date != day:
                order = self._buy_order(tick)
                self._last_buy_date = day
                return TickResults(orders=[order] if order else [])
            if self._is_close_window(tick_time) and self._last_sell_date != day:
                order = self._sell_order(tick)
                self._last_sell_date = day
                return TickResults(orders=[order] if order else [])
            return TickResults(orders=[])

        if self._is_open_window(tick_time) and self._last_sell_date != day:
            order = self._sell_order(tick)
            self._last_sell_date = day
            return TickResults(orders=[order] if order else [])
        if self._is_close_window(tick_time) and self._last_buy_date != day:
            order = self._buy_order(tick)
            self._last_buy_date = day
            return TickResults(orders=[order] if order else [])
        return TickResults(orders=[])

    def reconfigure(self, new_params: dict) -> None:
        super().reconfigure(new_params)
        if "market_open_hour" in new_params or "market_open_minute" in new_params:
            self.market_open = time(
                int(self.cfg.get("market_open_hour", 9)),
                int(self.cfg.get("market_open_minute", 30)),
            )
        if "market_close_hour" in new_params or "market_close_minute" in new_params:
            self.market_close = time(
                int(self.cfg.get("market_close_hour", 16)),
                int(self.cfg.get("market_close_minute", 0)),
            )
