from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from trading.core.classes import (
    BracketOrder, MarketSignal, Order, OrderAction, OrderStatus, OrderType,
    PriceData, SignalType, TickResults,
)
from trading.core.portfolio import Portfolio
from utils.logger import Logger
from utils.utils import find_marketsignal_in_list

logger = Logger().get_logger(__name__)


class RiskManagedSingleSymbolPortfolio(Portfolio):
    """Single-symbol ATR brackets, risk sizing, signal exits, and circuit breakers."""

    def __init__(self, cfg=None, order_manager=None, starting_cash=0.0,
                 starting_positions=None, keep_history=False):
        super().__init__(cfg, order_manager, starting_cash, starting_positions, keep_history)
        self.symbol = self.cfg["symbol"]
        self.tx_cost = float(self.cfg.get("tx_cost", 0.0))
        self.max_exposure = float(self.cfg.get("max_exposure", 0.5))
        self.risk_per_trade = float(self.cfg.get("risk_per_trade", 0.005))
        self.min_signal_strength = float(self.cfg.get("min_signal_strength", 50))
        self.min_trade_value = float(self.cfg.get("min_trade_value", 100))
        self.atr_stop_multiple = float(self.cfg.get("atr_stop_multiple", 2.25))
        self.profit_target_r_multiple = float(self.cfg.get("profit_target_r_multiple", 1.5))
        self.fallback_stop_pct = float(self.cfg.get("fallback_stop_pct", 1.5))
        self.cooldown_minutes = int(self.cfg.get("cooldown_minutes", 60))
        self.max_holding_minutes = int(self.cfg.get("max_holding_minutes", 0))
        self.max_entries_per_day = int(self.cfg.get("max_entries_per_day", 2))
        self.max_daily_loss_pct = float(self.cfg.get("max_daily_loss_pct", 0.02))
        self.max_drawdown_pct = float(self.cfg.get("max_drawdown_pct", 0.08))
        self.drawdown_confirmation_bars = int(self.cfg.get("drawdown_confirmation_bars", 3))
        self.halt_on_drawdown = bool(self.cfg.get("halt_on_drawdown", False))
        self.entry_start = self._parse_time(self.cfg.get("entry_start", "09:35"))
        self.entry_end = self._parse_time(self.cfg.get("entry_end", "15:30"))
        self.flatten_before_close = bool(self.cfg.get("flatten_before_close", False))
        self.flatten_time = self._parse_time(self.cfg.get("flatten_time", "15:55"))

        self._peak_equity = max(float(self.cash), 0.0)
        self._day = None
        self._day_start_equity = float(self.cash)
        self._entries_today = 0
        self._drawdown_bars = 0
        self._halted = False
        self._cooldown_until: datetime | None = None
        self._entry_time: datetime | None = None
        self._was_holding = bool(self.positions.get(self.symbol))

    @staticmethod
    def _parse_time(value) -> time:
        if isinstance(value, time):
            return value
        return time.fromisoformat(str(value))

    @staticmethod
    def _et_timestamp(value: datetime) -> datetime:
        return value if value.tzinfo is None else value.astimezone(ZoneInfo("America/New_York"))

    def _equity(self, price: float) -> float:
        position = self.positions.get(self.symbol)
        return self.cash + (position.quantity * price if position else 0.0)

    def _roll_day(self, timestamp: datetime, equity: float) -> None:
        day = self._et_timestamp(timestamp).date()
        if day != self._day:
            self._day = day
            self._day_start_equity = equity
            self._entries_today = 0

    def _set_cooldown(self, timestamp: datetime) -> None:
        if self.cooldown_minutes > 0:
            self._cooldown_until = timestamp + timedelta(minutes=self.cooldown_minutes)

    def _exit(self, tick: list[PriceData], timestamp: datetime, reason: str) -> TickResults:
        bracket = self.find_active_bracket(self.symbol)
        self._set_cooldown(timestamp)
        metadata = {"exit_reason": reason}
        if bracket is not None and bracket.status == OrderStatus.PENDING_SALE:
            return TickResults(trigger_manual_sales=[bracket], metadata=metadata)
        if bracket is not None and bracket.status == OrderStatus.PENDING:
            return TickResults(cancel_pending_order_types=[OrderType.BRACKET], metadata=metadata)
        position = self.positions.get(self.symbol)
        if position and position.quantity > 0:
            order = Order.create_market_order(
                self.symbol, OrderAction.SELL, position.quantity, self.tx_cost, tick
            )
            return TickResults(orders=[order], metadata=metadata)
        return TickResults(metadata=metadata)

    def _risk_exit_reason(self, equity: float) -> str | None:
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = 0.0 if self._peak_equity <= 0 else 1.0 - equity / self._peak_equity
        self._drawdown_bars = self._drawdown_bars + 1 if drawdown >= self.max_drawdown_pct > 0 else 0
        daily_loss = 0.0 if self._day_start_equity <= 0 else 1.0 - equity / self._day_start_equity
        if self.max_daily_loss_pct > 0 and daily_loss >= self.max_daily_loss_pct:
            return "daily_loss_limit"
        if self._drawdown_bars >= max(1, self.drawdown_confirmation_bars):
            self._halted = self.halt_on_drawdown
            if not self._halted:
                self._peak_equity = equity
            self._drawdown_bars = 0
            return "drawdown_limit"
        return None

    def _entry_allowed(self, timestamp: datetime) -> bool:
        market_time = self._et_timestamp(timestamp).time()
        return (
            not self._halted
            and self.entry_start <= market_time <= self.entry_end
            and (self._cooldown_until is None or timestamp >= self._cooldown_until)
            and (self.max_entries_per_day <= 0 or self._entries_today < self.max_entries_per_day)
        )

    def _entry_quantity(self, signal: MarketSignal, price: float, equity: float) -> tuple[int, float]:
        atr = float(signal.metadata.get("atr", 0.0) or 0.0)
        stop_distance = atr * self.atr_stop_multiple if atr > 0 else price * self.fallback_stop_pct / 100
        strength = max(0.0, min(float(signal.strength), 100.0)) / 100.0
        exposure_qty = int(equity * self.max_exposure * strength / price)
        risk_qty = int(equity * self.risk_per_trade * strength / stop_distance) if self.risk_per_trade > 0 else exposure_qty
        available = self.buying_power if self.buying_power is not None else self.cash
        cash_qty = int(max(0.0, available - self.tx_cost) / price)
        return min(exposure_qty, risk_qty, cash_qty), stop_distance

    def process_tick_market_signals_logic(self, signals: list[MarketSignal],
                                          tick: list[PriceData]) -> TickResults:
        if not tick:
            return TickResults()
        timestamp = tick[0].timestamp
        price = self.get_price(self.symbol, tick)
        if price is None:
            return TickResults()

        position = self.positions.get(self.symbol)
        holding = bool(position and position.quantity > 0)
        if holding and not self._was_holding:
            self._entry_time = timestamp
        elif self._was_holding and not holding:
            self._entry_time = None
            self._set_cooldown(timestamp)
        self._was_holding = holding

        equity = self._equity(price)
        self._roll_day(timestamp, equity)
        risk_reason = self._risk_exit_reason(equity)
        if risk_reason:
            return self._exit(tick, timestamp, risk_reason)
        if self.flatten_before_close and self._et_timestamp(timestamp).time() >= self.flatten_time:
            return self._exit(tick, timestamp, "end_of_day")
        if holding and self.max_holding_minutes > 0 and self._entry_time:
            if timestamp >= self._entry_time + timedelta(minutes=self.max_holding_minutes):
                return self._exit(tick, timestamp, "time_stop")

        signal = find_marketsignal_in_list(self.symbol, signals)
        if signal is None:
            return TickResults(metadata={"equity": equity})
        if signal.type == SignalType.SELL:
            return self._exit(tick, timestamp, "algorithm_sell")
        if signal.type != SignalType.BUY or signal.strength < self.min_signal_strength:
            return TickResults()
        if holding or self.find_active_bracket(self.symbol) is not None or not self._entry_allowed(timestamp):
            return TickResults()

        quantity, stop_distance = self._entry_quantity(signal, price, equity)
        if quantity <= 0 or quantity * price < self.min_trade_value:
            return TickResults()
        stop_price = price - stop_distance
        profit_price = price + stop_distance * self.profit_target_r_multiple
        bracket = BracketOrder.create_bracket_order(
            self.symbol, profit_price, stop_price, quantity, self.tx_cost, tick
        )
        bracket.intended_entry_price = price
        signal.metadata.update({
            "order_id": bracket.order_id,
            "risk_stop_distance": stop_distance,
            "target_exposure": quantity * price / equity if equity > 0 else 0.0,
        })
        self._entries_today += 1
        logger.info(
            "Risk-managed BUY %s qty=%s exposure=%.1f%% stop=%.2f target=%.2f",
            self.symbol, quantity, quantity * price / equity * 100, stop_price, profit_price,
        )
        return TickResults(orders=[bracket])

    def reconfigure(self, new_params: dict) -> None:
        super().reconfigure(new_params)
        for name in (
            "tx_cost", "max_exposure", "risk_per_trade", "min_signal_strength",
            "min_trade_value", "atr_stop_multiple", "profit_target_r_multiple",
            "fallback_stop_pct", "max_daily_loss_pct", "max_drawdown_pct",
        ):
            if name in new_params:
                setattr(self, name, float(new_params[name]))
        for name in ("cooldown_minutes", "max_holding_minutes", "max_entries_per_day", "drawdown_confirmation_bars"):
            if name in new_params:
                setattr(self, name, int(new_params[name]))
        if "entry_start" in new_params:
            self.entry_start = self._parse_time(new_params["entry_start"])
        if "entry_end" in new_params:
            self.entry_end = self._parse_time(new_params["entry_end"])
