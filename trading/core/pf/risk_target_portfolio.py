from __future__ import annotations

import math
from collections import deque

from trading.core.classes import MarketSignal, Order, OrderAction, PriceData, SignalType, TickResults
from trading.core.portfolio import Portfolio
from utils.logger import Logger
from utils.utils import find_marketsignal_in_list

logger = Logger().get_logger(__name__)


class RiskTargetPortfolio(Portfolio):
    """
    Single-symbol portfolio that converts BUY/SELL signals into volatility-sized
    market orders.

    This portfolio is useful when an Algorithm answers "should I be long?"
    and the Portfolio should answer "how much should I hold?" Instead of
    spending all available cash on every BUY signal, it estimates recent
    realized volatility from close-to-close returns and scales the target
    position toward a configured annualized volatility.

    Signal behavior:
        BUY:
            Rebalance the configured symbol toward a target long quantity.
            Target exposure is:

                min(max_exposure, target_volatility / realized_volatility)
                * (signal.strength / 100)

            Until enough close history exists, default_exposure is used in
            place of target_volatility / realized_volatility.

        SELL:
            Liquidate the current position in the configured symbol.

        No signal:
            Hold the current position unless drawdown protection is triggered.

    Drawdown behavior:
        The portfolio tracks peak equity using cash plus the current symbol's
        marked-to-market value. If drawdown_limit_pct is positive and the
        current drawdown reaches that limit, the portfolio liquidates the
        position. If halt_on_drawdown is true, later BUY signals are ignored
        for the rest of this portfolio object's lifetime.

    Important notes:
        - This implementation is long-only and single-symbol.
        - It uses market orders, not bracket orders.
        - Exposure is based on total equity, not only idle cash.
        - Position quantities are integer shares; fractional shares are not
          emitted.
        - The base Portfolio returns submitted orders from
          process_market_signals_for_tick(); TickResults.metadata produced by
          this class is mainly useful when calling this strategy method directly.

    Config keys:
        symbol (str, required):
            Symbol to trade. Signals for other symbols are ignored.

        target_volatility (float, default 0.15):
            Desired annualized portfolio volatility for this position. Use
            decimal form, so 0.15 means 15% annualized volatility. Lower values
            produce smaller positions when realized volatility is available.

        volatility_lookback (int, default 20):
            Number of close-to-close returns used to estimate realized
            volatility. The class stores volatility_lookback + 1 closes.
            For daily bars, 20 is roughly one trading month. For 5-minute bars,
            choose a larger value if you want a comparable calendar horizon.

        annualization_factor (float, default 252.0):
            Multiplier used to annualize per-bar volatility:
            realized_volatility = stdev(returns) * sqrt(annualization_factor).
            Common examples:
              - Daily bars: 252
              - Hourly regular-session bars: about 1638
              - 5-minute regular-session bars: about 19656
              - 1-minute regular-session bars: about 98280

        max_exposure (float, default 1.0):
            Maximum long exposure as a multiple of equity. 1.0 means fully
            invested at most. 0.5 means at most half invested. Values above 1.0
            request leveraged exposure, but fills are still constrained by the
            order manager and available cash/buying power.

        default_exposure (float, default min(1.0, max_exposure)):
            Exposure used before enough return history exists, or when realized
            volatility is zero. This prevents the strategy from doing nothing
            during startup. Set to 0.0 if you want no trading until volatility
            is measurable.

        min_trade_value (float, default 0.0):
            Minimum notional value for a rebalance order. If the difference
            between current quantity and target quantity is worth less than
            this amount, no order is emitted. Useful for reducing churn.

        min_signal_strength (int, default 0):
            Minimum MarketSignal.strength required to act. Signal strength is
            also used as a sizing multiplier: strength 50 targets half the
            exposure of strength 100.

        tx_cost (float, default 0.0):
            Transaction cost passed to each market order. The base Portfolio
            deducts it when filled.

        drawdown_limit_pct (float, default 0.0):
            Maximum tolerated drawdown from peak equity, in decimal form.
            0.10 means 10%. A value of 0.0 disables drawdown liquidation.

        halt_on_drawdown (bool, default True):
            When true, a drawdown breach prevents future BUY entries after the
            liquidation. SELL signals remain harmless because the portfolio is
            already flat. When false, the portfolio can re-enter on later BUY
            signals after liquidating.

    Example:
        portfolio:
          portfolio: "trading.core.pf.risk_target_portfolio.RiskTargetPortfolio"
          symbol: "SPY"
          cash: 100000
          keep_history: true
          target_volatility: 0.15
          volatility_lookback: 20
          annualization_factor: 252
          max_exposure: 1.0
          default_exposure: 0.5
          min_trade_value: 100
          min_signal_strength: 50
          tx_cost: 0.0
          drawdown_limit_pct: 0.10
          halt_on_drawdown: true
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
        self.target_volatility = float(self.cfg.get("target_volatility", 0.15))
        self.volatility_lookback = int(self.cfg.get("volatility_lookback", 20))
        self.annualization_factor = float(self.cfg.get("annualization_factor", 252.0))
        self.max_exposure = float(self.cfg.get("max_exposure", 1.0))
        self.default_exposure = float(self.cfg.get("default_exposure", min(1.0, self.max_exposure)))
        self.min_trade_value = float(self.cfg.get("min_trade_value", 0.0))
        self.min_signal_strength = int(self.cfg.get("min_signal_strength", 0))
        self.tx_cost = float(self.cfg.get("tx_cost", 0.0))
        self.drawdown_limit_pct = float(self.cfg.get("drawdown_limit_pct", 0.0))
        self.halt_on_drawdown = bool(self.cfg.get("halt_on_drawdown", True))

        self._closes = deque(maxlen=self.volatility_lookback + 1)
        self._peak_equity = max(float(self.cash), 0.0)
        self._drawdown_halted = False

    def _record_close(self, tick: list[PriceData]) -> float | None:
        price = self.get_price(self.symbol, tick)
        if price is None:
            return None
        self._closes.append(float(price))
        return float(price)

    def _current_equity(self, price: float) -> float:
        quantity = self.positions.get(self.symbol).quantity if self.symbol in self.positions else 0
        return float(self.cash) + quantity * price

    def _realized_volatility(self) -> float | None:
        if len(self._closes) < self.volatility_lookback + 1:
            return None
        returns = [
            self._closes[idx] / self._closes[idx - 1] - 1.0
            for idx in range(1, len(self._closes))
            if self._closes[idx - 1] > 0
        ]
        if len(returns) < 2:
            return None
        mean_return = sum(returns) / len(returns)
        variance = sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1)
        return math.sqrt(variance) * math.sqrt(self.annualization_factor)

    def _target_exposure(self, signal: MarketSignal) -> float:
        realized_vol = self._realized_volatility()
        if realized_vol is None or realized_vol <= 0:
            base_exposure = self.default_exposure
        else:
            base_exposure = self.target_volatility / realized_vol
        strength_scale = max(0.0, min(float(signal.strength), 100.0)) / 100.0
        return max(0.0, min(base_exposure * strength_scale, self.max_exposure))

    def _order_to_target_quantity(self, target_quantity: int, tick: list[PriceData]) -> Order | None:
        current_quantity = self.positions.get(self.symbol).quantity if self.symbol in self.positions else 0
        delta = target_quantity - current_quantity
        if delta == 0:
            return None

        price = self.get_price(self.symbol, tick)
        if price is None:
            return None
        if abs(delta) * price < self.min_trade_value:
            return None

        action = OrderAction.BUY if delta > 0 else OrderAction.SELL
        return Order.create_market_order(self.symbol, action, abs(delta), self.tx_cost, tick)

    def _liquidation_order(self, tick: list[PriceData]) -> Order | None:
        if self.symbol not in self.positions:
            return None
        quantity = self.positions[self.symbol].quantity
        if quantity <= 0:
            return None
        return Order.create_market_order(self.symbol, OrderAction.SELL, quantity, self.tx_cost, tick)

    def process_tick_market_signals_logic(
        self,
        signals: list[MarketSignal],
        tick: list[PriceData],
    ) -> TickResults:
        price = self._record_close(tick)
        if price is None:
            logger.debug("RiskTargetPortfolio: no price for %s, skipping", self.symbol)
            return TickResults(orders=[])

        equity = self._current_equity(price)
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = 0.0 if self._peak_equity <= 0 else (self._peak_equity - equity) / self._peak_equity
        if self.drawdown_limit_pct > 0 and drawdown >= self.drawdown_limit_pct:
            self._drawdown_halted = self.halt_on_drawdown
            order = self._liquidation_order(tick)
            if order is not None:
                logger.info(
                    "RiskTargetPortfolio drawdown limit hit for %s: drawdown=%.2f%% limit=%.2f%%",
                    self.symbol,
                    drawdown * 100.0,
                    self.drawdown_limit_pct * 100.0,
                )
                return TickResults(orders=[order], metadata={"drawdown": drawdown, "halted": self._drawdown_halted})

        signal = find_marketsignal_in_list(self.symbol, signals)
        if signal is None or signal.strength < self.min_signal_strength:
            return TickResults(orders=[], metadata={"drawdown": drawdown, "halted": self._drawdown_halted})

        if signal.type == SignalType.SELL:
            order = self._liquidation_order(tick)
            return TickResults(orders=[order] if order is not None else [], metadata={"drawdown": drawdown})

        if signal.type != SignalType.BUY or self._drawdown_halted:
            return TickResults(orders=[], metadata={"drawdown": drawdown, "halted": self._drawdown_halted})

        exposure = self._target_exposure(signal)
        target_value = equity * exposure
        target_quantity = int(target_value / price)
        order = self._order_to_target_quantity(target_quantity, tick)
        metadata = {
            "drawdown": drawdown,
            "target_exposure": exposure,
            "target_quantity": target_quantity,
            "realized_volatility": self._realized_volatility(),
        }
        if order is not None:
            signal.metadata["order_id"] = order.order_id
        return TickResults(orders=[order] if order is not None else [], metadata=metadata)
