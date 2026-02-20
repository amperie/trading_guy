"""
Simple portfolio that only manages one ticker
ticker is configured through the config {'symbol':'SPXU'}
"""
from trading.core.portfolio import Portfolio
from trading.core.classes import Order, MarketSignal, PriceData, SignalType, OrderAction, BracketOrder, OrderType
from trading.core.classes import TickResults
from utils.utils import find_marketsignal_in_list, find_pricedata_in_list
from utils.logger import Logger

logger = Logger().get_logger(__name__)

class SingleSymbolPortfolio(Portfolio):

    def process_tick_market_signals_logic__for_market_order(
            self, signals: list[MarketSignal],
            tick: list[PriceData]) -> list[Order]:
        # Testing just buying and selling everything

        symbol = self.cfg['symbol']

        signal = find_marketsignal_in_list(symbol, signals)

        price = find_pricedata_in_list(symbol, tick).close
        logger.debug(f"[market_order] {symbol} price={price:.2f} cash={self.cash:.2f} signal={signal.type.name if signal else None}")

        if signal is not None:
            # Buy as much as we can with the available cash (with buffer for broker drift)
            if signal.type == SignalType.BUY:
                available = self.buying_power if self.buying_power is not None else self.cash
                quantity = int(available / price)
                order = Order.create_market_order(
                    symbol, OrderAction.BUY, quantity, 0.0, tick
                )
                logger.info(f"BUY {symbol} qty={quantity} price={price:.2f} order_id={order.order_id}")
                logger.debug(f"BUY details: cash={self.cash:.2f} max_qty={quantity} order={order}")
            # Sell everything
            elif signal.type == SignalType.SELL and symbol in self.positions:
                order = Order.create_market_order(
                    symbol, OrderAction.SELL, self.positions[symbol].quantity, 0.0, tick
                )
                logger.info(f"SELL {symbol} qty={self.positions[symbol].quantity} price={price:.2f} order_id={order.order_id}")
                logger.debug(f"SELL details: position_qty={self.positions[symbol].quantity} order={order}")
            else:
                logger.debug(f"No action for {symbol}: signal={signal.type.name} has_position={symbol in self.positions}")
                return []
            return [order]
        else:
            logger.debug(f"No signal for {symbol}, skipping")
            return []

    def process_tick_market_signals_logic(
            self, signals: list[MarketSignal],
            tick: list[PriceData]) -> TickResults:
        # Testing buying and selling with bracket orders

        symbol = self.cfg['symbol']
        stop_pct = self.cfg['stop_pct']
        profit_pct = self.cfg['profit_pct']

        signal = find_marketsignal_in_list(symbol, signals)
        pd = find_pricedata_in_list(symbol, tick)
        if pd is None:
            logger.debug(f"No price data for {symbol} in tick, skipping")
            return TickResults(orders=[])
        price = pd.close
        logger.debug(f"[bracket] {symbol} price={price:.2f} cash={self.cash:.2f} signal={signal.type.name if signal else None} positions={list(self.positions.keys())}")

        if signal is not None and signal.type == SignalType.BUY:
            available = self.buying_power if self.buying_power is not None else self.cash
            quantity = int(available / price)
            stop_price = round(price * (1.0 - stop_pct / 100.0), 2)
            profit_price = round(price * (1.0 + profit_pct / 100), 2)
            bo = BracketOrder.create_bracket_order(
                symbol, profit_price, stop_price,
                quantity, 0.0, tick
            )
            signal.metadata['order_id'] = bo.order_id
            logger.info(f"BUY BRACKET {symbol} qty={quantity} price={price:.2f} stop={stop_price:.2f} profit={profit_price:.2f} order_id={bo.order_id}")
            logger.debug(f"BUY BRACKET details: cash={self.cash:.2f} stop_pct={stop_pct} profit_pct={profit_pct} bracket_order={bo}")
            ret_val = TickResults(orders=[bo])
            return ret_val
        elif signal is not None and signal.type == SignalType.SELL:

            # Skip selling signals to test bracket child orders
            return TickResults(orders=[])

            if symbol not in self.positions:
                logger.debug(f"SELL signal for {symbol} but no position held, skipping")
                return TickResults(orders=[])
            qty = self.positions[symbol].quantity
            so = Order.create_market_order(
                symbol, OrderAction.SELL, qty, 0.0, tick
            )
            signal.metadata['order_id'] = so.order_id
            # Cancel all other pending bracket orders so there isn't a race condition on sales
            self.om.cancel_all_pending_orders(OrderType.BRACKET)
            logger.info(f"SELL {symbol} qty={qty} price={price:.2f} order_id={so.order_id} (canceled pending brackets)")
            logger.debug(f"SELL details: position_qty={qty} order={so}")
            return TickResults(orders=[so])
        else:
            logger.debug(f"No actionable signal for {symbol}: signal={signal.type.name if signal else 'None'}")
            return TickResults(orders=[])