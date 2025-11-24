"""
Simple portfolio that only manages one ticker
ticker is configured through the config {'symbol':'SPXU'}
"""
from core.portfolio import Portfolio
from core.order_manager import OrderManager
from core.classes import Order, MarketSignal, PriceData, SignalType, OrderAction, BracketOrder
from utils.utils import find_marketsignal_in_list, find_pricedata_in_list

class SingleSymbolPortfolio(Portfolio):

    def process_tick_market_signals_logic__for_market_order(
            self, signals: list[MarketSignal],
            tick: list[PriceData]) -> list[Order]:
        # Testing just buying and selling everything

        symbol = self.cfg['symbol']
        signal = find_marketsignal_in_list(symbol, signals)

        price = find_pricedata_in_list(symbol, tick).close

        if signal is not None:
            # Buy as much as we can with the available cash
            if signal.type == SignalType.BUY:
                quantity = int(self.cash/price)
                order = Order.create_market_order(
                    symbol, OrderAction.BUY, quantity, 0.0, tick
                )
            # Sell everything
            elif signal.type == SignalType.SELL and symbol in self.positions:
                order = Order.create_market_order(
                    symbol, OrderAction.SELL, self.positions[symbol].quantity, 0.0, tick
                )
            else:
                return []
            return [order]
        else:
            return []

    def process_tick_market_signals_logic(
            self, signals: list[MarketSignal],
            tick: list[PriceData]) -> list[Order]:
        # Testing buying and selling with bracket orders

        symbol = self.cfg['symbol']
        signal = find_marketsignal_in_list(symbol, signals)
        price = find_pricedata_in_list(symbol, tick).close

        if signal is not None and signal.type == SignalType.BUY:
            # Buy as much as we can with the available cash
            quantity = int(self.cash / price)
            bo = BracketOrder.create_bracket_order(
                symbol, price+2.0, price-2.0, quantity, 0.0, tick
            )
            return [bo]
        else:
            return []