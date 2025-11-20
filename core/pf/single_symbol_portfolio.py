"""
Simple portfolio that only manages one ticker
ticker is configured through the config {'symbol':'SPXU'}
"""
from core.portfolio import Portfolio
from core.order_manager import OrderManager
from core.classes import Order, MarketSignal, PriceData, SignalType, OrderAction
from utils.utils import find_marketsignal_in_list, find_pricedata_in_list

class SingleSymbolPortfolio(Portfolio):

    def process_tick_market_signals_logic_buy_sell(
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
                order = self.order_manager.buy(
                    symbol, quantity, tick
                )
            # Sell everything
            elif signal.type == SignalType.SELL and symbol in self.positions:
                order = self.order_manager.sell(
                    symbol, self.positions[symbol].quantity,
                    tick
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
            bo = self.order_manager.bracket_order(
                symbol, quantity, 0.05, 0.05, tick, 0.0
            )
            return [bo]
        else:
            return []