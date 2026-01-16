"""
Simple portfolio that only manages one ticker
ticker is configured through the config {'symbol':'SPXU'}
"""
from trading.core.portfolio import Portfolio
from trading.core.classes import Order, MarketSignal, PriceData, SignalType, OrderAction, BracketOrder, OrderType
from trading.core.classes import TickResults
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
            tick: list[PriceData]) -> TickResults:
        # Testing buying and selling with bracket orders

        symbol = self.cfg['symbol']
        stop_pct = self.cfg['stop_pct']
        profit_pct = self.cfg['profit_pct']

        signal = find_marketsignal_in_list(symbol, signals)
        pd = find_pricedata_in_list(symbol, tick)
        if pd is None:
            return TickResults(orders=[])
        price = pd.close

        if signal is not None and signal.type == SignalType.BUY:
            # Buy as much as we can with the available cash
            quantity = int(self.cash / price)
            bo = BracketOrder.create_bracket_order(
                symbol, price * (1.0 + profit_pct/100), price * (1.0 - stop_pct/100.0),
                quantity, 0.0, tick
            )
            signal.metadata['order_id'] = bo.order_id
            ret_val = TickResults(orders=[bo])
            return ret_val
        elif signal is not None and signal.type == SignalType.SELL:
            if symbol not in self.positions:
                return TickResults(orders=[])
            so = Order.create_market_order(
                symbol, OrderAction.SELL, self.positions[symbol].quantity, 0.0, tick
            )
            signal.metadata['order_id'] = so.order_id
            # Cancel all other pending bracket orders so there isn't a race condition on sales
            self.om.cancel_all_pending_orders(OrderType.BRACKET)
            return TickResults(orders=[so])
        else:
            return TickResults(orders=[])