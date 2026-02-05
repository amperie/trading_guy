import os
from alpaca.data.live.stock import StockDataStream
from alpaca.data.enums import \
    DataFeed  # IEX or SIP
from dotenv import load_dotenv

from trading.core.algorithm import Algorithm
from trading.core.classes import PriceData, TickResults
from trading.core.om.order_manager import OrderManager
from trading.core.portfolio import Portfolio
from trading.data_providers.data_provider import DataProvider
from trading.engines.base_engine import BaseEngine


# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

class AlpacaRealTimeEngine(BaseEngine):

    """
    Engine to get real time data from Alpaca. Usually will feed to aggregator Engine which
    will then feed to another engine for processing signals and buys
    Test URL:
                url_override="wss://stream.data.alpaca.markets/v2/test"
    Test symbol: "FAKEPACA"
    """
    def __init__(
            self, cfg:dict= None, dp: DataProvider = None,
            al: Algorithm=None, om: OrderManager=None,
            pf: Portfolio=None
        ):
        super().__init__(cfg, dp, al, om, pf)
        self.cfg = cfg
        self._api_key = cfg.get("api_key", "")
        self._secret_key = cfg.get("secret_key", "")
        if self._secret_key == "" or self._api_key == "":
            raise Exception("secret_key and api_key are required for Alpaca Real-Time Engine")

        self._downstream_engine = cfg.get("downstream_engine", None)
        self._bar_subscribe = cfg.get("subscribe_to_bars", True)
        self._symbols_to_subscribe = cfg.get("symbols_to_subscribe", [])
        if len(self._symbols_to_subscribe) < 1:
            raise Exception("symbols_to_subscribe is required for Alpaca Real-Time Engine")

        self._quote_subscribe = cfg.get("subscribe_to_quotes", True)
        self._trades_subscribe = cfg.get("subscribe_to_trades", False)
        self._quote_subscribe = cfg.get("subscribe_to_trades", False)
        self._override_url = cfg.get("override_url", None)


    def run(self):

        if self._override_url is not None:
            stream = StockDataStream(
                api_key=api_key,
                secret_key=secret_key,
                url_override=self._override_url,
                feed=DataFeed.IEX,
                # or DataFeed.SIP if you have SIP access
                raw_data=False
            )
        else:
            stream = StockDataStream(
                api_key=api_key,
                secret_key=secret_key,
                feed=DataFeed.IEX,
                # or DataFeed.SIP if you have SIP access
                raw_data=False
            )
        self.stream = stream

        if self._bar_subscribe:
            stream.subscribe_bars(on_bar, *self._symbols_to_subscribe)
        if self._trades_subscribe:
            stream.subscribe_trades(on_trade, *self._symbols_to_subscribe)
        if self._quote_subscribe:
            stream.subscribe_quotes(on_quote, *self._symbols_to_subscribe)

        self.stream.run()

    def on_tick(self, tick: list[PriceData]) -> TickResults:
        pass

    def finalize(self):
        self.stream.stop()



async def on_bar(bar):
    print("BAR", bar.symbol, bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume)
    pd = PriceData.from_dict(bar.model_dump())

async def on_trade(trade):
    print("TRADE", trade.symbol, trade.timestamp, trade.price, trade.size)

async def on_quote(quote):
    print("QUOTE", quote.symbol, quote.timestamp, quote.bid_price, quote.ask_price)

