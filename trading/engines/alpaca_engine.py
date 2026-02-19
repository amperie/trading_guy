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
from trading.engines.base_engine import BaseEngine, AsyncEngine

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

class AlpacaRealTimeEngine(AsyncEngine):

    """
    Live streaming engine that connects to Alpaca's WebSocket API for real-time
    market data (bars, quotes, trades) and routes it through the standard
    algo -> portfolio -> order manager pipeline via AsyncEngine's queue.

    Subscribes to one or more symbols and converts Alpaca bar/quote/trade events
    into PriceData objects, which are enqueued via submit_tick() for non-blocking
    processing in a background thread.

    On startup (_connect), the engine synchronizes the Portfolio's cash and
    positions from the Alpaca broker before streaming begins. This recovers
    positions from previous sessions and ensures the Portfolio starts with
    accurate state. Periodic syncs during streaming are controlled by the
    Portfolio's sync_with_broker and sync_interval config.

    Can operate in two modes:
        - Full pipeline: provide al, om, pf to run signals and execute orders.
        - Data streamer: omit al/om/pf to just receive and forward market data
          to a downstream_engine.

    Config (cfg dict):
        api_key (str):                Alpaca API key (required)
        secret_key (str):             Alpaca secret key (required)
        symbols_to_subscribe (list):  Symbols to stream (required, e.g. ["AAPL", "SPY"])
        subscribe_to_bars (bool):     Subscribe to minute bars (default: True)
        subscribe_to_quotes (bool):   Subscribe to bid/ask quotes (default: True)
        subscribe_to_trades (bool):   Subscribe to individual trades (default: False)
        override_url (str|None):      Custom WebSocket URL for testing
                                      (e.g. "wss://stream.data.alpaca.markets/v2/test")
        downstream_engine (str|None): Engine to forward ticks to (not yet implemented)
        warmup (dict|None):           Optional config to pre-populate the algorithm's
            price history from historical bars before streaming begins. Keys:
            - provider (str): Dotted path to a DataProvider class (required).
                Example: "data_providers.alpaca_data_provider.AlpacaDataProvider"
            - limit (int): Max bars to fetch (optional). Auto-set to the
                algorithm's history_length if omitted and history_length > 0.
            - Any other keys are passed through to the DataProvider as cfg.
            Example:
                warmup:
                  provider: "data_providers.alpaca_data_provider.AlpacaDataProvider"
                  api_key: "PK..."
                  secret_key: "..."
                  symbols: ["AAPL"]
                  timeframe: "Minute"
                  # limit auto-derived from algorithm's history_length

    Startup behavior:
        1. Creates StockDataStream with provided credentials
        2. Subscribes to bars/trades/quotes per config
        3. Calls Portfolio._sync_from_broker() to pull current Alpaca
           account state (cash, positions) before any ticks arrive
        4. Warms up algorithm history from historical data (if warmup configured)
        5. Starts the WebSocket stream

    Testing:
        Use override_url="wss://stream.data.alpaca.markets/v2/test" with symbol "FAKEPACA"
        to stream simulated data without market hours.

    Example:
        cfg = {
            "api_key": "PK...",
            "secret_key": "PS...",
            "symbols_to_subscribe": ["AAPL", "SPY"],
            "subscribe_to_bars": True,
        }
        engine = AlpacaRealTimeEngine(cfg=cfg, al=algo, om=om, pf=portfolio)
        engine.run()  # blocks, streaming bars through the pipeline
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
        self.stream = None

    async def _connect(self):
        # Open the connection to Alpaca
        if self._override_url is not None:
            stream = StockDataStream(
                api_key=self._api_key,
                secret_key=self._secret_key,
                url_override=self._override_url,
                feed=DataFeed.IEX,
                # or DataFeed.SIP if you have SIP access
                raw_data=False
            )
        else:
            stream = StockDataStream(
                api_key=self._api_key,
                secret_key=self._secret_key,
                feed=DataFeed.IEX,
                # or DataFeed.SIP if you have SIP access
                raw_data=False
            )
        self.stream = stream

        if self._bar_subscribe:
            stream.subscribe_bars(self._handle_bar, *self._symbols_to_subscribe)
        if self._trades_subscribe:
            stream.subscribe_trades(self._handle_trade, *self._symbols_to_subscribe)
        if self._quote_subscribe:
            stream.subscribe_quotes(self._handle_quote, *self._symbols_to_subscribe)

        # Sync portfolio state from broker before streaming starts
        if self.pf is not None:
            self.pf._sync_from_broker()
        if self.om is not None and hasattr(self.om, 'sync_orders_from_broker'):
            order_sync_limit = self.pf.cfg.get("order_sync_limit", 0)
            self.om.sync_orders_from_broker(limit=order_sync_limit)

        # Warm up algorithm history from historical data if configured
        warmup_cfg = self.cfg.get("warmup", None)
        if warmup_cfg is not None and self.al is not None:
            from utils.utils import instantiate_from_string
            warmup_cfg = dict(warmup_cfg)  # don't mutate original
            provider_path = warmup_cfg.pop("provider")
            # Default limit to the algorithm's history_length so we fetch
            # exactly the number of bars needed to fill the deque.
            if "limit" not in warmup_cfg and self.al.history_length > 0:
                warmup_cfg["limit"] = self.al.history_length
            dp = instantiate_from_string(provider_path, cfg=warmup_cfg)
            self.warm_up(dp)

        # stream.run() is a sync wrapper around asyncio.run(_run_forever()),
        # which can't be called from an already-running event loop.
        # Await the underlying coroutine directly since we're already async.
        await self.stream._run_forever()

    async def _disconnect(self):
        self.stream.stop()

    async def _handle_bar(self, bar):
        print("BAR", bar.symbol, bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume)
        pd = PriceData.from_dict(bar.model_dump())
        await self.submit_tick([pd])

    async def _handle_quote(self, quote):
        print("QUOTE", quote.symbol, quote.timestamp, quote.bid_price, quote.ask_price)
        raise NotImplementedError()

    async def _handle_trade(self, trade):
        print("TRADE", trade.symbol, trade.timestamp, trade.price, trade.size)
        raise NotImplementedError()

    def on_tick(self, tick: list[PriceData]) -> TickResults:
        market_signals = self.al.on_data(tick)
        return self.pf.process_market_signals_for_tick(market_signals, tick)


