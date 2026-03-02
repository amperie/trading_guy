import os
from alpaca.data.live.stock import StockDataStream
from alpaca.data.enums import \
    DataFeed  # IEX or SIP

from dotenv import load_dotenv

load_dotenv()

async def nothing(bar):
    pass

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

stream = StockDataStream(
    api_key=api_key,
    secret_key=secret_key,
    url_override="ws://hp.lan:8765",
    feed=DataFeed.SIP,
    raw_data=False
)

stream.subscribe_bars(nothing, *["SPY"])
stream.run()
pass