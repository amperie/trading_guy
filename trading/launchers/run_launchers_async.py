import os
from dotenv import load_dotenv

load_dotenv()


def run_alpaca_fakepaca():
    """
    Start the AlpacaRealTimeEngine streaming FAKEPACA from the Alpaca test
    WebSocket, running TestAlgorithm + SingleSymbolPortfolio.

    FAKEPACA is Alpaca's simulated symbol that streams data outside market
    hours, useful for end-to-end testing of the live pipeline.
    """
    from trading.core.algorithms.test_algorithm import TestAlgorithm
    from trading.core.pf.single_symbol_portfolio import SingleSymbolPortfolio
    from trading.core.om.alpaca_om import AlpacaOrderManager
    from trading.engines.alpaca_engine import AlpacaRealTimeEngine

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
        )

    symbol = "UPRO"

    alg_cfg = {
        "history_length": 10,
    }

    pf_cfg = {
        "symbol": symbol,
        "stop_pct": 1.0,
        "profit_pct": 1.0,
        "sync_with_broker": True,
        "sync_interval": 1,
        "order_sync_limit": 10,
    }

    engine_cfg = {
        "api_key": api_key,
        "secret_key": secret_key,
        "symbols_to_subscribe": [symbol],
        "subscribe_to_bars": True,
        "subscribe_to_quotes": False,
        "subscribe_to_trades": False,
        # "override_url": "wss://stream.data.alpaca.markets/v2/test",
    }

    om_cfg = {
        "api_key": api_key,
        "secret_key": secret_key,
        "paper": True,
    }
    om = AlpacaOrderManager(om_cfg)
    alg = TestAlgorithm(alg_cfg)
    pf = SingleSymbolPortfolio(pf_cfg, om, 1000.0, {}, True)

    engine = AlpacaRealTimeEngine(cfg=engine_cfg, al=alg, om=om, pf=pf)

    print("Starting AlpacaRealTimeEngine with FAKEPACA test stream")
    print(f"  Algorithm:  TestAlgorithm")
    print(f"  Portfolio:  SingleSymbolPortfolio (cash=$1000)")
    print(f"  Symbol:     {symbol}")
    # print(f"  Stream:     {engine_cfg['override_url']}")
    print("Press Ctrl+C to stop.\n")

    engine.run()


if __name__ == "__main__":
    run_alpaca_fakepaca()
