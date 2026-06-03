from trading.data_providers import alpaca_data_provider as module
from trading.data_providers.alpaca_data_provider import AlpacaDataProvider


def test_alpaca_data_provider_uses_dedicated_data_account(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, api_key, secret_key, **kwargs):
            captured.update(api_key=api_key, secret_key=secret_key, kwargs=kwargs)

    class FakePath:
        def __init__(self, value):
            self.value = value

        def is_file(self):
            return self.value == "accounts.yaml"

        def read_text(self, encoding=None):
            return """
paper:
  api_key: trade-key
  secret_key: trade-secret
AlpacaAPIAccount:
  api_key: data-key
  secret_key: data-secret
"""

    monkeypatch.setattr(module, "Path", FakePath)
    monkeypatch.setattr(module, "StockHistoricalDataClient", FakeClient)

    dp = AlpacaDataProvider({
        "api_key": "cfg-key",
        "secret_key": "cfg-secret",
        "symbols": ["SPY"],
        "timeframe": "Minute",
    })

    assert dp.api_key == "data-key"
    assert dp.secret_key == "data-secret"
    assert captured["api_key"] == "data-key"
    assert captured["secret_key"] == "data-secret"


def test_alpaca_data_provider_falls_back_to_config_creds(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, api_key, secret_key, **kwargs):
            captured.update(api_key=api_key, secret_key=secret_key)

    monkeypatch.setattr(AlpacaDataProvider, "_data_account_creds", staticmethod(lambda: None))
    monkeypatch.setattr(module, "StockHistoricalDataClient", FakeClient)

    dp = AlpacaDataProvider({
        "api_key": "cfg-key",
        "secret_key": "cfg-secret",
        "symbols": ["SPY"],
        "timeframe": "Minute",
    })

    assert dp.api_key == "cfg-key"
    assert dp.secret_key == "cfg-secret"
    assert captured == {"api_key": "cfg-key", "secret_key": "cfg-secret"}
