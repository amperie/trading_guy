from trading.engines.alpaca_engine import _backoff_delay_seconds


def test_backoff_doubles_then_caps_at_one_minute():
    assert _backoff_delay_seconds(1) == 1
    assert _backoff_delay_seconds(2) == 2
    assert _backoff_delay_seconds(3) == 4
    assert _backoff_delay_seconds(6) == 32
    assert _backoff_delay_seconds(7) == 60
    assert _backoff_delay_seconds(20) == 60
