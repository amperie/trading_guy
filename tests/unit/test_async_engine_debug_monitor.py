import asyncio

import pytest

from trading.core.classes import TickResults
from trading.engines.base_engine import AsyncEngine


class _TestAsyncEngine(AsyncEngine):
    def __init__(self):
        super().__init__({"state_store": {"enabled": False}}, None, None, None, None)
        self.debug_hits = 0
        self.disconnected = False

    def _check_debug(self):
        self.debug_hits += 1
        self._debug_requested = False

    async def _connect(self):
        await asyncio.sleep(0.3)

    async def _disconnect(self):
        self.disconnected = True

    def on_tick(self, tick):
        return TickResults(orders=[])


@pytest.mark.anyio
async def test_debug_monitor_handles_ctrl_c_without_ticks():
    eng = _TestAsyncEngine()
    task = asyncio.create_task(eng.start())
    await asyncio.sleep(0.05)
    eng._debug_requested = True
    await asyncio.sleep(0.2)
    assert eng.debug_hits == 1
    await task
