import json
from types import SimpleNamespace

import pytest
import ray

from algo_crucible import progress
from algo_crucible.regime_inputs import capture
from algo_crucible.regime_jobs import run_windows
from algo_crucible.state_store import LocalCrucibleStateStore
from tests.unit.test_market_regime import bar, detector_cfg


def test_ray_matches_serial_and_reuses_completed_windows(tmp_path, monkeypatch):
    sources = []
    for window in range(3):
        ticks = [[bar(100+i % (7+window), i)] for i in range(100)]
        portfolio = SimpleNamespace(value_history={tick[0].timestamp: 1000+i for i, tick in enumerate(ticks)})
        inputs = capture(portfolio, ticks, detector_cfg(), [])
        path = tmp_path / f'{window}.json'
        path.write_text(json.dumps(dict(result=dict(regime_inputs=inputs, candidate_id='candidate', window_id=str(window)))))
        sources.append(dict(path=path, baseline=False))
    store = LocalCrucibleStateStore(tmp_path/'state')
    serial = run_windows(sources, store, 'serial', use_ray=False)
    events = []
    monkeypatch.setattr(progress, 'reporter', lambda *event: events.append(event))
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    try:
        parallel = run_windows(sources, store, 'parallel', max_concurrent_jobs=2)
        assert parallel == serial
        assert any('2 in flight' in event[1] for event in events)
        assert events[-1][2:] == (3, 3)
        # Cache reads bypass the computation, even when the worker would fail.
        def forbidden(*args, **kwargs):
            raise AssertionError('Cached window was recomputed')
        monkeypatch.setattr('algo_crucible.regime_jobs.score_window', forbidden)
        assert run_windows(sources, store, 'parallel', use_ray=False) == serial
        monkeypatch.undo()
        invalid = tmp_path/'invalid.json'
        invalid.write_text(json.dumps(dict(result=dict(regime_inputs={}, candidate_id='candidate', window_id='broken'))))
        with pytest.raises(RuntimeError, match='Regime window broken failed'):
            run_windows([dict(path=invalid, baseline=False)], store, 'failed')
        assert not list((tmp_path/'state'/'failed').rglob('*.json'))
    finally:
        ray.shutdown()
