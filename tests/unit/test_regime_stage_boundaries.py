import json
from types import SimpleNamespace

import pytest

from algo_crucible import progress
from algo_crucible.regime_inputs import capture, score
from algo_crucible.scoring import regime_scorecard
from tests.unit.test_market_regime import bar, detector_cfg
from tests.unit.test_algo_crucible_milestone1 import _configs, _write_data
from algo_crucible.orchestrator import CrucibleOrchestrator


def test_baseline_never_classifies_regimes(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Regime classification must not run in baseline')
    monkeypatch.setattr('algo_crucible.scoring.classify_ticks', forbidden)
    path = tmp_path/'prices.csv'
    _write_data(path)
    result = CrucibleOrchestrator(*_configs(tmp_path, path)).run_milestone1()
    assert result['summary']['regime_evaluation'] == 'deferred'


def test_deferred_json_preserves_scores():
    ticks = [[bar(100 + i % 7, i)] for i in range(40)]
    portfolio = SimpleNamespace(value_history={tick[0].timestamp: 1000 + i for i, tick in enumerate(ticks)})
    trades = [SimpleNamespace(entry_time=ticks[0][0].timestamp, exit_time=ticks[-1][0].timestamp)]
    inputs = json.loads(json.dumps(capture(portfolio, ticks, detector_cfg(), trades)))
    assert score(inputs) == regime_scorecard(portfolio, ticks, detector_cfg(), trades)


def test_stage_lifecycle_includes_failure(monkeypatch):
    events = []
    monkeypatch.setattr(progress, 'reporter', lambda *event: events.append(event))
    class Example:
        resolved_cfg = SimpleNamespace(crucible_run_id='test')
        @progress.stage
        def run(self):
            raise ValueError('failed fixture')
    with pytest.raises(ValueError, match='failed fixture'):
        Example().run()
    assert events[0][1] == 'Starting run'
    assert events[-1][1].startswith('Failed run after')
