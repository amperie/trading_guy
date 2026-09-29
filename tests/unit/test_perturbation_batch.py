import csv
import tracemalloc

import pytest

from algo_crucible.jobs import CrucibleJob, RayJobRunner
from algo_crucible.perturbation_batch import bounded_plan, write_streams
from algo_crucible.state_store import LocalCrucibleStateStore


def test_balanced_plan():
    jobs = [CrucibleJob('07_perturbation', 'backtest', dict(source_candidate_id=c, scenario_id=s,
            window={'window_id': f'{w:04d}'})) for c in range(3) for s in range(20) for w in range(113)]
    selected, plan = bounded_plan(jobs)
    assert len(selected) == 2500
    assert plan['planned_jobs'] == 6780
    assert bounded_plan(list(reversed(jobs))) == (selected, plan)
    assert len(plan['coverage']) == 60
    for group in plan['coverage']:
        assert group['selected_windows'] in (41, 42)
        assert group['window_ids'][0] == '0000'
        assert group['window_ids'][-1] == '0112'
    assert len(bounded_plan(jobs[:10])[0]) == 10
    with pytest.raises(ValueError, match='budget'):
        bounded_plan(jobs, 50)


def evidence(payload):
    return dict(source_candidate_id='candidate', scenario_id='scenario', window_id=str(payload['n']),
                overall_scorecard={'score': 1},
                return_stream=[{'step': i, 'return_pct': i / 1000} for i in range(100)],
                signal_forward_return_stream=[{'step': i, 'forward_return_pct': i / 1000} for i in range(100)])


def test_2500_results_are_disk_backed_and_streamed(tmp_path):
    store = LocalCrucibleStateStore(tmp_path)
    jobs = [CrucibleJob('07_perturbation', 'backtest', {'n': i}) for i in range(2500)]
    runner = RayJobRunner(use_ray=False)
    tracemalloc.start()
    try:
        batch = runner.run_jobs(run_id='run', jobs=jobs, worker=evidence, state_store=store, compact_results=True)
        assert batch.jobs_complete == 2500
        assert all('return_stream' not in row['result'] for row in batch.results)
        paths = write_streams(store, 'run', tmp_path/'run', batch.results, {'candidate'})
        assert tracemalloc.get_traced_memory()[1] < 40 * 1024**2
    finally:
        tracemalloc.stop()
    for path in paths.values():
        with open(path, newline='') as stream:
            assert sum(1 for _ in csv.DictReader(stream)) == 250000
    reused = runner.run_jobs(run_id='run', jobs=jobs[:2], worker=lambda _: pytest.fail('reran saved job'),
                             state_store=store, compact_results=True)
    assert reused.jobs_reused == 2
    assert not hasattr(runner, '_result_sink')


def test_ray_compact_results(tmp_path):
    pytest.importorskip('ray')
    batch = RayJobRunner(use_ray=True, max_concurrent_jobs=2).run_jobs(run_id='run',
        jobs=[CrucibleJob('07_perturbation', 'backtest', {'n': i}) for i in range(3)],
        worker=evidence, state_store=LocalCrucibleStateStore(tmp_path), compact_results=True)
    assert batch.jobs_complete == 3
    assert all('return_stream' not in row['result'] for row in batch.results)
    assert len(list(tmp_path.glob('run/stages/07_perturbation/results/*.json'))) == 3
