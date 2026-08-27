from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

from algo_crucible.config import resolve_config_dicts
from algo_crucible.jobs import CrucibleJob, RayJobRunner
from algo_crucible.state_store import LocalCrucibleStateStore


def _resolved(run_name: str = "jobs_v1"):
    return resolve_config_dicts(
        {
            "crucible": {"run_name": run_name},
            "resume": {"local_cache_dir": "unused"},
            "state_store": {"backend": "local"},
        },
        {
            "workload": {"run_name": run_name},
            "algorithm": {"algorithm": "algo.A", "params": {}},
            "portfolio": {"portfolio": "pf.P", "params": {}},
        },
    )


def _worker(payload: dict) -> dict:
    if payload.get("fail"):
        raise ValueError("forced failure")
    return {"value": payload["value"] * 2}


def test_job_id_is_deterministic():
    first = CrucibleJob("02_jobs", "synthetic", {"value": 1}).resolved()
    second = CrucibleJob("02_jobs", "synthetic", {"value": 1}).resolved()
    different = CrucibleJob("02_jobs", "synthetic", {"value": 2}).resolved()

    assert first.job_id == second.job_id
    assert first.job_id != different.job_id


def test_completed_job_results_are_reused_on_resume(tmp_path: Path):
    store = LocalCrucibleStateStore(tmp_path / "runs")
    resolved = _resolved()
    run = store.start_or_resume(resolved)
    runner = RayJobRunner(use_ray=False)
    first_two = [CrucibleJob("02_jobs", "synthetic", {"value": i}) for i in range(2)]
    all_five = [CrucibleJob("02_jobs", "synthetic", {"value": i}) for i in range(5)]

    first = runner.run_jobs(run_id=run["crucible_run_id"], jobs=first_two, worker=_worker, state_store=store)
    resumed = runner.run_jobs(run_id=run["crucible_run_id"], jobs=all_five, worker=_worker, state_store=store)

    result_files = list((Path(run["run_dir"]) / "stages" / "02_jobs" / "results").glob("*.json"))
    assert first.jobs_complete == 2
    assert resumed.jobs_total == 5
    assert resumed.jobs_complete == 5
    assert resumed.jobs_reused == 2
    assert len(result_files) == 5


def test_failed_jobs_are_rerun_when_policy_allows(tmp_path: Path):
    store = LocalCrucibleStateStore(tmp_path / "runs")
    resolved = _resolved("jobs_fail_v1")
    run = store.start_or_resume(resolved)
    runner = RayJobRunner(use_ray=False)
    failing = CrucibleJob("02_jobs", "synthetic", {"value": 1, "fail": True})
    fixed_same_id = CrucibleJob("02_jobs", "synthetic", {"value": 1, "fail": False}, failing.resolved().job_id)

    failed = runner.run_jobs(run_id=run["crucible_run_id"], jobs=[failing], worker=_worker, state_store=store)
    rerun = runner.run_jobs(run_id=run["crucible_run_id"], jobs=[fixed_same_id], worker=_worker, state_store=store)

    assert failed.jobs_failed == 1
    assert rerun.jobs_complete == 1
    assert rerun.jobs_reused == 0
    assert rerun.results[0]["result"]["value"] == 2


def test_failed_jobs_are_reused_when_rerun_disabled(tmp_path: Path):
    store = LocalCrucibleStateStore(tmp_path / "runs")
    resolved = _resolved("jobs_no_rerun_v1")
    run = store.start_or_resume(resolved)
    runner = RayJobRunner(use_ray=False)
    failing = CrucibleJob("02_jobs", "synthetic", {"value": 1, "fail": True})

    failed = runner.run_jobs(run_id=run["crucible_run_id"], jobs=[failing], worker=_worker, state_store=store)
    reused = runner.run_jobs(
        run_id=run["crucible_run_id"],
        jobs=[failing],
        worker=_worker,
        state_store=store,
        rerun_failed_jobs=False,
    )

    assert failed.jobs_failed == 1
    assert reused.jobs_failed == 1
    assert reused.jobs_reused == 1


def test_ray_path_uses_remote_worker(monkeypatch, tmp_path: Path):
    calls = {"init": 0, "remote": 0}

    class FakeRemote:
        def __init__(self, fn):
            self.fn = fn

        def remote(self, *args):
            calls["remote"] += 1
            return self.fn(*args)

    fake_ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **_: calls.__setitem__("init", calls["init"] + 1),
        remote=lambda fn: FakeRemote(fn),
        get=lambda refs: refs,
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    store = LocalCrucibleStateStore(tmp_path / "runs")
    run = store.start_or_resume(_resolved("jobs_ray_v1"))

    result = RayJobRunner(use_ray=True, max_concurrent_jobs=2).run_jobs(
        run_id=run["crucible_run_id"],
        jobs=[CrucibleJob("02_jobs", "synthetic", {"value": i}) for i in range(3)],
        worker=_worker,
        state_store=store,
    )

    assert calls == {"init": 1, "remote": 3}
    assert result.jobs_complete == 3
