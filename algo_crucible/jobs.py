from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from algo_crucible.ids import hash16
from utils.logger import Logger

logger = Logger().get_logger(__name__)


@dataclass(frozen=True)
class CrucibleJob:
    stage: str
    kind: str
    payload: dict[str, Any]
    job_id: str = ""

    def resolved(self) -> "CrucibleJob":
        if self.job_id:
            return self
        return CrucibleJob(self.stage, self.kind, self.payload, f"job_{hash16(self.identity_payload())}")

    def identity_payload(self) -> dict[str, Any]:
        return {"stage": self.stage, "kind": self.kind, "payload": self.payload}

    def manifest(self) -> dict[str, Any]:
        job = self.resolved()
        return {
            "job_id": job.job_id,
            "stage": job.stage,
            "kind": job.kind,
            "payload": job.payload,
            "payload_hash": hash16(job.payload),
        }


@dataclass
class JobBatchResult:
    jobs_total: int
    jobs_complete: int
    jobs_failed: int
    jobs_reused: int
    results: list[dict[str, Any]] = field(default_factory=list)


class RayJobRunner:
    def __init__(self, *, use_ray: bool = True, max_concurrent_jobs: int | None = None):
        self.use_ray = use_ray
        self.max_concurrent_jobs = max_concurrent_jobs

    def run_jobs(
        self,
        *,
        run_id: str,
        jobs: list[CrucibleJob],
        worker: Callable[[dict[str, Any]], dict[str, Any]],
        state_store,
        rerun_failed_jobs: bool = True,
    ) -> JobBatchResult:
        resolved = [job.resolved() for job in jobs]
        pending: list[CrucibleJob] = []
        reused: list[dict[str, Any]] = []
        for job in resolved:
            existing = state_store.read_artifact_json(run_id, _result_path(job))
            if existing and (existing.get("status") == "complete" or not rerun_failed_jobs):
                reused.append(existing)
                continue
            state_store.write_artifact_json(run_id, _manifest_path(job), job.manifest())
            pending.append(job)

        logger.info(
            f"Running crucible jobs stage={resolved[0].stage if resolved else 'n/a'} "
            f"total={len(resolved)} reused={len(reused)} pending={len(pending)}"
        )
        fresh = self._run_ray(pending, worker) if self.use_ray and pending else self._run_local(pending, worker)
        for result in fresh:
            state_store.write_artifact_json(run_id, _result_path_from_dict(result), result)
        results = sorted([*reused, *fresh], key=lambda row: row["job_id"])
        failed = [row for row in results if row.get("status") == "failed"]
        logger.info(
            f"Completed crucible jobs stage={resolved[0].stage if resolved else 'n/a'} "
            f"complete={sum(1 for row in results if row.get('status') == 'complete')} "
            f"failed={len(failed)} reused={len(reused)}"
        )
        for row in failed[:10]:
            logger.warning(
                f"Crucible job failed stage={row.get('stage')} job_id={row.get('job_id')} "
                f"error_type={row.get('error_type')} error={row.get('error')}"
            )
        return JobBatchResult(
            jobs_total=len(resolved),
            jobs_complete=sum(1 for row in results if row.get("status") == "complete"),
            jobs_failed=sum(1 for row in results if row.get("status") == "failed"),
            jobs_reused=len(reused),
            results=results,
        )

    def _run_local(self, jobs: list[CrucibleJob], worker) -> list[dict[str, Any]]:
        return [_run_one(job, worker) for job in jobs]

    def _run_ray(self, jobs: list[CrucibleJob], worker) -> list[dict[str, Any]]:
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
        remote_worker = ray.remote(_run_one)
        results = []
        limit = self.max_concurrent_jobs or len(jobs) or 1
        for offset in range(0, len(jobs), limit):
            refs = [remote_worker.remote(job, worker) for job in jobs[offset : offset + limit]]
            results.extend(ray.get(refs))
        return results


def _run_one(job: CrucibleJob, worker) -> dict[str, Any]:
    started = time.time()
    try:
        payload = worker(job.payload)
        return {
            "job_id": job.job_id,
            "stage": job.stage,
            "kind": job.kind,
            "status": "complete",
            "duration_seconds": time.time() - started,
            "result": payload,
        }
    except Exception as exc:
        return {
            "job_id": job.job_id,
            "stage": job.stage,
            "kind": job.kind,
            "status": "failed",
            "duration_seconds": time.time() - started,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _manifest_path(job: CrucibleJob) -> str:
    return f"stages/{job.stage}/manifests/{job.job_id}.json"


def _result_path(job: CrucibleJob) -> str:
    return f"stages/{job.stage}/results/{job.job_id}.json"


def _result_path_from_dict(result: dict[str, Any]) -> str:
    return f"stages/{result['stage']}/results/{result['job_id']}.json"
