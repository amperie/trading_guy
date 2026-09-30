"""Bounded, independent regime windows. Only the driver reads/writes checkpoint files."""
import hashlib
import json
import logging
import time

from . import progress
from .regime_inputs import score

log = logging.getLogger(__name__)
admission_factory = None  # Optional host policy; standalone engine behavior stays unchanged.


class WindowProgress:
    def __init__(self):
        self.values = {}

    def update(self, key, value):
        self.values[key] = value

    def snapshot(self):
        return self.values


def score_window(item, key, sink=None):
    original = progress.reporter
    last = 0
    def report(phase, message, completed=None, total=None):
        nonlocal last
        now = time.monotonic()
        if sink is not None and (now-last >= 1 or completed is None or completed in (0, total)):
            sink.update.remote(key, dict(phase=phase, completed=completed, total=total))
            last = now
        elif sink is None and original:
            original(phase, message, completed, total)
    progress.reporter = report
    started = time.monotonic()
    log.info('Regime window starting window=%s', key)
    try:
        rows = score(item['inputs'])
        log.info('Regime window completed window=%s elapsed_seconds=%.3f', key, time.monotonic()-started)
        return dict(rows=rows, baseline=item['baseline'], candidate_id=item.get('candidate_id'), window_id=item.get('window_id'))
    except Exception:
        log.exception('Regime window failed window=%s elapsed_seconds=%.3f', key, time.monotonic()-started)
        raise
    finally:
        progress.reporter = original


def run_windows(sources, store, run_id, *, use_ray=True, max_concurrent_jobs=2):
    if type(max_concurrent_jobs) is not int or max_concurrent_jobs < 1:
        raise ValueError('regime_gate.max_concurrent_jobs must be positive')
    limit = max_concurrent_jobs
    pending, results, states = {}, {}, {}
    fresh = []
    for index, source in enumerate(sources):
        digest = hashlib.sha256()
        with source['path'].open('rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                digest.update(block)
        digest.update(str(source['baseline']).encode())
        destination = f'stages/05_regime_gate/results/{digest.hexdigest()}.json'
        cached = store.read_artifact_json(run_id, destination)
        if cached is not None:
            results[index] = cached
            progress.result('regime_windows', dict(cached, reused=True))
        else:
            fresh.append((index, source, destination))
        progress.emit('regime_preparing', 'Checking completed regime windows', index+1, len(sources))
    queue = iter(fresh)
    ray = sink = remote = None
    admission = admission_factory() if use_ray and fresh and admission_factory else None
    if use_ray and fresh:
        import ray
        if not ray.is_initialized():
            ray.init(include_dashboard=False, log_to_driver=False)
        limit = min(limit, max(1, int(ray.cluster_resources().get('CPU', 1))))
        sink = ray.remote(num_cpus=0)(WindowProgress).remote()
        remote = ray.remote(num_cpus=1)(score_window)

    def emit(failed=0):
        details = '; '.join(f'{key}: {value["completed"]}/{value["total"]} observations'
            for key, value in states.items() if value.get('total') is not None)
        message = f'Regime windows: {len(results)}/{len(sources)} completed, {len(pending)} in flight, {len(states)} active workers, {failed} failed'
        if admission:
            message += '; ' + admission.message
        progress.emit('regime_windows', message + (f'; {details}' if details else ''), len(results), len(sources))

    try:
        exhausted = False
        emit()
        while len(results) < len(sources):
            if admission and not exhausted:
                limit = admission.limit(max_concurrent_jobs, len(pending))
            while not exhausted and len(pending) < limit:
                try:
                    index, source, destination = next(queue)
                except StopIteration:
                    exhausted = True
                    break
                with source['path'].open(encoding='utf-8') as stream:
                    value = json.load(stream)
                item = dict(inputs=value, baseline=True) if source['baseline'] else dict(
                    inputs=value['result']['regime_inputs'], baseline=False,
                    candidate_id=value['result']['candidate_id'], window_id=value['result']['window_id'])
                label = 'baseline' if item['baseline'] else str(item['window_id'])
                if remote is None:
                    result = score_window(item, label)
                    store.write_artifact_json(run_id, destination, result)
                    results[index] = result
                    progress.result('regime_windows', result)
                else:
                    ref = admission.submit(ray, score_window, item, label, sink) if admission else remote.remote(item, label, sink)
                    pending[ref] = (index, destination, label)
                    del item, value
                emit()
            if not pending:
                continue
            ready, _ = ray.wait(list(pending), num_returns=1, timeout=1)
            states = ray.get(sink.snapshot.remote())
            for ref in ready:
                index, destination, label = pending[ref]
                try:
                    result = ray.get(ref)
                    if admission:
                        result = admission.finish(result)
                except Exception as exc:
                    emit(failed=1)
                    raise RuntimeError(f'Regime window {label} failed') from exc
                store.write_artifact_json(run_id, destination, result)
                results[index] = result
                progress.result('regime_windows', result)
                pending.pop(ref)
            active = {item[2] for item in pending.values()}
            states = {key: value for key, value in states.items() if key in active}
            emit()
        return [results[index] for index in range(len(sources))]
    finally:
        if ray:
            for ref in pending:
                ray.cancel(ref, force=True)
            if sink:
                ray.kill(sink)
