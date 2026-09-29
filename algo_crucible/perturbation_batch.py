"""Balanced bounded plans and streaming downstream perturbation evidence."""
import csv
from collections import defaultdict
from pathlib import Path

from algo_crucible.monte_carlo import return_stream_rows, signal_forward_return_rows
from algo_crucible.progress import emit


def bounded_plan(jobs, limit=2500, min_windows=1):
    if limit < 1 or min_windows < 1:
        raise ValueError('Perturbation limits must be positive')
    groups = defaultdict(list)
    for job in jobs:
        groups[(job.payload['source_candidate_id'], job.payload['scenario_id'])].append(job)
    groups = {key: sorted(value, key=lambda job: job.payload['window']['window_id'])
              for key, value in sorted(groups.items())}
    counts = {key: min(min_windows, len(value)) for key, value in groups.items()}
    if sum(counts.values()) > limit:
        raise ValueError('Perturbation budget cannot cover every candidate/scenario minimum window count')
    remaining = min(limit, len(jobs)) - sum(counts.values())
    while remaining:
        for key, value in groups.items():
            if counts[key] < len(value) and remaining:
                counts[key] += 1
                remaining -= 1
    selected, coverage = [], []
    for key, value in groups.items():
        count = counts[key]
        indices = [(len(value)-1)//2] if count == 1 else [i*(len(value)-1)//(count-1) for i in range(count)]
        chosen = [value[i] for i in indices]
        selected.extend(chosen)
        coverage.append(dict(candidate_id=key[0], scenario_id=key[1], available_windows=len(value),
                             selected_windows=len(chosen), window_ids=[j.payload['window']['window_id'] for j in chosen]))
    return selected, dict(method='balanced_evenly_spaced_windows_v1', max_jobs=limit,
        planned_jobs=len(jobs), selected_jobs=len(selected), omitted_jobs=len(jobs)-len(selected), coverage=coverage)


def write_streams(store, run_id, run_dir, results, accepted):
    folder = Path(run_dir)/'stages/07_perturbation/summaries'
    folder.mkdir(parents=True, exist_ok=True)
    specs = [
        ('perturbation_return_stream', return_stream_rows,
         ['candidate_id','scenario_id','window_id','step','timestamp','return_pct','equity']),
        ('perturbation_signal_forward_returns', signal_forward_return_rows,
         ['candidate_id','scenario_id','window_id','step','timestamp','symbol','signal','signal_type',
          'signal_strength','forward_timestamp','forward_return_pct','raw_forward_return_pct','horizon_bars']),
    ]
    paths = {}
    for name, convert, columns in specs:
        path = folder/(name+'.csv')
        temporary = path.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for index, row in enumerate(results, 1):
                payload = row.get('result') or {}
                if row.get('status') == 'complete' and str(payload.get('source_candidate_id') or payload.get('candidate_id')) in accepted:
                    full = store.read_artifact_json(run_id, f"stages/{row['stage']}/results/{row['job_id']}.json")
                    if not full:
                        raise ValueError('Missing persisted perturbation evidence')
                    writer.writerows(convert([full], accepted))
                    del full
                if index % 25 == 0 or index == len(results):
                    emit(name, 'Streaming saved perturbation evidence', index, len(results))
        temporary.replace(path)
        if hasattr(store, 'log_existing_artifact'):
            store.log_existing_artifact(run_id, path, artifact_path='stages/07_perturbation/summaries')
        paths[name] = str(path)
    return paths
