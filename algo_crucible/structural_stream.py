"""Columnar structural analysis: partition CSVs on disk, retain one candidate's numeric arrays."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from algo_crucible.progress import emit
from algo_crucible.structural_break import _ratio_threshold, _summary_row, analyze_structural_breaks


def analyze_file(run_dir, platform, on_result=None):
    root = Path(run_dir)
    source = root/'stages/07_perturbation/summaries/perturbation_signal_forward_returns.csv'
    cfg = platform.get('structural_break_stability', {})
    signal, returns = cfg.get('signal_column', 'signal'), cfg.get('forward_return_column', 'forward_return_pct')
    destination = root/'stages/08_structural_break_stability/summaries/structural_break_windows.csv'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    summaries, observations = [], 0
    fields = {'candidate_id', 'timestamp', 'window_id', 'step', signal, returns}
    with TemporaryDirectory(dir=root, prefix='structural-') as temporary:
        paths = {}
        try:
            chunks = pd.read_csv(source, usecols=lambda key: key in fields, chunksize=25000,
                                 dtype={'candidate_id': str, 'timestamp': str, 'window_id': str})
            for frame in chunks:
                if not {signal, returns, 'candidate_id'}.issubset(frame.columns):
                    break
                observations += len(frame)
                for candidate, group in frame.groupby('candidate_id', sort=False):
                    path = paths.setdefault(candidate, Path(temporary)/f'{len(paths)}.csv')
                    group.to_csv(path, mode='a', header=not path.exists(), index=False)
                emit('structural_loading', 'Partitioning signal evidence', observations, None)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass
        if not paths:
            fallback = root/'stages/07_perturbation/summaries/perturbation_return_stream.csv'
            try:
                # Partition on disk before the legacy per-candidate return calculation.
                for frame in pd.read_csv(fallback, chunksize=25000, dtype={'candidate_id': str}):
                    observations += len(frame)
                    for candidate, group in frame.groupby('candidate_id', sort=False):
                        path = paths.setdefault(candidate, Path(temporary)/f'{len(paths)}.csv')
                        group.to_csv(path, mode='a', header=not path.exists(), index=False)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                pass
            for candidate, path in paths.items():
                result = analyze_structural_breaks(pd.read_csv(path, dtype={'candidate_id': str}).to_dict('records'), platform)
                summaries.extend(result['summary_rows'])
                append_frame(destination, pd.DataFrame(result['window_rows']))
                if on_result:
                    for row in result['summary_rows']:
                        on_result(row)
        else:
            for index, (candidate, path) in enumerate(paths.items(), 1):
                frame = pd.read_csv(path, dtype={'timestamp': str, 'window_id': str}).fillna({'timestamp': '', 'window_id': ''})
                for key in (signal, returns):
                    frame[key] = pd.to_numeric(frame[key], errors='coerce')
                frame = frame[np.isfinite(frame[signal]) & np.isfinite(frame[returns])].copy()
                for key, default in [('timestamp',''), ('window_id',''), ('step',0)]:
                    if key not in frame:
                        frame[key] = default
                frame['step'] = pd.to_numeric(frame['step'], errors='coerce').fillna(0).astype(int)
                frame.sort_values(['timestamp','window_id','step'], kind='stable', inplace=True)
                frame.reset_index(drop=True, inplace=True)
                row = analyze_candidate(candidate, frame, signal, returns, cfg, destination)
                summaries.append(row)
                if on_result:
                    on_result(row)
                emit('structural_candidates', 'Completed structural candidate analysis', index, len(paths))
    if not destination.exists():
        destination.write_text('candidate_id,analysis_mode,ic\n', encoding='utf-8')
    return dict(summary_rows=summaries, windows_path=str(destination), observation_count=observations)


def append_frame(path, frame):
    if not frame.empty:
        frame.to_csv(path, mode='a', header=not path.exists(), index=False)


def analyze_candidate(candidate, frame, signal, returns, cfg, destination):
    count = len(frame)
    if count < int(cfg.get('min_observations',20)):
        return _summary_row(candidate, count, False, 'insufficient_observations', 'rolling_ic')
    values = []
    for window in [int(w) for w in cfg.get('ic_windows',[63,126]) if int(w)>1]:
        total = max(0, count-window+1)
        if not total:
            continue
        correlation = frame[signal].rolling(window).corr(frame[returns]).iloc[window-1:].to_numpy()
        correlation[~np.isfinite(correlation)] = np.nan
        values.append(correlation[np.isfinite(correlation)])
        for offset in range(0,total,25000):
            end = min(offset+25000,total)
            append_frame(destination,pd.DataFrame(dict(candidate_id=candidate,analysis_mode='rolling_ic',
                ic_window=window,start_index=np.arange(offset,end),end_index=np.arange(offset,end)+window-1,
                start_timestamp=frame['timestamp'].iloc[offset:end].to_numpy(),
                end_timestamp=frame['timestamp'].iloc[offset+window-1:end+window-1].to_numpy(),
                observation_count=window,ic=correlation[offset:end])))
            emit('rolling_ic', f'Calculating rolling IC candidate={candidate} window={window}',end,total)
    ic = np.concatenate(values) if values else np.array([])
    if len(ic)<int(cfg.get('min_ic_observations',3)):
        return _summary_row(candidate,count,False,'insufficient_ic_windows','rolling_ic')
    mean, std = float(ic.mean()), float(ic.std(ddof=1)) if len(ic)>1 else 0.0
    ratio = mean/std if std>0 else float('inf') if mean>0 else 0.0
    positive = float((ic>0).mean())
    recent = float(ic[-int(cfg.get('recent_window_count',3)):].mean())
    split, post, degradation = None, float(ic[0]), 0.0
    if len(ic)>1:
        prefix = np.cumsum(ic)[:-1] / np.arange(1,len(ic))
        suffix = np.cumsum(ic[::-1])[-2::-1] / np.arange(len(ic)-1,0,-1)
        differences = np.maximum(0,prefix-suffix)
        split = int(np.argmax(differences))
        post, degradation = float(suffix[split]), float(differences[split])
        split += 1
    reasons = []
    for failed, reason in [
        (mean<float(cfg.get('min_ic_mean',0)), 'rolling_ic_mean_below_gate'),
        (ratio<float(cfg.get('min_ic_ir',0)), 'rolling_ic_ir_below_gate'),
        (positive<_ratio_threshold(cfg.get('min_positive_ic_window_pct',.60)), 'positive_ic_window_rate_below_gate'),
        (recent<float(cfg.get('recent_min_ic_mean',0)), 'recent_ic_sign_flip'),
        (post<float(cfg.get('min_post_break_ic_mean',0)), 'post_break_ic_mean_below_gate'),
        (degradation>float(cfg.get('max_post_break_ic_degradation',.50)), 'post_break_ic_degradation_above_gate'),
    ]:
        if failed:
            reasons.append(reason)
    return dict(_summary_row(candidate,count,not reasons,','.join(reasons),'rolling_ic'),
                ic_window_count=len(ic),ic_mean=mean,ic_ir=ratio,positive_ic_window_pct=positive*100,
                recent_ic_mean=recent,break_after_ic_window=split,post_break_ic_mean=post,
                post_break_ic_degradation=degradation)
