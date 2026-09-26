"""Historical regime calculation with bounded rolling kernels, without history copies per bar."""
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer

from .market_regime import MarketRegimeDetector, MarketRegimeSnapshot


class Bounds(BaseIndexer):
    def __init__(self, starts):
        self.starts = np.asarray(starts, dtype=np.int64)

    def get_window_bounds(self, num_values=0, min_periods=None, center=None, closed=None, step=None):
        return self.starts, np.arange(1, num_values + 1, dtype=np.int64)


def rolling(values, starts, operation, **kwargs):
    widths = np.arange(1, len(values)+1)-starts
    sizes = np.unique(widths[starts > 0])
    if operation in ('mean', 'max', 'rank') and len(sizes) <= 8:
        # Session gaps alternate between a few horizons. Evaluate each horizon
        # once instead of repeatedly rebuilding a large expanding rank window.
        series = pd.Series(values)
        output = getattr(series.expanding(min_periods=1), operation)(**kwargs).to_numpy()
        for size in sizes:
            mask = (starts > 0) & (widths == size)
            result = getattr(series.rolling(int(size), min_periods=1), operation)(**kwargs).to_numpy() if size else np.full(len(values), np.nan)
            output[mask] = result[mask]
        return output
    # Rolling extrema kernels assume advancing left bounds. Restart when a cadence
    # change expands the window backwards (for example, the bar after an overnight gap).
    output = np.empty(len(values))
    cuts = np.r_[0, np.flatnonzero(np.diff(starts) < 0)+1, len(values)]
    for begin, end in zip(cuts[:-1], cuts[1:]):
        if begin == end:
            continue
        offset = min(int(begin), int(starts[begin]))
        bounds = np.r_[np.zeros(begin-offset, dtype=np.int64), starts[begin:end]-offset]
        result = getattr(pd.Series(values[offset:end]).rolling(Bounds(bounds), min_periods=1), operation)(**kwargs)
        output[begin:end] = result.to_numpy()[begin-offset:]
    return output


def classify(ticks, cfg, progress):
    grouped = defaultdict(list)
    for index, tick in enumerate(ticks):
        for bar in tick:
            grouped[bar.symbol].append((index, bar))
    output = [{} for _ in ticks]
    total = sum(map(len, grouped.values()))
    done = 0
    for symbol, records in grouped.items():
        detector = MarketRegimeDetector(cfg)
        bars = [bar for _, bar in records]
        n = len(bars)
        positions = np.arange(n)
        closes = np.array([bar.close for bar in bars], dtype=float)
        minutes, inferred, settings = None, [], []
        cached = {}
        keys = ('trend_lookback', 'baseline_ma_window', 'volatility_lookback',
                'volatility_percentile_window', 'drawdown_lookback')
        for i, bar in enumerate(bars):
            if i and isinstance(bar.timestamp, datetime) and isinstance(bars[i-1].timestamp, datetime):
                delta = (bar.timestamp - bars[i-1].timestamp).total_seconds() / 60
                if delta > 0:
                    minutes = delta
            inferred.append(minutes)
            if minutes not in cached:
                bpd = detector._bars_per_day_from_minutes(minutes or float(detector.cfg['default_bar_minutes']))
                cached[minutes] = (bpd, *(detector._configured_bars(k, k+'_days', k+'_hours', bpd) for k in keys))
            settings.append(cached[minutes])
        settings = np.array(settings)
        trend, baseline, volatility, percentile, drawdown = settings[:, 1:].astype(np.int64).T
        history_start = np.maximum(0, positions - detector._max_history + 1)
        mean = rolling(closes, np.maximum(history_start, positions-baseline+1), 'mean')
        peak = rolling(closes, np.maximum(history_start, positions-drawdown+1), 'max')
        returns = np.full(n, np.nan)
        valid = np.flatnonzero(closes[:-1] != 0) + 1
        returns[valid] = closes[valid] / closes[valid-1] - 1
        end = np.searchsorted(valid, positions, side='right')
        start = np.maximum(np.searchsorted(valid, history_start+1), end-volatility)
        starts = np.maximum(history_start+1, valid[np.minimum(start, len(valid)-1)] if len(valid) else positions+1)
        starts = np.minimum(starts, positions+1)
        vols = rolling(returns, starts, 'std', ddof=0)
        vols[end-start < volatility] = np.nan
        # Near-tied variances can swap percentile ranks through rounding alone.
        # Reproduce the original summation only for those ambiguous samples.
        ordered = np.flatnonzero(np.isfinite(vols) & (vols > 0))
        ordered = ordered[np.argsort(vols[ordered])]
        tied = np.abs(np.diff(vols[ordered])) <= np.maximum(vols[ordered][1:], vols[ordered][:-1])*1e-11 + 1e-18
        ambiguous = np.unique(np.r_[ordered[:-1][tied], ordered[1:][tied]])
        rank_vols = vols.copy()
        for i in ambiguous:
            sample = returns[starts[i]:i+1]
            sample = sample[~np.isnan(sample)].tolist()
            rank_vols[i] = detector._realized_volatility(sample, int(volatility[i]))
        ranks = np.full(n, np.nan)
        available = np.flatnonzero(~np.isnan(vols))
        capacity = detector._vol_history[symbol].maxlen
        rank_starts = np.maximum(0, np.arange(len(available))-np.minimum(percentile[available], capacity)+1)
        rank = rolling(rank_vols[available], rank_starts, 'rank', method='average')
        ranks[available] = 100 * (rank - .5) / (np.arange(len(available))-rank_starts+1)
        if progress:
            progress(done, total)
        for i, (index, bar) in enumerate(records):
            seen = min(i+1, detector._max_history)
            previous = i-int(trend[i])
            tr = None if previous < history_start[i] or closes[previous] == 0 else closes[i]/closes[previous]-1
            distance = None if seen < baseline[i] or mean[i] == 0 else closes[i]/mean[i]-1
            if distance is not None and abs(distance) < 1e-12 and tr is not None and abs(tr) >= float(detector.cfg['trend_threshold']):
                distance = detector._distance_from_ma(closes[max(0, i-int(baseline[i])+1):i+1].tolist(), int(baseline[i]))
            dd = None if peak[i] == 0 else closes[i]/peak[i]-1
            vol = None if np.isnan(vols[i]) else float(vols[i])
            rank = None if np.isnan(ranks[i]) else float(ranks[i])
            ready = seen >= (max(trend[i]+1, baseline[i], volatility[i]+1) if detector.cfg['require_full_windows'] else 2)
            trend_regime, strength = detector._trend_regime(tr, distance, ready)
            vol_regime = detector._volatility_regime(rank, ready)
            output[index][symbol] = MarketRegimeSnapshot(symbol, bar.timestamp, seen, bar.close,
                trend_regime, vol_regime, f'{trend_regime}_{vol_regime}', tr, strength, vol,
                None if vol is None else vol * (float(detector.cfg['annualization_days'])*settings[i, 0])**.5,
                rank, dd, distance, inferred[i], float(settings[i, 0]), bool(ready))
            if progress and ((i+1) % 1000 == 0 or i+1 == n):
                progress(done+i+1, total)
        done += n
    return output
