import math
from pathlib import Path

import pandas as pd
import pytest

from algo_crucible.structural_break import analyze_structural_breaks
from algo_crucible.structural_stream import analyze_file


def test_columnar_analysis_matches_reference(tmp_path):
    rows = [dict(candidate_id=c,timestamp=f'{i:05d}',window_id='w',step=i,
                 signal=math.sin(i*.12), forward_return_pct=math.sin(i*.12+.3))
            for c in ('a','b') for i in range(400)]
    folder = tmp_path/'stages/07_perturbation/summaries'
    folder.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(folder/'perturbation_signal_forward_returns.csv',index=False)
    cfg={'structural_break_stability':{'ic_windows':[30,60]}}
    expected=analyze_structural_breaks(rows,cfg)
    updates=[]
    actual=analyze_file(tmp_path,cfg,updates.append)
    assert actual['observation_count']==len(rows)
    for left,right in zip(expected['summary_rows'],actual['summary_rows']):
        for key,value in left.items():
            assert right[key] == (pytest.approx(value,abs=1e-8) if isinstance(value,float) else value), key
    output=pd.read_csv(actual['windows_path'])
    assert len(output)==len(expected['window_rows'])
    assert output.ic.tolist()==pytest.approx([r['ic'] for r in expected['window_rows']],abs=1e-8)
    assert len(updates)==2


def test_constant_signals_stay_unavailable(tmp_path):
    folder=tmp_path/'stages/07_perturbation/summaries'
    folder.mkdir(parents=True)
    pd.DataFrame([dict(candidate_id='a',signal=1,forward_return_pct=i,timestamp=str(i),step=i)
                  for i in range(150)]).to_csv(folder/'perturbation_signal_forward_returns.csv',index=False)
    result=analyze_file(tmp_path,{})
    assert result['summary_rows'][0]['failure_reason']=='insufficient_ic_windows'


def test_return_fallback_preserves_candidate_identity(tmp_path):
    folder=tmp_path/'stages/07_perturbation/summaries'
    folder.mkdir(parents=True)
    rows=[dict(candidate_id='001',return_pct=math.sin(i)) for i in range(40)]
    pd.DataFrame(rows).to_csv(folder/'perturbation_return_stream.csv',index=False)
    actual=analyze_file(tmp_path,{})
    expected=analyze_structural_breaks(rows,{})['summary_rows'][0]
    for key,value in expected.items():
        assert actual['summary_rows'][0][key] == (pytest.approx(value,abs=1e-12) if isinstance(value,float) else value)
