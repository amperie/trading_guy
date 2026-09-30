from algo_crucible.windows import generate_walk_forward_windows
import pytest


def test_screening_windows_preserve_endpoints_and_original_ids():
    args=dict(data_start='2020-01-01',data_end='2023-01-01',optimization_window_days=30,
              validation_window_days=10,embargo_days=2)
    full=generate_walk_forward_windows(**args)
    small=generate_walk_forward_windows(**args,max_windows=3)
    assert len(small)==3 and small[0]==full[0] and small[-1]==full[-1]
    assert small==generate_walk_forward_windows(**args,max_windows=3)
    assert len(generate_walk_forward_windows(**args,max_windows=1))==1
    with pytest.raises(ValueError):
        generate_walk_forward_windows(**args,max_windows=2,min_windows=3)
