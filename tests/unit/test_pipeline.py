from trading.pipeline import evaluate_research_gates, evaluate_review_gates


class _Metrics:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_research_gates_pass_with_configured_thresholds():
    raw_cfg = {
        "pipeline": {
            "gates": {
                "research": {
                    "min_val_annualized_return": 5,
                    "max_val_max_drawdown_pct": 20,
                    "min_val_total_trades": 10,
                    "min_wf_annualized_return": 4,
                    "max_wf_max_drawdown_pct": 25,
                }
            }
        }
    }
    report = evaluate_research_gates(
        raw_cfg,
        {"analysis": {"metrics": _Metrics(annualized_return=8, max_drawdown_pct=12)}},
        {"val_results": {"metrics": _Metrics(annualized_return=7, max_drawdown_pct=18, total_trades=14)}},
        {"aggregate": {"wf_annualized_return": 6, "wf_max_drawdown_pct": 19}},
    )
    assert report.passed is True
    assert all(check.passed for check in report.checks)


def test_research_gates_fail_when_validation_or_walk_forward_miss():
    raw_cfg = {
        "pipeline": {
            "gates": {
                "research": {
                    "min_val_annualized_return": 5,
                    "max_wf_max_drawdown_pct": 10,
                }
            }
        }
    }
    report = evaluate_research_gates(
        raw_cfg,
        None,
        {"val_results": {"metrics": _Metrics(annualized_return=3, max_drawdown_pct=8, total_trades=20)}},
        {"aggregate": {"wf_max_drawdown_pct": 12}},
    )
    assert report.passed is False
    assert [check.name for check in report.checks if not check.passed] == [
        "val_annualized_return",
        "wf_max_drawdown_pct",
    ]


def test_review_gates_fail_on_large_drift():
    raw_cfg = {
        "pipeline": {
            "gates": {
                "review": {
                    "max_alpaca_live_equity_drift_pct": 2,
                    "max_mongo_live_equity_drift_pct": 1,
                }
            }
        }
    }
    report = evaluate_review_gates(
        raw_cfg,
        {
            "alpaca_live_equity_drift_pct": 3,
            "mongo_live_equity_drift_pct": 0.5,
        },
    )
    assert report.passed is False
    assert [check.name for check in report.checks if not check.passed] == [
        "alpaca_live_equity_drift_pct"
    ]
