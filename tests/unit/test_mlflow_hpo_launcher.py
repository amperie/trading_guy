from trading.launchers.mlflow_hpo_launcher import (
    _component_source_unusable,
    _merge_missing_component_sources,
    parse_dotted_overrides,
    prepare_hpo_config_from_source,
    SourceRunContext,
)


def test_component_source_unusable_when_temp_path_is_missing():
    cfg = {
        "algorithm": {
            "implementation": "AdxMomentumFilterEmaAlgorithm",
            "source_path": r"C:\Users\pablo\AppData\Local\Temp\tmpptmh8tsg\AdxMomentumFilterEmaAlgorithm.py",
        }
    }

    assert _component_source_unusable(cfg, "algorithm") is True


def test_merge_missing_component_sources_replaces_dead_source_path():
    raw_cfg = {
        "algorithm": {
            "implementation": "AdxMomentumFilterEmaAlgorithm",
            "source_path": r"C:\Users\pablo\AppData\Local\Temp\tmpptmh8tsg\AdxMomentumFilterEmaAlgorithm.py",
            "class_name": "AdxMomentumFilterEmaAlgorithm",
        }
    }
    fallback_cfg = {
        "algorithm": {
            "implementation": "AdxMomentumFilterEmaAlgorithm",
            "source_path": "trading/promoted/source/AdxMomentumFilterEmaAlgorithm.py",
            "class_name": "AdxMomentumFilterEmaAlgorithm",
        }
    }

    merged = _merge_missing_component_sources(raw_cfg, fallback_cfg)

    assert merged["algorithm"]["source_path"] == "trading/promoted/source/AdxMomentumFilterEmaAlgorithm.py"


def test_parse_dotted_overrides_supports_nested_keys():
    overrides = parse_dotted_overrides(["momentum_lookback=1200", "risk.threshold=1.5", "enabled=true"])

    assert overrides["momentum_lookback"] == 1200
    assert overrides["risk"]["threshold"] == 1.5
    assert overrides["enabled"] is True


def test_parse_dotted_overrides_rejects_bad_syntax():
    try:
        parse_dotted_overrides(["not-valid"])
    except ValueError as exc:
        assert "Expected KEY=VALUE" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid override syntax")


def test_prepare_hpo_config_from_source_applies_algorithm_param_overrides():
    source_context = SourceRunContext(
        run_id="abc123",
        run_name="demo",
        tracking_uri="http://localhost:5000",
        source_url="http://localhost:5000/#/experiments/1/runs/abc123",
        config_source="artifact:config/demo.yaml",
        raw_config={
            "algorithm": {
                "implementation": "tests.fixtures.custom_components.CustomAlgorithm",
                "params": {"lookback": 10, "history_length": 30},
            },
            "hpo": {},
        },
    )

    prepared = prepare_hpo_config_from_source(
        source_context,
        "hpo_from_mlflow",
        algorithm_overrides={"lookback": 25, "nested": {"threshold": 1.25}},
    )

    assert prepared["algorithm"]["params"]["lookback"] == 25
    assert prepared["algorithm"]["params"]["history_length"] == 30
    assert prepared["algorithm"]["params"]["nested"]["threshold"] == 1.25
