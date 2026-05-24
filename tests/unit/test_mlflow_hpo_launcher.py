from trading.launchers.mlflow_hpo_launcher import _component_source_unusable, _merge_missing_component_sources


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
