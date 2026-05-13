from __future__ import annotations

import time
from pathlib import Path

from trading.launchers import mlflow_hpo_launcher as launcher


class _WorkspaceTempDir:
    def __init__(self, path):
        self._path = path

    def __enter__(self):
        self._path.mkdir(parents=True, exist_ok=True)
        return str(self._path)

    def __exit__(self, exc_type, exc, tb):
        for child in self._path.iterdir():
            child.unlink()
        self._path.rmdir()
        return False


def test_extract_run_id_and_tracking_uri():
    run_url = "http://localhost:5000/#/experiments/12/runs/abc123def456"
    assert launcher.extract_run_id(run_url) == "abc123def456"
    assert launcher.infer_tracking_uri(run_url) == "http://localhost:5000"


def test_editor_command_defaults_windows(monkeypatch):
    monkeypatch.setattr(launcher.platform, "system", lambda: "Windows")
    monkeypatch.delenv("CODEX_EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    assert launcher._editor_command() == ["notepad.exe"]


def test_editor_command_defaults_linux(monkeypatch):
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.delenv("CODEX_EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/vim" if name == "vim" else None)
    assert launcher._editor_command() == ["vim"]


def test_reconstruct_config_from_params():
    params = {
        "config.mode": "backtest",
        "config.algorithm.implementation": "pkg.Algo",
        "config.algorithm.params.lookback": "20",
        "config.portfolio.implementation": "pkg.Portfolio",
        "config.portfolio.params.stop_pct": "5.5",
        "ignored": "value",
    }

    raw = launcher.reconstruct_config_from_params(params)

    assert raw["mode"] == "backtest"
    assert raw["algorithm"]["implementation"] == "pkg.Algo"
    assert raw["algorithm"]["params"]["lookback"] == 20
    assert raw["portfolio"]["params"]["stop_pct"] == 5.5


def test_merge_hpo_artifact_config_overlays_saved_hpo(monkeypatch):
    raw_cfg = {"hpo": {"num_samples": 10}}
    client = object()
    monkeypatch.setattr(
        launcher,
        "_load_json_artifact",
        lambda mlflow_client, tracking_uri, run_id, artifact_path: {
            "search_space": {"stop_pct": {"type": "uniform", "low": 1.0, "high": 5.0}},
            "portfolio_param_keys": ["stop_pct"],
            "num_samples": 77,
        },
    )

    merged = launcher._merge_hpo_artifact_config(raw_cfg, client, "http://localhost:5000", "run-1")

    assert merged["hpo"]["num_samples"] == 77
    assert merged["hpo"]["portfolio_param_keys"] == ["stop_pct"]
    assert "search_space" in merged["hpo"]


def test_download_artifact_times_out(monkeypatch):
    monkeypatch.setattr(launcher, "ARTIFACT_DOWNLOAD_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(
        "mlflow.artifacts.download_artifacts",
        lambda artifact_uri, tracking_uri=None: time.sleep(0.05) or "never-returned-in-time",
    )

    try:
        launcher._download_artifact("http://localhost:5000", "run-1", "config/runtime_config.yaml")
        assert False, "Expected TimeoutError"
    except TimeoutError as exc:
        assert "config/runtime_config.yaml" in str(exc)
        assert "run_id=run-1" in str(exc)


def test_load_json_artifact_skips_missing_artifact_without_download(monkeypatch):
    class FakeClient:
        def list_artifacts(self, run_id, path=""):
            return []

    called = {"download": False}
    monkeypatch.setattr(
        launcher,
        "_download_artifact",
        lambda tracking_uri, run_id, artifact_path: called.update({"download": True}) or "unexpected",
    )

    result = launcher._load_json_artifact(
        FakeClient(),
        "http://localhost:5000",
        "run-1",
        "config/hpo_config.json",
    )

    assert result is None
    assert called["download"] is False


def test_prepare_hpo_config_from_source():
    context = launcher.SourceRunContext(
        run_id="abc123",
        run_name="source-run",
        tracking_uri="http://localhost:5000",
        source_url="http://localhost:5000/#/experiments/1/runs/abc123",
        raw_config={
            "mode": "backtest",
            "algorithm": {"algorithm": "pkg.Algo", "lookback": 10},
            "portfolio": {"portfolio": "pkg.Portfolio", "stop_pct": 5.0},
            "order_manager": {"order_manager": "pkg.OM"},
            "data_provider": {"provider": "pkg.Provider", "path": "data.csv"},
            "analysis": {"run_name": "orig"},
        },
        config_source="artifact:config/example.yaml",
    )

    prepared = launcher.prepare_hpo_config_from_source(
        source_context=context,
        experiment_name="Recreated HPO",
    )

    assert prepared["mode"] == "hpo"
    assert prepared["analysis"]["experiment_name"] == "Recreated HPO"
    assert prepared["mlflow"]["tracking_uri"] == "http://localhost:5000"
    assert prepared["hpo"] == {}


def test_edit_hpo_config_uses_saved_yaml(monkeypatch):
    monkeypatch.setattr(launcher.platform, "system", lambda: "Windows")
    monkeypatch.setattr(launcher.tempfile, "TemporaryDirectory", lambda prefix="": _WorkspaceTempDir(Path("scratch") / "tmp_editor_test"))
    written: dict[str, object] = {}

    def fake_run(cmd, check):
        path = cmd[-1]
        from pathlib import Path
        Path(path).write_text("mode: hpo\nhpo:\n  num_samples: 77\n", encoding="utf-8")
        written["cmd"] = cmd
        written["check"] = check

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    edited = launcher.edit_hpo_config({"mode": "hpo", "hpo": {"num_samples": 30}}, editor="notepad.exe")

    assert written["cmd"][0] == "notepad.exe"
    assert written["check"] is True
    assert edited["hpo"]["num_samples"] == 77


def test_sanitize_source_config_uses_execution_config_and_generated_algorithm():
    raw = {
        "proposal_name": "demo",
        "execution_config": {
            "mode": "backtest",
            "algorithm": {
                "implementation": "__generated__",
                "params": {"lookback": 10},
            },
            "portfolio": {
                "implementation": "pkg.Portfolio",
                "params": {"stop_pct": 5.0},
            },
            "order_manager": {"implementation": "pkg.OM", "params": {}},
            "data_provider": {"implementation": "pkg.Provider", "params": {}},
            "generated_algorithm": {
                "class_name": "GeneratedAlgo",
                "source_url": "https://example.com/GeneratedAlgo.py",
                "script_path": "dev/generated/GeneratedAlgo.py",
            },
            "unused_field": {"a": 1},
        },
    }

    sanitized = launcher.sanitize_source_config(raw)

    assert sanitized["algorithm"]["implementation"] == "GeneratedAlgo"
    assert sanitized["algorithm"]["class_name"] == "GeneratedAlgo"
    assert sanitized["algorithm"]["source_url"] == "https://example.com/GeneratedAlgo.py"
    assert sanitized["algorithm"]["source_path"] == "dev/generated/GeneratedAlgo.py"
    assert "generated_algorithm" not in sanitized
    assert "unused_field" not in sanitized


def test_resolve_component_sources_from_artifacts_downloads_missing_script(monkeypatch):
    class Artifact:
        def __init__(self, path, is_dir=False):
            self.path = path
            self.is_dir = is_dir

    cfg = {
        "algorithm": {
            "implementation": "GeneratedAlgo",
            "class_name": "GeneratedAlgo",
            "source_path": "dev\\experiments\\trading\\implementations\\GeneratedAlgo.py",
        }
    }
    client = type("Client", (), {
        "list_artifacts": lambda self, run_id, path="": [Artifact("code/GeneratedAlgo.py")]
    })()
    monkeypatch.setattr(
        launcher,
        "_download_artifact",
        lambda tracking_uri, run_id, artifact_path: "scratch/downloaded/GeneratedAlgo.py",
    )

    resolved = launcher._resolve_component_sources_from_artifacts(
        cfg,
        client,
        "http://localhost:5000",
        "run-123",
    )

    assert resolved["algorithm"]["source_path"].endswith("scratch\\downloaded\\GeneratedAlgo.py") or resolved["algorithm"]["source_path"].endswith("scratch/downloaded/GeneratedAlgo.py")


def test_extract_source_run_url_from_description():
    description = (
        "Recreated HPO from source MLflow run ddea820c81e843d1831ba531bd4c14ce "
        "(http://hp.lan:8899/#/experiments/596060974901698399/runs/ddea820c81e843d1831ba531bd4c14ce)"
    )
    assert launcher._extract_source_run_url_from_description(description) == (
        "http://hp.lan:8899/#/experiments/596060974901698399/runs/ddea820c81e843d1831ba531bd4c14ce"
    )


def test_merge_missing_component_sources_prefers_fallback_when_primary_missing():
    primary = {
        "algorithm": {"implementation": "pkg.Algo"},
        "portfolio": {"implementation": "pkg.Portfolio", "source_path": "local/portfolio.py"},
    }
    fallback = {
        "algorithm": {
            "implementation": "pkg.Algo",
            "source_path": "downloaded/algo.py",
            "class_name": "Algo",
        },
        "portfolio": {
            "implementation": "pkg.Portfolio",
            "source_path": "downloaded/portfolio.py",
        },
    }

    merged = launcher._merge_missing_component_sources(primary, fallback)

    assert merged["algorithm"]["source_path"] == "downloaded/algo.py"
    assert merged["algorithm"]["class_name"] == "Algo"
    assert merged["portfolio"]["source_path"] == "local/portfolio.py"


def test_load_source_run_context_falls_back_to_description_source_run(monkeypatch):
    class FakeRun:
        def __init__(self, run_name, description=""):
            self.data = type(
                "Data",
                (),
                {
                    "params": {},
                    "tags": {
                        "mlflow.runName": run_name,
                        "mlflow.note.content": description,
                    },
                },
            )()

    class FakeClient:
        def __init__(self, tracking_uri=None):
            self.tracking_uri = tracking_uri

        def get_run(self, run_id):
            if run_id == "child123":
                return FakeRun(
                    "child-run",
                    "Recreated HPO from source MLflow run source456 "
                    "(http://localhost:5000/#/experiments/1/runs/source456)",
                )
            if run_id == "source456":
                return FakeRun("source-run")
            raise AssertionError(f"Unexpected run_id: {run_id}")

        def list_artifacts(self, run_id, path=""):
            return []

    configs = {
        "child123": {
            "mode": "backtest",
            "algorithm": {"implementation": "pkg.Algo", "params": {}},
            "portfolio": {"implementation": "pkg.Portfolio", "params": {}},
            "order_manager": {"implementation": "pkg.OM", "params": {}},
            "data_provider": {"implementation": "pkg.Provider", "params": {}},
        },
        "source456": {
            "mode": "backtest",
            "algorithm": {"implementation": "pkg.Algo", "source_path": "downloaded/algo.py", "params": {}},
            "portfolio": {"implementation": "pkg.Portfolio", "source_path": "downloaded/portfolio.py", "params": {}},
            "order_manager": {"implementation": "pkg.OM", "params": {}},
            "data_provider": {"implementation": "pkg.Provider", "params": {}},
        },
    }

    monkeypatch.setattr(launcher, "MLFLOW_AVAILABLE", True)
    monkeypatch.setattr("mlflow.set_tracking_uri", lambda uri: None)
    monkeypatch.setattr("mlflow.tracking.MlflowClient", FakeClient)
    monkeypatch.setattr(launcher, "_find_config_artifact_path", lambda client, run_id: f"config/{run_id}.yaml")
    monkeypatch.setattr(launcher, "_download_artifact", lambda tracking_uri, run_id, artifact_path: artifact_path)
    monkeypatch.setattr(launcher, "load_raw_config", lambda local_path: configs[Path(local_path).stem])
    monkeypatch.setattr(launcher, "_resolve_component_sources_from_artifacts", lambda cfg, client, tracking_uri, run_id: cfg)
    monkeypatch.setattr(launcher, "_merge_hpo_artifact_config", lambda raw_cfg, client, tracking_uri, run_id: raw_cfg)

    context = launcher.load_source_run_context("http://localhost:5000/#/experiments/1/runs/child123")

    assert context.run_id == "child123"
    assert context.raw_config["algorithm"]["source_path"] == "downloaded/algo.py"
    assert context.raw_config["portfolio"]["source_path"] == "downloaded/portfolio.py"


def test_run_launcher_wires_summary(monkeypatch):
    context = launcher.SourceRunContext(
        run_id="abc123",
        run_name="source-run",
        tracking_uri="http://localhost:5000",
        source_url="http://localhost:5000/#/experiments/1/runs/abc123",
        raw_config={
            "mode": "backtest",
            "algorithm": {"algorithm": "pkg.Algo", "lookback": 10},
            "portfolio": {"portfolio": "pkg.Portfolio", "stop_pct": 5.0},
            "order_manager": {"order_manager": "pkg.OM"},
            "data_provider": {"provider": "pkg.Provider", "path": "data.csv"},
        },
        config_source="params",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(launcher, "load_source_run_context", lambda run_url, tracking_uri=None: context)
    monkeypatch.setattr(launcher, "edit_hpo_config", lambda prepared_cfg, editor=None: prepared_cfg)
    monkeypatch.setattr(launcher, "persist_edited_hpo_config", lambda source_context, edited_cfg: "scratch/generated_hpo_configs/test.yaml")
    monkeypatch.setattr(
        launcher,
        "prompt_for_hpo_launch",
        lambda source_context: "Recreated HPO",
    )
    monkeypatch.setattr(launcher, "load_account_creds", lambda account: {"api_key": "x", "secret_key": "y"})
    monkeypatch.setattr(launcher, "_fill_hpo_data_provider_creds", lambda cfg, creds: captured.setdefault("creds", creds))
    monkeypatch.setattr(
        launcher,
        "edit_hpo_config",
        lambda prepared_cfg, editor=None: {
            **prepared_cfg,
            "hpo": {
                "search_space": {"lookback": {"type": "randint", "low": 5, "high": 21}},
                "algorithm_param_keys": ["lookback"],
                "portfolio_param_keys": [],
                "num_samples": 30,
                "max_concurrent_trials": 4,
            },
        },
    )
    monkeypatch.setattr(
        launcher,
        "run_hpo_from_raw_config",
        lambda cfg, num_samples_override=None, max_concurrent_override=None, config_artifact_path=None: captured.update(
            {"config_artifact_path": config_artifact_path}
        ) or {"lookback": 17},
    )
    monkeypatch.setattr(
        launcher,
        "log_hpo_launcher_summary",
        lambda source_context, prepared_cfg, best_config, edited_config_path=None: captured.update(
            {"prepared_cfg": prepared_cfg, "best_config": best_config, "edited_config_path": edited_config_path}
        ),
    )

    best = launcher.run_launcher("http://localhost:5000/#/experiments/1/runs/abc123", "paper")

    assert best == {"lookback": 17}
    assert captured["prepared_cfg"]["analysis"]["experiment_name"] == "Recreated HPO"
    assert captured["best_config"] == {"lookback": 17}
    assert captured["config_artifact_path"] == "scratch/generated_hpo_configs/test.yaml"
    assert captured["edited_config_path"] == "scratch/generated_hpo_configs/test.yaml"


def test_run_launcher_requires_search_space_after_edit(monkeypatch):
    context = launcher.SourceRunContext(
        run_id="abc123",
        run_name="source-run",
        tracking_uri="http://localhost:5000",
        source_url="http://localhost:5000/#/experiments/1/runs/abc123",
        raw_config={
            "mode": "backtest",
            "algorithm": {"algorithm": "pkg.Algo", "lookback": 10},
            "portfolio": {"portfolio": "pkg.Portfolio", "stop_pct": 5.0},
            "order_manager": {"order_manager": "pkg.OM"},
            "data_provider": {"provider": "pkg.Provider", "path": "data.csv"},
        },
        config_source="params",
    )
    monkeypatch.setattr(launcher, "load_source_run_context", lambda run_url, tracking_uri=None: context)
    monkeypatch.setattr(
        launcher,
        "prompt_for_hpo_launch",
        lambda source_context: "Recreated HPO",
    )
    monkeypatch.setattr(
        launcher,
        "edit_hpo_config",
        lambda prepared_cfg, editor=None: prepared_cfg,
    )
    monkeypatch.setattr(launcher, "load_account_creds", lambda account: {"api_key": "x", "secret_key": "y"})
    monkeypatch.setattr(launcher, "_fill_hpo_data_provider_creds", lambda cfg, creds: None)

    try:
        launcher.run_launcher("http://localhost:5000/#/experiments/1/runs/abc123", "paper")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "empty hpo.search_space" in str(exc)
