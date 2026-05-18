"""
MLflow client for tracking trading experiments, metrics, and artifacts
Provides convenient methods for logging runs, parameters, metrics, and various artifact types
"""
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Optional, Any, Union
from datetime import datetime
import tempfile

try:
    import mlflow
    from mlflow.tracking import MlflowClient as BaseMlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    BaseMlflowClient = object

from utils.logger import Logger
from utils.config_manager import ConfigManager

logger = Logger().get_logger(__name__)


def _configure_gitpython_environment() -> None:
    """Help GitPython find git.exe in worker processes on Windows."""
    if os.environ.get("GIT_PYTHON_GIT_EXECUTABLE"):
        return

    git_path = shutil.which("git")
    if git_path:
        os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = git_path
        return

    common_windows_git = Path(r"C:\Program Files\Git\cmd\git.exe")
    if common_windows_git.exists():
        os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = str(common_windows_git)
        return

    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")


class MLflowClient:
    """
    Wrapper around MLflow for tracking trading experiments

    Features:
    - Run management with context manager support
    - Parameter and metric logging
    - Artifact logging (text, JSON, markdown, charts, HTML)
    - Automatic experiment creation
    - Run descriptions and tags

    Usage:
        # Basic usage
        client = MLflowClient(experiment_name="Trading Backtest")

        with client.start_run(run_name="SMA Crossover Strategy"):
            client.log_description("Testing 5/20 SMA crossover on AAPL")
            client.log_params({"symbol": "AAPL", "sma_short": 5, "sma_long": 20})
            client.log_metrics({"total_return": 15.2, "sharpe_ratio": 1.8})
            client.log_chart(fig, "equity_curve")
            client.log_json({"trades": [...]}, "trades")

        # Or manual run management
        client.start_run(run_name="Manual Strategy")
        client.log_param("strategy", "mean_reversion")
        client.log_metric("win_rate", 65.5)
        client.end_run()
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        run_name_prefix: Optional[str] = None,
        auto_log_system_info: bool = True,
        enabled: bool = True
    ):
        """
        Initialize MLflow client

        Args:
            experiment_name: Name of the MLflow experiment (default: from config)
            tracking_uri: MLflow tracking server URI (default: from config)
            artifact_location: Custom artifact storage location (default: from config)
            run_name_prefix: Optional prefix for run names (default: from config)
            auto_log_system_info: Automatically log system/environment info (default: from config)
            enabled: Enable/disable MLflow tracking (default: from config)
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "MLflow is not installed. Install with: pip install mlflow"
            )

        _configure_gitpython_environment()

        # Load config
        config = ConfigManager().get("mlflow", {})

        # Use provided values or fall back to config
        self.enabled = enabled if enabled is not None else config.get("enabled", True)
        self.experiment_name = experiment_name or config.get("experiment_name", "Trading Backtest")
        self.tracking_uri = tracking_uri or config.get("tracking_uri", "file:./mlruns")
        self.artifact_location = artifact_location or config.get("artifact_location")
        self.run_name_prefix = run_name_prefix or config.get("run_name_prefix", "")
        self.auto_log_system_info = auto_log_system_info if auto_log_system_info is not None else config.get("auto_log_system_info", True)

        if not self.enabled:
            logger.info("MLflow tracking is disabled (enabled=false in config)")
            self.experiment = None
            self.experiment_id = None
            self._active_run = None
            self._run_id = None
            return

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        self.experiment = None
        self.experiment_id = None
        self._refresh_experiment()

        # Track active run
        self._active_run = None
        self._run_id = None

        logger.info(f"MLflow client initialized: experiment='{self.experiment_name}', uri='{self.tracking_uri}'")

    @classmethod
    def from_config(cls, experiment_name: Optional[str] = None) -> "MLflowClient":
        """
        Create MLflowClient from config.yaml settings

        Args:
            experiment_name: Optional experiment name to override config

        Returns:
            MLflowClient instance
        """
        return cls(experiment_name=experiment_name)

    def _get_or_create_experiment(self):
        """Get existing experiment or create new one."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if experiment is None:
            if self.artifact_location:
                experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=self.artifact_location
                )
            else:
                experiment_id = mlflow.create_experiment(self.experiment_name)

            experiment = mlflow.get_experiment(experiment_id)
            logger.info(f"Created new MLflow experiment: {self.experiment_name} ({experiment_id})")
        else:
            logger.info(
                f"Using existing MLflow experiment: {self.experiment_name} ({experiment.experiment_id})"
            )

        return experiment

    def _refresh_experiment(self):
        """Resolve and cache the current experiment id, creating it when missing."""
        self.experiment = self._get_or_create_experiment()
        self.experiment_id = self.experiment.experiment_id
        return self.experiment

    def _log_system_info(self):
        """Log system and environment information"""
        try:
            system_info = {
                "python_version": sys.version.split()[0],
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "hostname": platform.node(),
            }

            # Log as tags
            for key, value in system_info.items():
                mlflow.set_tag(f"system.{key}", value)

            logger.debug("Logged system information")
        except Exception as e:
            logger.warning(f"Failed to log system info: {e}")

    def start_run(
        self,
        run_name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[dict] = None,
        nested: bool = False
    ) -> "MLflowClient":
        """
        Start a new MLflow run

        Args:
            run_name: Name for this run
            description: Description of the run
            tags: Dictionary of tags to attach to the run
            nested: Whether this is a nested run

        Returns:
            self (for context manager support)
        """
        if not self.enabled:
            logger.debug("MLflow tracking disabled, skipping start_run")
            return self

        # Apply run name prefix if configured
        if run_name and self.run_name_prefix:
            run_name = f"{self.run_name_prefix}{run_name}"

        # Re-resolve the experiment before creating a run in case the cached
        # experiment id is stale or the server lost state since client init.
        self._refresh_experiment()

        try:
            self._active_run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                nested=nested
            )
        except Exception as exc:
            logger.warning(
                f"MLflow start_run failed for experiment '{self.experiment_name}' "
                f"(id={self.experiment_id}). Re-resolving experiment and retrying once: {exc}"
            )
            self._refresh_experiment()
            self._active_run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                nested=nested
            )
        self._run_id = self._active_run.info.run_id

        # Log description as a tag if provided
        if description:
            self.log_description(description)

        # Log additional tags
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        # Log automatic tags
        mlflow.set_tag("start_time", datetime.now().isoformat())

        # Auto-log system info if enabled
        if self.auto_log_system_info:
            self._log_system_info()

        logger.info(f"Started MLflow run: {run_name or self._run_id}")

        return self

    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if not self.enabled:
            return

        if self._active_run:
            mlflow.set_tag("end_time", datetime.now().isoformat())
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run: {self._run_id} (status: {status})")
            self._active_run = None
            self._run_id = None
        else:
            logger.warning("No active run to end")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type is not None:
            # Exception occurred, mark run as failed
            self.end_run(status="FAILED")
            logger.error(f"Run failed with exception: {exc_type.__name__}: {exc_val}")
        else:
            self.end_run(status="FINISHED")
        return False

    def log_description(self, description: str):
        """
        Log run description

        Args:
            description: Description text
        """
        if not self.enabled:
            return
        mlflow.set_tag("mlflow.note.content", description)
        logger.debug(f"Logged description: {description[:50]}...")

    def log_param(self, key: str, value: Any):
        """
        Log a single parameter

        Args:
            key: Parameter name
            value: Parameter value
        """
        if not self.enabled:
            return
        mlflow.log_param(key, value)
        logger.debug(f"Logged parameter: {key}={value}")

    def log_params(self, params: dict):
        """
        Log multiple parameters

        Args:
            params: Dictionary of parameters
        """
        if not self.enabled:
            return
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} parameters")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a single metric

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number for time-series metrics
        """
        if not self.enabled:
            return
        mlflow.log_metric(key, value, step=step)
        logger.debug(f"Logged metric: {key}={value}" + (f" (step={step})" if step else ""))

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """
        Log multiple metrics

        Args:
            metrics: Dictionary of metrics
            step: Optional step number for time-series metrics
        """
        if not self.enabled:
            return
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics" + (f" (step={step})" if step else ""))

    def log_text(self, text: str, filename: str):
        """
        Log text as an artifact

        Args:
            text: Text content
            filename: Filename for the artifact (e.g., "report.txt")
        """
        if not self.enabled:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            mlflow.log_artifact(filepath)
        logger.debug(f"Logged text artifact: {filename}")

    def log_json(self, data: Union[dict, list], filename: str):
        """
        Log JSON data as an artifact

        Args:
            data: Dictionary or list to save as JSON
            filename: Filename for the artifact (e.g., "data.json")
        """
        if not self.enabled:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            mlflow.log_artifact(filepath)
        logger.debug(f"Logged JSON artifact: {filename}")

    def log_markdown(self, markdown: str, filename: str):
        """
        Log markdown content as an artifact

        Args:
            markdown: Markdown text
            filename: Filename for the artifact (e.g., "report.md")
        """
        if not self.enabled:
            return
        if not filename.endswith('.md'):
            filename += '.md'

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown)
            mlflow.log_artifact(filepath)
        logger.debug(f"Logged markdown artifact: {filename}")

    def log_html(self, html: str, filename: str):
        """
        Log HTML content as an artifact

        Args:
            html: HTML content
            filename: Filename for the artifact (e.g., "report.html")
        """
        if not self.enabled:
            logger.debug("MLflow disabled, skipping HTML logging")
            return
        if not filename.endswith('.html'):
            filename += '.html'

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            logger.debug(f"Writing HTML to temp file: {filepath}")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.debug(f"Logging artifact to MLflow: {filename}")
            mlflow.log_artifact(filepath)
            logger.debug(f"✓ Successfully logged HTML artifact: {filename}")

    def log_chart(self, figure, filename: str, format: str = 'png', dpi: int = 300):
        """
        Log matplotlib/plotly figure as an artifact

        Args:
            figure: Matplotlib or Plotly figure object
            filename: Filename for the artifact (without extension)
            format: Image format (png, jpg, svg, pdf)
            dpi: DPI for raster formats
        """
        if not self.enabled:
            return

        # Remove extension if provided
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]

        filename_with_ext = f"{filename}.{format}"

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, filename_with_ext)

            # Try matplotlib first
            try:
                figure.savefig(filepath, dpi=dpi, bbox_inches='tight', format=format)
            except AttributeError:
                # Try plotly
                try:
                    if format in ['png', 'jpg', 'jpeg', 'svg', 'pdf']:
                        figure.write_image(filepath, format=format)
                    else:
                        figure.write_html(filepath)
                except Exception as e:
                    logger.error(f"Failed to save figure: {e}")
                    raise

            mlflow.log_artifact(filepath)

        logger.debug(f"Logged chart artifact: {filename_with_ext}")

    def log_figure(self, figure, filename: str):
        """
        Shorthand for log_chart with default PNG format

        Args:
            figure: Matplotlib or Plotly figure
            filename: Filename for the artifact
        """
        self.log_chart(figure, filename, format='png')

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an existing file as an artifact

        Args:
            local_path: Path to local file
            artifact_path: Optional subdirectory in artifact store
        """
        if not self.enabled:
            return
        mlflow.log_artifact(local_path, artifact_path)
        logger.debug(f"Logged artifact: {local_path}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Log all files in a directory as artifacts

        Args:
            local_dir: Path to local directory
            artifact_path: Optional subdirectory in artifact store
        """
        if not self.enabled:
            return
        mlflow.log_artifacts(local_dir, artifact_path)
        logger.debug(f"Logged artifacts from directory: {local_dir}")

    def log_dict(self, dictionary: dict, filename: str):
        """
        Log a dictionary as JSON artifact
        Alias for log_json for convenience

        Args:
            dictionary: Dictionary to log
            filename: Filename for the artifact
        """
        self.log_json(dictionary, filename)

    def set_tag(self, key: str, value: Any):
        """
        Set a tag on the current run

        Args:
            key: Tag name
            value: Tag value
        """
        if not self.enabled:
            return
        mlflow.set_tag(key, value)
        logger.debug(f"Set tag: {key}={value}")

    def set_tags(self, tags: dict):
        """
        Set multiple tags on the current run

        Args:
            tags: Dictionary of tags
        """
        if not self.enabled:
            return
        mlflow.set_tags(tags)
        logger.debug(f"Set {len(tags)} tags")

    def log_model_info(self, model_info: dict):
        """
        Log model/strategy information as parameters and tags

        Args:
            model_info: Dictionary with model metadata
        """
        if not self.enabled:
            return
        for key, value in model_info.items():
            # Use tags for string values, params for numbers
            if isinstance(value, (int, float, bool)):
                self.log_param(key, value)
            else:
                self.set_tag(key, str(value))

        logger.debug(f"Logged model info: {len(model_info)} items")

    def get_run_url(self) -> str:
        """
        Get the URL to view the current run in MLflow UI

        Returns:
            URL string
        """
        if self._run_id:
            # Construct URL based on tracking URI
            if self.tracking_uri.startswith("file:"):
                # Local tracking
                return f"http://localhost:5000/#/experiments/{self.experiment_id}/runs/{self._run_id}"
            else:
                # Remote tracking server
                return f"{self.tracking_uri}/#/experiments/{self.experiment_id}/runs/{self._run_id}"
        else:
            logger.warning("No active run")
            return ""

    @property
    def run_id(self) -> Optional[str]:
        """Get current run ID"""
        return self._run_id

    @property
    def is_active(self) -> bool:
        """Check if there's an active run"""
        return self._active_run is not None
