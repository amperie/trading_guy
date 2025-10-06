import yaml
from pathlib import Path
from typing import Any, Dict
from types import SimpleNamespace


class ConfigManager:
    _instance = None
    _config: Dict[str, Any] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only load once
        if self._config is None:
            self._load_config()

    def _load_config(self, config_path: str = "../config.yaml"):
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, 'r') as f:
            self._config = yaml.safe_load(f)

    def reload(self, config_path: str = "../config.yaml"):
        self._load_config(config_path)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by key, supports nested keys with dots."""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    @staticmethod
    def dict_to_namespace(d: Any) -> Any:
        """Recursively convert dict to SimpleNamespace."""
        if isinstance(d, dict):
            return SimpleNamespace(**{k: ConfigManager.dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [ConfigManager.dict_to_namespace(item) for item in d]
        else:
            return d

    def get_as_object(self, key: str = None) -> SimpleNamespace:
        """Get config as object with dot notation access."""
        if key:
            value = self.get(key)
            return self.dict_to_namespace(value) if isinstance(value, dict) else value
        return self.dict_to_namespace(self._config)

    @property
    def config(self) -> Dict[str, Any]:
        """Return the entire config dictionary."""
        return self._config