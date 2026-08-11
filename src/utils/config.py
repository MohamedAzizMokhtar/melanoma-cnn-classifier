import yaml
from pathlib import Path
from typing import Any


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    """Validate that required configuration sections exist."""
    required_sections = ["experiment", "paths", "data", "training", "model"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")


def get_config_value(config: dict, key_path: str, default: Any = None) -> Any:
    """Get a nested config value using dot notation.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path (e.g., 'training.epochs').
        default: Default value if key is not found.

    Returns:
        The configuration value.
    """
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value
