"""Configuration file loader for YAML and TOML formats."""

from pathlib import Path
from typing import Any

import toml
import yaml


class ConfigLoader:
    """Loads configuration from YAML or TOML files."""

    @staticmethod
    def load_yaml(file_path: Path) -> dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            file_path: Path to YAML file

        Returns:
            Dictionary containing configuration data

        Raises:
            FileNotFoundError: If file does not exist
            yaml.YAMLError: If file is not valid YAML
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f)
                return config if config is not None else {}
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in {file_path}: {e}") from e

    @staticmethod
    def load_toml(file_path: Path) -> dict[str, Any]:
        """Load configuration from TOML file.

        Args:
            file_path: Path to TOML file

        Returns:
            Dictionary containing configuration data

        Raises:
            FileNotFoundError: If file does not exist
            toml.TomlDecodeError: If file is not valid TOML
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                return toml.load(f)
        except toml.TomlDecodeError as e:
            raise ValueError(f"Invalid TOML in {file_path}: {e}") from e

    @staticmethod
    def load(file_path: Path) -> dict[str, Any]:
        """Load configuration from file, auto-detecting format.

        Args:
            file_path: Path to configuration file (.yaml, .yml, or .toml)

        Returns:
            Dictionary containing configuration data

        Raises:
            ValueError: If file extension is not supported
            FileNotFoundError: If file does not exist
        """
        suffix = file_path.suffix.lower()

        if suffix in [".yaml", ".yml"]:
            return ConfigLoader.load_yaml(file_path)
        elif suffix == ".toml":
            return ConfigLoader.load_toml(file_path)
        else:
            raise ValueError(
                f"Unsupported configuration file format: {suffix}. "
                "Supported formats: .yaml, .yml, .toml"
            )

    @staticmethod
    def save_yaml(config: dict[str, Any], file_path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            config: Configuration dictionary
            file_path: Path to save YAML file
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def save_toml(config: dict[str, Any], file_path: Path) -> None:
        """Save configuration to TOML file.

        Args:
            config: Configuration dictionary
            file_path: Path to save TOML file
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            toml.dump(config, f)
