"""Tests for configuration loader."""

from pathlib import Path

import pytest

from voice_agent.config.loader import ConfigLoader


def test_load_yaml_valid(temp_config_dir: Path) -> None:
    """Test loading valid YAML configuration."""
    config_file = temp_config_dir / "test.yaml"
    config_file.write_text(
        """
tenant_id: test_tenant
persona:
  name: Test Agent
  voice_id: test_voice
"""
    )

    config = ConfigLoader.load_yaml(config_file)

    assert config["tenant_id"] == "test_tenant"
    assert config["persona"]["name"] == "Test Agent"
    assert config["persona"]["voice_id"] == "test_voice"


def test_load_yaml_not_found() -> None:
    """Test loading non-existent YAML file."""
    with pytest.raises(FileNotFoundError):
        ConfigLoader.load_yaml(Path("/nonexistent/file.yaml"))


def test_load_yaml_invalid(temp_config_dir: Path) -> None:
    """Test loading invalid YAML file."""
    config_file = temp_config_dir / "invalid.yaml"
    config_file.write_text("invalid: yaml: content: [")

    with pytest.raises(ValueError, match="Invalid YAML"):
        ConfigLoader.load_yaml(config_file)


def test_load_toml_valid(temp_config_dir: Path) -> None:
    """Test loading valid TOML configuration."""
    config_file = temp_config_dir / "test.toml"
    config_file.write_text(
        """
tenant_id = "test_tenant"

[persona]
name = "Test Agent"
voice_id = "test_voice"
"""
    )

    config = ConfigLoader.load_toml(config_file)

    assert config["tenant_id"] == "test_tenant"
    assert config["persona"]["name"] == "Test Agent"
    assert config["persona"]["voice_id"] == "test_voice"


def test_load_toml_not_found() -> None:
    """Test loading non-existent TOML file."""
    with pytest.raises(FileNotFoundError):
        ConfigLoader.load_toml(Path("/nonexistent/file.toml"))


def test_load_auto_detect_yaml(temp_config_dir: Path) -> None:
    """Test auto-detection of YAML format."""
    config_file = temp_config_dir / "test.yaml"
    config_file.write_text("tenant_id: test_tenant")

    config = ConfigLoader.load(config_file)

    assert config["tenant_id"] == "test_tenant"


def test_load_auto_detect_yml(temp_config_dir: Path) -> None:
    """Test auto-detection of .yml extension."""
    config_file = temp_config_dir / "test.yml"
    config_file.write_text("tenant_id: test_tenant")

    config = ConfigLoader.load(config_file)

    assert config["tenant_id"] == "test_tenant"


def test_load_auto_detect_toml(temp_config_dir: Path) -> None:
    """Test auto-detection of TOML format."""
    config_file = temp_config_dir / "test.toml"
    config_file.write_text('tenant_id = "test_tenant"')

    config = ConfigLoader.load(config_file)

    assert config["tenant_id"] == "test_tenant"


def test_load_unsupported_format(temp_config_dir: Path) -> None:
    """Test loading file with unsupported extension."""
    config_file = temp_config_dir / "test.json"
    config_file.write_text('{"tenant_id": "test"}')

    with pytest.raises(ValueError, match="Unsupported configuration file format"):
        ConfigLoader.load(config_file)


def test_save_yaml(temp_config_dir: Path) -> None:
    """Test saving configuration to YAML."""
    config = {"tenant_id": "test_tenant", "persona": {"name": "Test Agent"}}
    config_file = temp_config_dir / "output.yaml"

    ConfigLoader.save_yaml(config, config_file)

    assert config_file.exists()
    loaded = ConfigLoader.load_yaml(config_file)
    assert loaded == config


def test_save_toml(temp_config_dir: Path) -> None:
    """Test saving configuration to TOML."""
    config = {"tenant_id": "test_tenant", "persona": {"name": "Test Agent"}}
    config_file = temp_config_dir / "output.toml"

    ConfigLoader.save_toml(config, config_file)

    assert config_file.exists()
    loaded = ConfigLoader.load_toml(config_file)
    assert loaded == config
