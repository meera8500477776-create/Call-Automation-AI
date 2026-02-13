"""Configuration validation using Pydantic schemas."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from .loader import ConfigLoader
from .schema import TenantConfig


class ValidationResult:
    """Result of configuration validation."""

    def __init__(
        self,
        is_valid: bool,
        errors: Optional[List[str]] = None,
        config: Optional[TenantConfig] = None,
    ):
        """Initialize validation result.

        Args:
            is_valid: Whether the configuration is valid
            errors: List of validation error messages
            config: Validated configuration object (if valid)
        """
        self.is_valid = is_valid
        self.errors = errors or []
        self.config = config

    def __bool__(self) -> bool:
        """Allow using ValidationResult in boolean context."""
        return self.is_valid

    def __repr__(self) -> str:
        """String representation of validation result."""
        if self.is_valid:
            return f"ValidationResult(valid=True, tenant_id={self.config.tenant_id})"
        return f"ValidationResult(valid=False, errors={len(self.errors)})"


class ConfigValidator:
    """Validates tenant configuration files against schema."""

    @staticmethod
    def validate_dict(config_dict: Dict[str, Any]) -> ValidationResult:
        """Validate configuration dictionary against schema.

        Args:
            config_dict: Configuration dictionary to validate

        Returns:
            ValidationResult with validation status and errors
        """
        try:
            config = TenantConfig.model_validate(config_dict)
            return ValidationResult(is_valid=True, config=config)
        except ValidationError as e:
            errors = []
            for error in e.errors():
                loc = " -> ".join(str(x) for x in error["loc"])
                msg = error["msg"]
                errors.append(f"{loc}: {msg}")
            return ValidationResult(is_valid=False, errors=errors)

    @staticmethod
    def validate_file(file_path: Path) -> ValidationResult:
        """Validate configuration file against schema.

        Args:
            file_path: Path to configuration file (.yaml, .yml, or .toml)

        Returns:
            ValidationResult with validation status and errors
        """
        try:
            config_dict = ConfigLoader.load(file_path)
            return ConfigValidator.validate_dict(config_dict)
        except (FileNotFoundError, ValueError) as e:
            return ValidationResult(is_valid=False, errors=[str(e)])

    @staticmethod
    def load_and_validate(file_path: Path) -> TenantConfig:
        """Load and validate configuration file, raising exception on error.

        Args:
            file_path: Path to configuration file

        Returns:
            Validated TenantConfig object

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If configuration is invalid
        """
        result = ConfigValidator.validate_file(file_path)
        if not result.is_valid:
            error_msg = "\n".join(result.errors)
            raise ValueError(f"Invalid configuration in {file_path}:\n{error_msg}")
        return result.config
