"""Tenant registry for managing multi-tenant configurations."""

import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .schema import TenantConfig
from .validator import ConfigValidator

logger = logging.getLogger(__name__)


class TenantRegistry:
    """Registry for loading and managing tenant configurations.

    Provides:
    - Configuration loading from YAML/TOML files
    - Caller ID to tenant mapping
    - Hot-reload capability with file watching
    - Configuration caching for performance
    """

    def __init__(self, config_dir: Path, caller_mapping: dict[str, str] | None = None):
        """Initialize the tenant registry.

        Args:
            config_dir: Directory containing tenant configuration files
            caller_mapping: Optional mapping of caller IDs to tenant IDs
        """
        self.config_dir = Path(config_dir)
        self._cache: dict[str, TenantConfig] = {}
        self._cache_lock = threading.RLock()
        self._caller_mapping = caller_mapping or {}
        self._observer: Observer | None = None
        self._file_handler: FileSystemEventHandler | None = None

        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {self.config_dir}")

    def load_tenant_config(self, tenant_id: str) -> TenantConfig:
        """Load tenant configuration from file.

        Checks cache first, then loads from file if not cached.

        Args:
            tenant_id: Unique tenant identifier

        Returns:
            Validated TenantConfig object

        Raises:
            FileNotFoundError: If configuration file not found
            ValueError: If configuration is invalid
        """
        # Check cache first
        with self._cache_lock:
            if tenant_id in self._cache:
                logger.debug(f"Returning cached config for tenant: {tenant_id}")
                return self._cache[tenant_id]

        # Load from file
        config = self._load_config_from_file(tenant_id)

        # Cache the configuration
        with self._cache_lock:
            self._cache[tenant_id] = config

        logger.info(f"Loaded configuration for tenant: {tenant_id}")
        return config

    def _load_config_from_file(self, tenant_id: str) -> TenantConfig:
        """Load configuration from file without caching.

        Args:
            tenant_id: Unique tenant identifier

        Returns:
            Validated TenantConfig object

        Raises:
            FileNotFoundError: If configuration file not found
            ValueError: If configuration is invalid
        """
        # Try different file extensions
        for ext in [".yaml", ".yml", ".toml"]:
            config_file = self.config_dir / f"{tenant_id}{ext}"
            if config_file.exists():
                return ConfigValidator.load_and_validate(config_file)

        raise FileNotFoundError(
            f"Configuration file not found for tenant '{tenant_id}' in {self.config_dir}"
        )

    def get_tenant_by_caller(self, caller_id: str) -> str:
        """Map caller ID to tenant ID.

        Args:
            caller_id: Caller identification (phone number, etc.)

        Returns:
            Tenant ID associated with the caller

        Raises:
            KeyError: If caller ID is not mapped to any tenant
        """
        if caller_id not in self._caller_mapping:
            raise KeyError(f"No tenant mapping found for caller ID: {caller_id}")

        return self._caller_mapping[caller_id]

    def reload_config(self, tenant_id: str) -> None:
        """Hot-reload configuration for a specific tenant.

        Reloads configuration from file and updates cache.

        Args:
            tenant_id: Unique tenant identifier

        Raises:
            FileNotFoundError: If configuration file not found
            ValueError: If configuration is invalid
        """
        logger.info(f"Reloading configuration for tenant: {tenant_id}")

        # Load fresh configuration from file
        config = self._load_config_from_file(tenant_id)

        # Update cache
        with self._cache_lock:
            self._cache[tenant_id] = config

        logger.info(f"Successfully reloaded configuration for tenant: {tenant_id}")

    def start_watching(self) -> None:
        """Start watching configuration directory for file changes.

        Automatically reloads configurations when files are modified.
        """
        if self._observer is not None:
            logger.warning("File watching is already active")
            return

        self._file_handler = _ConfigFileHandler(self)
        self._observer = Observer()
        self._observer.schedule(self._file_handler, str(self.config_dir), recursive=False)
        self._observer.start()

        logger.info(f"Started watching configuration directory: {self.config_dir}")

    def stop_watching(self) -> None:
        """Stop watching configuration directory for file changes."""
        if self._observer is None:
            logger.warning("File watching is not active")
            return

        self._observer.stop()
        self._observer.join()
        self._observer = None
        self._file_handler = None

        logger.info("Stopped watching configuration directory")

    def clear_cache(self) -> None:
        """Clear all cached configurations."""
        with self._cache_lock:
            self._cache.clear()
        logger.info("Cleared configuration cache")

    def get_cached_tenant_ids(self) -> list[str]:
        """Get list of tenant IDs currently in cache.

        Returns:
            List of cached tenant IDs
        """
        with self._cache_lock:
            return list(self._cache.keys())

    def add_caller_mapping(self, caller_id: str, tenant_id: str) -> None:
        """Add a caller ID to tenant ID mapping.

        Args:
            caller_id: Caller identification
            tenant_id: Tenant ID to map to
        """
        self._caller_mapping[caller_id] = tenant_id
        logger.debug(f"Added caller mapping: {caller_id} -> {tenant_id}")

    def remove_caller_mapping(self, caller_id: str) -> None:
        """Remove a caller ID mapping.

        Args:
            caller_id: Caller identification to remove
        """
        if caller_id in self._caller_mapping:
            del self._caller_mapping[caller_id]
            logger.debug(f"Removed caller mapping: {caller_id}")


class _ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration file changes."""

    def __init__(self, registry: TenantRegistry):
        """Initialize the file handler.

        Args:
            registry: TenantRegistry instance to notify of changes
        """
        super().__init__()
        self.registry = registry
        self._last_modified: dict[str, float] = {}
        self._debounce_seconds = 1.0  # Debounce file system events

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events.

        Args:
            event: File system event
        """
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Only process configuration files
        if file_path.suffix.lower() not in [".yaml", ".yml", ".toml"]:
            return

        # Debounce rapid file system events
        current_time = time.time()
        last_modified = self._last_modified.get(event.src_path, 0)
        if current_time - last_modified < self._debounce_seconds:
            return

        self._last_modified[event.src_path] = current_time

        # Extract tenant ID from filename
        tenant_id = file_path.stem

        try:
            self.registry.reload_config(tenant_id)
            logger.info(f"Auto-reloaded configuration for tenant: {tenant_id}")
        except Exception as e:
            logger.error(f"Failed to reload configuration for tenant {tenant_id}: {e}")
