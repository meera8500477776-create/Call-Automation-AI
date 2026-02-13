"""Example usage of TenantRegistry for multi-tenant configuration management."""

from pathlib import Path

from voice_agent.config import TenantRegistry

# Initialize the registry with the tenant configuration directory
config_dir = Path("config/tenants")
registry = TenantRegistry(config_dir)

# Example 1: Load a tenant configuration
print("=== Example 1: Load Tenant Configuration ===")
healthcare_config = registry.load_tenant_config("example_healthcare")
print(f"Tenant ID: {healthcare_config.tenant_id}")
print(f"Persona Name: {healthcare_config.persona.name}")
print(f"LLM Provider: {healthcare_config.llm.provider}")
print(f"Number of Tools: {len(healthcare_config.tools)}")
print()

# Example 2: Caller ID to Tenant Mapping
print("=== Example 2: Caller ID to Tenant Mapping ===")
# Add caller mappings
registry.add_caller_mapping("+15551234567", "example_healthcare")
registry.add_caller_mapping("+15559876543", "example_automotive")

# Look up tenant by caller ID
caller_id = "+15551234567"
tenant_id = registry.get_tenant_by_caller(caller_id)
print(f"Caller {caller_id} maps to tenant: {tenant_id}")

# Load the tenant configuration
config = registry.load_tenant_config(tenant_id)
print(f"Loaded configuration for: {config.persona.name}")
print()

# Example 3: Configuration Caching
print("=== Example 3: Configuration Caching ===")
print(f"Cached tenant IDs: {registry.get_cached_tenant_ids()}")

# Load another tenant
automotive_config = registry.load_tenant_config("example_automotive")
print(f"Loaded automotive tenant: {automotive_config.persona.name}")
print(f"Cached tenant IDs: {registry.get_cached_tenant_ids()}")
print()

# Example 4: Hot-Reload Configuration
print("=== Example 4: Hot-Reload Configuration ===")
print(f"Current persona name: {healthcare_config.persona.name}")

# In production, you would modify the config file here
# For this example, we'll just demonstrate the reload call
print("Calling reload_config() to refresh from file...")
registry.reload_config("example_healthcare")

# Load the refreshed configuration
refreshed_config = registry.load_tenant_config("example_healthcare")
print(f"Reloaded persona name: {refreshed_config.persona.name}")
print()

# Example 5: File Watching (Auto-Reload)
print("=== Example 5: File Watching ===")
print("Starting file watcher...")
registry.start_watching()
print("File watcher is now monitoring the config directory for changes.")
print("Any modifications to tenant config files will be automatically reloaded.")
print()

# In production, the watcher would run continuously
# For this example, we'll stop it immediately
print("Stopping file watcher...")
registry.stop_watching()
print("File watcher stopped.")
print()

# Example 6: Clear Cache
print("=== Example 6: Clear Cache ===")
print(f"Cached tenant IDs before clear: {registry.get_cached_tenant_ids()}")
registry.clear_cache()
print(f"Cached tenant IDs after clear: {registry.get_cached_tenant_ids()}")
print()

print("=== TenantRegistry Examples Complete ===")
