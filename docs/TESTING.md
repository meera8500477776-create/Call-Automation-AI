# Testing Guide

## Testing Framework

This project uses:
- **pytest**: Main testing framework
- **pytest-asyncio**: For async test support
- **pytest-cov**: For code coverage reporting
- **Hypothesis**: For property-based testing

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage
```bash
pytest --cov=src --cov-report=term-missing --cov-report=html
```

### Run specific test file
```bash
pytest tests/config/test_loader.py
```

### Run specific test
```bash
pytest tests/config/test_loader.py::test_load_yaml_valid
```

### Run tests matching pattern
```bash
pytest -k "yaml"
```

### Run with verbose output
```bash
pytest -v
```

### Run property-based tests with more examples
```bash
pytest --hypothesis-show-statistics
```

## Test Structure

### Unit Tests

Test individual functions and classes in isolation:

```python
def test_load_yaml_valid(temp_config_dir: Path) -> None:
    """Test loading valid YAML configuration."""
    config_file = temp_config_dir / "test.yaml"
    config_file.write_text("tenant_id: test")
    
    config = ConfigLoader.load_yaml(config_file)
    
    assert config["tenant_id"] == "test"
```

### Property-Based Tests

Test universal properties across many inputs using Hypothesis:

```python
from hypothesis import given, strategies as st

@given(st.text())
def test_encryption_round_trip(data: str) -> None:
    """Property: Encrypting then decrypting should return original data."""
    encrypted = encrypt(data)
    decrypted = decrypt(encrypted)
    assert decrypted == data
```

### Integration Tests

Test component interactions:

```python
@pytest.mark.asyncio
async def test_voice_pipeline_integration() -> None:
    """Test complete voice pipeline flow."""
    pipeline = VoicePipeline(config)
    result = await pipeline.process_audio(audio_data)
    assert result.transcript is not None
```

## Test Fixtures

### Built-in Fixtures

Defined in `tests/conftest.py`:

- **test_data_dir**: Path to test data directory
- **temp_config_dir**: Temporary directory for test configurations
- **env_setup**: Automatically sets up test environment variables

### Using Fixtures

```python
def test_with_fixture(temp_config_dir: Path) -> None:
    """Test using a fixture."""
    config_file = temp_config_dir / "test.yaml"
    # Use the temporary directory
```

## Coverage Reports

### Terminal Report

Shows coverage percentage and missing lines:
```bash
pytest --cov=src --cov-report=term-missing
```

### HTML Report

Generates detailed HTML coverage report:
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html  # View in browser
```

## Property-Based Testing Guidelines

### Writing Properties

Properties should express universal truths:

```python
@given(st.text(), st.text())
def test_concatenation_length(a: str, b: str) -> None:
    """Property: Length of concatenation equals sum of lengths."""
    result = a + b
    assert len(result) == len(a) + len(b)
```

### Common Property Patterns

1. **Round-trip**: Serialize then deserialize should return original
2. **Invariants**: Properties that always hold (e.g., sorted list is ordered)
3. **Idempotence**: Applying operation twice equals applying once
4. **Commutativity**: Order of operations doesn't matter
5. **Associativity**: Grouping of operations doesn't matter

### Hypothesis Strategies

Common strategies for generating test data:

```python
from hypothesis import strategies as st

# Basic types
st.integers()
st.floats()
st.text()
st.booleans()

# Collections
st.lists(st.integers())
st.dictionaries(keys=st.text(), values=st.integers())

# Custom strategies
st.builds(MyClass, field1=st.text(), field2=st.integers())
```

## Async Testing

For async functions, use `pytest-asyncio`:

```python
import pytest

@pytest.mark.asyncio
async def test_async_function() -> None:
    """Test async function."""
    result = await my_async_function()
    assert result is not None
```

## Mocking

Use pytest's monkeypatch or unittest.mock:

```python
def test_with_mock(monkeypatch):
    """Test with mocked dependency."""
    def mock_api_call(*args, **kwargs):
        return {"status": "success"}
    
    monkeypatch.setattr("module.api_call", mock_api_call)
    result = function_that_calls_api()
    assert result["status"] == "success"
```

## Test Organization

### Naming Conventions

- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`

### Directory Structure

Mirror the source structure:
```
src/voice_agent/config/loader.py
tests/config/test_loader.py
```

## Continuous Integration

Tests should:
- Run quickly (< 5 minutes for full suite)
- Be deterministic (no flaky tests)
- Have clear failure messages
- Clean up resources (files, connections)

## Best Practices

1. **One assertion per test** (when possible)
2. **Descriptive test names** that explain what is being tested
3. **Arrange-Act-Assert** pattern
4. **Test edge cases** (empty inputs, boundaries, errors)
5. **Use fixtures** for common setup
6. **Mock external dependencies** (APIs, databases)
7. **Write properties** for universal correctness
8. **Keep tests fast** by avoiding unnecessary I/O

## Debugging Tests

### Run with print statements
```bash
pytest -s
```

### Drop into debugger on failure
```bash
pytest --pdb
```

### Show local variables on failure
```bash
pytest -l
```

### Run last failed tests
```bash
pytest --lf
```

## Coverage Goals

- **Minimum**: 80% code coverage
- **Target**: 90% code coverage
- **Critical paths**: 100% coverage (security, encryption, authentication)

## Example Test Session

```bash
# Activate virtual environment
source venv/bin/activate

# Run tests with coverage
pytest --cov=src --cov-report=term-missing

# Check code quality
black --check src tests
ruff check src tests
mypy src

# View coverage report
open htmlcov/index.html
```
