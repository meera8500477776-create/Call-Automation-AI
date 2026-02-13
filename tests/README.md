# Test Suite

This directory contains the test suite for the Config-Driven Voice Agent Platform.

## Directory Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and pytest configuration
├── test_basic.py            # Framework verification tests
├── config/                  # Configuration system tests
│   ├── test_settings.py
│   └── test_loader.py
├── monitoring/              # Monitoring and logging tests
│   └── test_logger.py
└── data/                    # Test data files (created as needed)
```

## Test Categories

### 1. Unit Tests
Test individual functions, classes, and methods in isolation.

**Naming Convention**: `test_<component>_<functionality>.py`

**Example**:
```python
def test_tenant_config_loading() -> None:
    """Test that tenant configuration loads correctly from YAML."""
    config = load_tenant_config("test_tenant.yaml")
    assert config.tenant_id == "test_tenant"
    assert config.persona.name == "Sarah"
```

### 2. Property-Based Tests
Test universal properties that should hold for all inputs using Hypothesis.

**Naming Convention**: Include "property" in test name or docstring

**Example**:
```python
from hypothesis import given, strategies as st

@given(st.text(), st.text())
def test_encryption_round_trip_property(data: str, key_id: str) -> None:
    """Property: Encrypting then decrypting should return original data."""
    encrypted = encrypt_data(data.encode(), key_id)
    decrypted = decrypt_data(encrypted, key_id)
    assert decrypted.decode() == data
```

### 3. Integration Tests
Test interactions between multiple components.

**Naming Convention**: `test_integration_<feature>.py`

**Example**:
```python
async def test_voice_pipeline_integration() -> None:
    """Test complete voice pipeline from audio input to output."""
    pipeline = VoicePipeline(config)
    audio_input = load_test_audio("sample.wav")
    
    result = await pipeline.process_audio_chunk(audio_input)
    
    assert result.transcript is not None
    assert result.llm_response is not None
    assert result.audio_output is not None
```

### 4. End-to-End Tests
Test complete workflows from start to finish.

**Naming Convention**: `test_e2e_<workflow>.py`

**Example**:
```python
async def test_e2e_appointment_scheduling() -> None:
    """Test complete appointment scheduling workflow."""
    # Simulate incoming call
    session = await orchestrator.start_session(caller_id="555-1234")
    
    # User: "I need to schedule an appointment"
    await session.process_transcript("I need to schedule an appointment")
    
    # User: "Next Tuesday at 2pm"
    await session.process_transcript("Next Tuesday at 2pm")
    
    # Verify appointment was booked
    assert session.current_intent == "schedule_appointment"
    assert session.extracted_parameters["date"] == "next_tuesday"
    assert session.extracted_parameters["time"] == "14:00"
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Category
```bash
# Unit tests only
pytest tests/config/

# Property-based tests only
pytest -k "property"

# Integration tests only
pytest -k "integration"
```

### Run with Coverage
```bash
pytest --cov=src --cov-report=term-missing --cov-report=html
```

### Run Specific Test
```bash
pytest tests/test_basic.py::test_pytest_works -v
```

## Writing New Tests

### Step 1: Choose Test Type
- **Unit Test**: Testing a single function/class
- **Property Test**: Testing a universal property
- **Integration Test**: Testing component interactions
- **E2E Test**: Testing complete workflows

### Step 2: Create Test File
Place test file in appropriate directory:
- Configuration tests → `tests/config/`
- Pipeline tests → `tests/pipeline/`
- Adapter tests → `tests/adapters/`
- etc.

### Step 3: Write Test
Follow the pattern:
```python
def test_<what_you_are_testing>() -> None:
    """Clear description of what is being tested."""
    # Arrange: Set up test data
    input_data = create_test_data()
    
    # Act: Execute the code under test
    result = function_under_test(input_data)
    
    # Assert: Verify the result
    assert result == expected_value
```

### Step 4: Run Test
```bash
pytest tests/your_test_file.py -v
```

## Test Fixtures

Common fixtures are defined in `conftest.py`:

### test_data_dir
Path to test data directory for loading test files.

```python
def test_with_data(test_data_dir) -> None:
    config_file = test_data_dir / "config.yaml"
    # Use config_file...
```

### temp_config_dir
Temporary directory for creating test configurations.

```python
def test_with_temp_config(temp_config_dir) -> None:
    config_file = temp_config_dir / "test.yaml"
    config_file.write_text("tenant_id: test")
    # Use config_file...
```

### env_setup
Automatically sets up test environment variables (applied to all tests).

## Property-Based Testing Guidelines

### 1. Define Clear Properties
Properties should be universal truths about your code:
- **Round-trip**: Encode then decode returns original
- **Idempotence**: Applying operation twice = applying once
- **Invariants**: Certain conditions always hold
- **Commutativity**: Order doesn't matter

### 2. Use Appropriate Strategies
Hypothesis provides strategies for generating test data:

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

# Custom types
st.builds(TenantConfig, tenant_id=st.text(), ...)
```

### 3. Set Reasonable Limits
```python
@given(st.integers(min_value=0, max_value=1000))
def test_with_bounded_integers(x: int) -> None:
    # Test with integers between 0 and 1000
    pass
```

### 4. Handle Edge Cases
```python
from hypothesis import assume

@given(st.integers())
def test_division(x: int) -> None:
    assume(x != 0)  # Skip zero to avoid division by zero
    result = 100 / x
    assert result * x == 100
```

## Mocking External Services

For tests that interact with external APIs, use mocks:

```python
from unittest.mock import Mock, patch

def test_with_mock_api() -> None:
    """Test with mocked external API."""
    with patch('module.external_api_call') as mock_api:
        mock_api.return_value = {"status": "success"}
        
        result = function_that_calls_api()
        
        assert result == expected_value
        mock_api.assert_called_once()
```

## Async Testing

For async functions, use `async def` and `await`:

```python
async def test_async_function() -> None:
    """Test async function."""
    result = await async_function()
    assert result == expected_value
```

pytest-asyncio automatically handles async test execution.

## Test Data

Test data files should be placed in `tests/data/`:

```
tests/data/
├── configs/
│   ├── valid_tenant.yaml
│   └── invalid_tenant.yaml
├── audio/
│   ├── sample_speech.wav
│   └── silence.wav
└── fhir/
    ├── appointment_response.json
    └── slot_response.json
```

## Coverage Goals

- **Overall Coverage**: >80%
- **Critical Components**: >90%
  - Configuration loading
  - Security/encryption
  - FHIR adapter
  - Voice pipeline

## Best Practices

1. **Test One Thing**: Each test should verify one specific behavior
2. **Clear Names**: Test names should describe what is being tested
3. **Arrange-Act-Assert**: Follow the AAA pattern
4. **Independent Tests**: Tests should not depend on each other
5. **Fast Tests**: Keep tests fast by mocking external services
6. **Readable Tests**: Tests are documentation - make them clear
7. **Type Hints**: Use type hints for better IDE support and clarity

## Common Patterns

### Testing Exceptions
```python
import pytest

def test_raises_exception() -> None:
    """Test that function raises expected exception."""
    with pytest.raises(ValueError, match="Invalid input"):
        function_that_should_raise("invalid")
```

### Parametrized Tests
```python
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("", ""),
])
def test_uppercase(input: str, expected: str) -> None:
    """Test uppercase conversion with multiple inputs."""
    assert input.upper() == expected
```

### Testing Async Generators
```python
async def test_async_generator() -> None:
    """Test async generator."""
    results = []
    async for item in async_generator():
        results.append(item)
    assert len(results) > 0
```

## Debugging Tests

### Run with Print Statements
```bash
pytest -s  # Don't capture stdout
```

### Run with Debugger
```bash
pytest --pdb  # Drop into debugger on failure
```

### Run Last Failed Tests
```bash
pytest --lf  # Run only tests that failed last time
```

### Verbose Output
```bash
pytest -vv  # Extra verbose output
```

## CI/CD Integration

Tests are designed to run in CI/CD pipelines. Ensure:
1. All tests pass before merging
2. Coverage doesn't decrease
3. No new linting errors
4. Type checking passes

## Questions?

For questions about testing:
1. Check this README
2. Check `docs/TESTING_FRAMEWORK_SETUP.md`
3. Look at existing tests for examples
4. Refer to pytest/Hypothesis documentation
