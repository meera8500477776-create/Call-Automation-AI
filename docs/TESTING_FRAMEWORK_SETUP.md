# Testing Framework Setup

## Overview

The testing framework for the Config-Driven Voice Agent Platform has been configured with pytest and Hypothesis for property-based testing. This document describes the setup, configuration, and usage.

## Python Version Requirement

**IMPORTANT**: This project requires Python 3.10 or higher due to the `pipecat-ai` dependency.

Current system Python version: 3.9.6
Required Python version: >=3.10

### Installing Python 3.10+

#### macOS (using Homebrew)
```bash
brew install python@3.10
# or
brew install python@3.11
```

#### macOS (using pyenv)
```bash
brew install pyenv
pyenv install 3.10.13
pyenv local 3.10.13
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

#### Windows
Download and install from [python.org](https://www.python.org/downloads/)

## Testing Framework Components

### 1. pytest
- **Version**: >=7.4.0
- **Purpose**: Main testing framework
- **Configuration**: `pyproject.toml` under `[tool.pytest.ini_options]`

### 2. pytest-asyncio
- **Version**: >=0.21.0
- **Purpose**: Support for async/await test functions
- **Configuration**: `asyncio_mode = "auto"` in pytest config

### 3. pytest-cov
- **Version**: >=4.1.0
- **Purpose**: Code coverage reporting
- **Configuration**: Generates both terminal and HTML coverage reports

### 4. Hypothesis
- **Version**: >=6.92.0
- **Purpose**: Property-based testing library
- **Usage**: Generate test cases automatically to verify universal properties

## Project Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── test_basic.py            # Basic framework verification tests
├── config/                  # Configuration system tests
│   ├── test_settings.py
│   └── test_loader.py
└── monitoring/              # Monitoring system tests
    └── test_logger.py
```

## Configuration Files

### pyproject.toml

The main configuration file includes:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
]
```

### conftest.py

Shared fixtures include:
- `test_data_dir`: Path to test data directory
- `temp_config_dir`: Temporary directory for test configurations
- `env_setup`: Automatic environment variable setup for all tests

## Installation

### Step 1: Create Virtual Environment with Python 3.10+

```bash
# Using python3.10 explicitly
python3.10 -m venv venv

# Or using pyenv
pyenv local 3.10.13
python -m venv venv
```

### Step 2: Activate Virtual Environment

```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install project with development dependencies
pip install -e ".[dev]"

# Or install from requirements.txt
pip install -r requirements.txt
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_basic.py
```

### Run with Verbose Output
```bash
pytest -v
```

### Run with Coverage Report
```bash
pytest --cov=src --cov-report=term-missing
```

### Run Property-Based Tests Only
```bash
pytest -k "hypothesis" -v
```

### Run Specific Test Function
```bash
pytest tests/test_basic.py::test_pytest_works
```

## Coverage Reporting

Coverage reports are generated in two formats:

1. **Terminal Output**: Shows coverage percentage and missing lines
2. **HTML Report**: Detailed interactive report in `htmlcov/index.html`

To view HTML coverage report:
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Writing Tests

### Unit Tests

```python
def test_example() -> None:
    """Test description."""
    result = some_function()
    assert result == expected_value
```

### Async Tests

```python
async def test_async_example() -> None:
    """Test async function."""
    result = await some_async_function()
    assert result == expected_value
```

### Property-Based Tests

```python
from hypothesis import given, strategies as st

@given(st.integers())
def test_property_example(x: int) -> None:
    """Test universal property."""
    result = some_function(x)
    assert isinstance(result, int)
```

### Using Fixtures

```python
def test_with_fixture(test_data_dir) -> None:
    """Test using shared fixture."""
    config_file = test_data_dir / "config.yaml"
    assert config_file.exists()
```

## Test Environment Variables

The following environment variables are automatically set for all tests:

- `OPENAI_API_KEY`: test-openai-key
- `DEEPGRAM_API_KEY`: test-deepgram-key
- `ELEVENLABS_API_KEY`: test-elevenlabs-key
- `TWILIO_ACCOUNT_SID`: test-twilio-sid
- `TWILIO_AUTH_TOKEN`: test-twilio-token
- `ENCRYPTION_KEY`: test-encryption-key-32-bytes-long!
- `LOG_LEVEL`: DEBUG
- `LOG_FORMAT`: console

## Code Quality Tools

### Black (Code Formatter)
```bash
black src tests
```

### Ruff (Linter)
```bash
ruff check src tests
```

### MyPy (Type Checker)
```bash
mypy src
```

## Continuous Integration

The testing framework is designed to work with CI/CD pipelines. Example GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v3
```

## Troubleshooting

### Issue: "No module named pytest"
**Solution**: Install development dependencies with `pip install -e ".[dev]"`

### Issue: "Python version mismatch"
**Solution**: Ensure you're using Python 3.10 or higher. Check with `python --version`

### Issue: "Import errors in tests"
**Solution**: Install the project in editable mode with `pip install -e .`

### Issue: "Coverage not showing all files"
**Solution**: Ensure you're running tests from the project root directory

## Next Steps

After verifying the testing framework setup:

1. Run basic tests to verify installation: `pytest tests/test_basic.py -v`
2. Check coverage: `pytest --cov=src --cov-report=term-missing`
3. Begin implementing feature tests following the task list
4. Write both unit tests and property-based tests for each component

## References

- [pytest documentation](https://docs.pytest.org/)
- [Hypothesis documentation](https://hypothesis.readthedocs.io/)
- [pytest-asyncio documentation](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
