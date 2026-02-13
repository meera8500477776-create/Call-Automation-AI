# Setup Notes

## Python Version Requirement

This project requires **Python 3.10 or higher** due to dependencies:
- `pipecat-ai` requires Python 3.10+
- Modern type hints (e.g., `list[str]`) require Python 3.10+

## Installation Options

### Option 1: Using pyenv (Recommended)

If you don't have Python 3.10+, install it using pyenv:

```bash
# Install pyenv (if not already installed)
curl https://pyenv.run | bash

# Install Python 3.10 or higher
pyenv install 3.10.13

# Set local Python version for this project
pyenv local 3.10.13

# Verify version
python --version  # Should show Python 3.10.13
```

### Option 2: Using conda

```bash
# Create conda environment with Python 3.10
conda create -n voice-agent python=3.10

# Activate environment
conda activate voice-agent

# Install dependencies
pip install -e ".[dev]"
```

### Option 3: System Python

If your system has Python 3.10+:

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## Quick Setup

Once you have Python 3.10+:

```bash
# Run the setup script
./setup.sh

# Or manually:
python3.10 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your API keys
```

## Verification

```bash
# Check Python version
python --version

# Run tests
pytest

# Check code quality
black --check src tests
ruff check src tests
mypy src
```

## Common Issues

### Issue: "Package requires a different Python"

**Solution**: Ensure you're using Python 3.10 or higher. Check with `python --version`.

### Issue: "pipecat-ai not found"

**Solution**: This usually means Python version is too old. Pipecat requires 3.10+.

### Issue: Virtual environment activation fails

**Solution**: 
- On Linux/Mac: `source venv/bin/activate`
- On Windows: `venv\Scripts\activate`
- On Windows PowerShell: `venv\Scripts\Activate.ps1`
