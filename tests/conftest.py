"""Pytest configuration and shared fixtures."""

import os
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def temp_config_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create temporary directory for test configurations."""
    return tmp_path_factory.mktemp("config")


@pytest.fixture(autouse=True)
def env_setup(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Set up test environment variables."""
    # Set test environment variables
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-deepgram-key")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-elevenlabs-key")
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "test-twilio-sid")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-twilio-token")
    monkeypatch.setenv("ENCRYPTION_KEY", "test-encryption-key-32-bytes-long!")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FORMAT", "console")

    yield

    # Cleanup is automatic with monkeypatch
