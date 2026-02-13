"""Basic tests to verify test framework setup."""

from hypothesis import given
from hypothesis import strategies as st


def test_pytest_works() -> None:
    """Verify pytest is working."""
    assert True


@given(st.integers())
def test_hypothesis_works(x: int) -> None:
    """Verify Hypothesis property-based testing is working."""
    assert isinstance(x, int)


def test_fixtures_work(test_data_dir) -> None:
    """Verify pytest fixtures are working."""
    assert test_data_dir.name == "data"


def test_env_setup() -> None:
    """Verify environment setup fixture is working."""
    import os

    assert os.getenv("OPENAI_API_KEY") == "test-openai-key"
    assert os.getenv("LOG_LEVEL") == "DEBUG"
