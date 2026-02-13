"""Example tests demonstrating different testing patterns.

This file serves as a reference for writing tests in the project.
"""

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

# ============================================================================
# UNIT TESTS
# ============================================================================

@pytest.mark.unit
def test_simple_assertion() -> None:
    """Example of a simple unit test with assertion."""
    result = 2 + 2
    assert result == 4


@pytest.mark.unit
def test_with_fixture(test_data_dir: Path) -> None:
    """Example of using a fixture from conftest.py."""
    assert test_data_dir.exists()
    assert test_data_dir.name == "data"


@pytest.mark.unit
def test_exception_handling() -> None:
    """Example of testing exception handling."""
    with pytest.raises(ValueError, match="invalid literal"):
        int("not a number")


@pytest.mark.unit
@pytest.mark.parametrize("input_val,expected", [
    (0, 0),
    (1, 1),
    (2, 4),
    (3, 9),
    (-2, 4),
])
def test_parametrized(input_val: int, expected: int) -> None:
    """Example of parametrized test with multiple inputs."""
    assert input_val ** 2 == expected


# ============================================================================
# PROPERTY-BASED TESTS
# ============================================================================

@pytest.mark.property
@given(st.integers())
def test_property_addition_commutative(x: int) -> None:
    """Property: Addition is commutative (a + b = b + a)."""
    y = 5
    assert x + y == y + x


@pytest.mark.property
@given(st.text())
def test_property_string_length(s: str) -> None:
    """Property: Length of string is always non-negative."""
    assert len(s) >= 0


@pytest.mark.property
@given(st.lists(st.integers()))
def test_property_list_reverse(lst: list[int]) -> None:
    """Property: Reversing a list twice returns the original list."""
    assert list(reversed(list(reversed(lst)))) == lst


@pytest.mark.property
@given(st.integers(min_value=0, max_value=100))
def test_property_bounded_integers(x: int) -> None:
    """Property: Bounded integers stay within bounds."""
    assert 0 <= x <= 100


# ============================================================================
# ASYNC TESTS
# ============================================================================

@pytest.mark.unit
async def test_async_function() -> None:
    """Example of testing an async function."""
    async def async_add(a: int, b: int) -> int:
        return a + b

    result = await async_add(2, 3)
    assert result == 5


@pytest.mark.unit
async def test_async_with_fixture(test_data_dir: Path) -> None:
    """Example of using fixtures in async tests."""
    assert test_data_dir.exists()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
def test_integration_example() -> None:
    """Example of an integration test.

    Integration tests verify that multiple components work together correctly.
    """
    # This would test interaction between multiple components
    pass


# ============================================================================
# SLOW TESTS
# ============================================================================

@pytest.mark.slow
def test_slow_operation() -> None:
    """Example of a test marked as slow.

    Run slow tests with: pytest -m slow
    Skip slow tests with: pytest -m "not slow"
    """
    pass


# ============================================================================
# TESTS REQUIRING EXTERNAL APIS
# ============================================================================

@pytest.mark.requires_api
def test_external_api() -> None:
    """Example of a test that requires external API access.

    Run API tests with: pytest -m requires_api
    Skip API tests with: pytest -m "not requires_api"
    """
    pass


# ============================================================================
# HELPER FUNCTIONS FOR TESTS
# ============================================================================

def create_test_data() -> dict:
    """Helper function to create test data."""
    return {
        "tenant_id": "test_tenant",
        "name": "Test Tenant",
        "active": True,
    }


# ============================================================================
# FIXTURES SPECIFIC TO THIS TEST FILE
# ============================================================================

@pytest.fixture
def sample_data() -> dict:
    """Fixture providing sample data for tests in this file."""
    return create_test_data()


@pytest.mark.unit
def test_with_local_fixture(sample_data: dict) -> None:
    """Example of using a fixture defined in this file."""
    assert sample_data["tenant_id"] == "test_tenant"
    assert sample_data["active"] is True
