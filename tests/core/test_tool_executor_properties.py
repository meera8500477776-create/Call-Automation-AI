"""Property-based tests for ToolExecutor using Hypothesis.

**Validates: Requirements 10.4**
"""

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from voice_agent.core.tool_executor import ToolDefinition, ToolExecutor


# JSON Schema strategies for generating valid and invalid parameters
def json_value_strategy():
    """Generate valid JSON values."""
    return st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(),
    )


def json_object_strategy():
    """Generate JSON objects (dictionaries)."""
    return st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=json_value_strategy(),
        min_size=0,
        max_size=10,
    )


class TestToolExecutorProperties:
    """Property-based tests for ToolExecutor."""

    @given(
        tool_name=st.text(min_size=1, max_size=50),
        required_param=st.text(min_size=1, max_size=20),
        param_value=st.one_of(st.integers(), st.text(), st.booleans()),
    )
    @settings(max_examples=100)
    def test_property_parameter_validation_rejects_missing_required(
        self, tool_name: str, required_param: str, param_value
    ):
        """Property 8: Parameter validation - For any tool with a parameter schema,
        invalid parameters should be rejected.

        **Validates: Requirements 10.4**

        This property tests that when a tool requires a parameter, calling it
        without that parameter results in validation failure.
        """
        executor = ToolExecutor()

        # Map Python types to JSON Schema types
        type_mapping = {
            int: "integer",
            str: "string",
            bool: "boolean",
        }
        json_type = type_mapping.get(type(param_value), "string")

        # Define a schema that requires a specific parameter
        schema = {
            "type": "object",
            "properties": {
                required_param: {"type": json_type}
            },
            "required": [required_param],
            "additionalProperties": False,
        }

        # Create a simple tool handler
        def handler(params):
            return {"result": "success"}

        tool_def = ToolDefinition(
            name=tool_name,
            description="Test tool",
            parameters_schema=schema,
            handler=handler,
        )

        executor.register_tool(tool_def)

        # Test with missing required parameter (empty dict)
        is_valid, error = executor.validate_parameters(tool_name, {})

        # Property: Validation should fail when required parameter is missing
        assert not is_valid, (
            f"Validation should fail when required parameter '{required_param}' is missing"
        )
        assert error is not None, "Error message should be provided"
        assert "required" in error.lower() or "missing" in error.lower(), (
            f"Error message should mention missing/required parameter: {error}"
        )

    @given(
        tool_name=st.text(min_size=1, max_size=50),
        param_name=st.text(min_size=1, max_size=20),
        correct_value=st.integers(),
        wrong_value=st.text(min_size=1),
    )
    @settings(max_examples=100)
    def test_property_parameter_validation_rejects_wrong_type(
        self, tool_name: str, param_name: str, correct_value: int, wrong_value: str
    ):
        """Property 8 (variant): Parameter validation - Parameters with wrong types should be rejected.

        **Validates: Requirements 10.4**

        This property tests that when a tool expects an integer parameter,
        providing a string value results in validation failure.
        """
        executor = ToolExecutor()

        # Define a schema that expects an integer
        schema = {
            "type": "object",
            "properties": {
                param_name: {"type": "integer"}
            },
            "required": [param_name],
            "additionalProperties": False,
        }

        def handler(params):
            return {"result": "success"}

        tool_def = ToolDefinition(
            name=tool_name,
            description="Test tool",
            parameters_schema=schema,
            handler=handler,
        )

        executor.register_tool(tool_def)

        # Test with wrong type (string instead of integer)
        is_valid, error = executor.validate_parameters(
            tool_name, {param_name: wrong_value}
        )

        # Property: Validation should fail when parameter type is wrong
        assert not is_valid, (
            f"Validation should fail when parameter '{param_name}' has wrong type "
            f"(expected integer, got string)"
        )
        assert error is not None, "Error message should be provided"

    @given(
        tool_name=st.text(min_size=1, max_size=50),
        param_name=st.text(min_size=1, max_size=20),
        param_value=st.integers(),
    )
    @settings(max_examples=100)
    def test_property_parameter_validation_accepts_valid(
        self, tool_name: str, param_name: str, param_value: int
    ):
        """Property 8 (variant): Parameter validation - Valid parameters should be accepted.

        **Validates: Requirements 10.4**

        This property tests that when parameters match the schema,
        validation succeeds.
        """
        executor = ToolExecutor()

        # Define a schema
        schema = {
            "type": "object",
            "properties": {
                param_name: {"type": "integer"}
            },
            "required": [param_name],
            "additionalProperties": False,
        }

        def handler(params):
            return {"result": "success"}

        tool_def = ToolDefinition(
            name=tool_name,
            description="Test tool",
            parameters_schema=schema,
            handler=handler,
        )

        executor.register_tool(tool_def)

        # Test with valid parameters
        is_valid, error = executor.validate_parameters(
            tool_name, {param_name: param_value}
        )

        # Property: Validation should succeed when parameters are valid
        assert is_valid, (
            f"Validation should succeed for valid parameters: {error}"
        )
        assert error is None, "Error message should be None for valid parameters"

    @given(
        tool_name=st.text(min_size=1, max_size=50),
        param_name=st.text(min_size=1, max_size=20),
        min_value=st.integers(min_value=0, max_value=100),
        test_value=st.integers(min_value=-1000, max_value=-1),
    )
    @settings(max_examples=100)
    def test_property_parameter_validation_rejects_out_of_range(
        self, tool_name: str, param_name: str, min_value: int, test_value: int
    ):
        """Property 8 (variant): Parameter validation - Out-of-range values should be rejected.

        **Validates: Requirements 10.4**

        This property tests that when a parameter has a minimum value constraint,
        values below that minimum are rejected.
        """
        executor = ToolExecutor()

        # Define a schema with minimum constraint
        schema = {
            "type": "object",
            "properties": {
                param_name: {
                    "type": "integer",
                    "minimum": min_value
                }
            },
            "required": [param_name],
            "additionalProperties": False,
        }

        def handler(params):
            return {"result": "success"}

        tool_def = ToolDefinition(
            name=tool_name,
            description="Test tool",
            parameters_schema=schema,
            handler=handler,
        )

        executor.register_tool(tool_def)

        # Test with value below minimum
        is_valid, error = executor.validate_parameters(
            tool_name, {param_name: test_value}
        )

        # Property: Validation should fail when value is out of range
        assert not is_valid, (
            f"Validation should fail when parameter '{param_name}' value {test_value} "
            f"is below minimum {min_value}"
        )
        assert error is not None, "Error message should be provided"

    @given(
        tool_name=st.text(min_size=1, max_size=50),
        required_params=st.lists(
            st.text(min_size=1, max_size=20),
            min_size=1,
            max_size=5,
            unique=True,
        ),
    )
    @settings(max_examples=50)
    def test_property_parameter_validation_rejects_partial_params(
        self, tool_name: str, required_params: list
    ):
        """Property 8 (variant): Parameter validation - Partial parameters should be rejected.

        **Validates: Requirements 10.4**

        This property tests that when a tool requires multiple parameters,
        providing only some of them results in validation failure.
        """
        executor = ToolExecutor()

        # Define a schema with multiple required parameters
        properties = {param: {"type": "string"} for param in required_params}
        schema = {
            "type": "object",
            "properties": properties,
            "required": required_params,
            "additionalProperties": False,
        }

        def handler(params):
            return {"result": "success"}

        tool_def = ToolDefinition(
            name=tool_name,
            description="Test tool",
            parameters_schema=schema,
            handler=handler,
        )

        executor.register_tool(tool_def)

        # Provide only the first parameter (partial)
        partial_params = {required_params[0]: "value"}

        is_valid, error = executor.validate_parameters(tool_name, partial_params)

        # Property: Validation should fail when not all required parameters are provided
        if len(required_params) > 1:
            assert not is_valid, (
                f"Validation should fail when only {len(partial_params)} of "
                f"{len(required_params)} required parameters are provided"
            )
            assert error is not None, "Error message should be provided"

    @given(
        tool_name=st.text(min_size=1, max_size=50),
        allowed_params=st.lists(
            st.text(min_size=1, max_size=20),
            min_size=1,
            max_size=3,
            unique=True,
        ),
        extra_param=st.text(min_size=1, max_size=20),
    )
    @settings(max_examples=50)
    def test_property_parameter_validation_rejects_extra_params(
        self, tool_name: str, allowed_params: list, extra_param: str
    ):
        """Property 8 (variant): Parameter validation - Extra parameters should be rejected.

        **Validates: Requirements 10.4**

        This property tests that when additionalProperties is false,
        providing extra parameters results in validation failure.
        """
        # Ensure extra_param is not in allowed_params
        assume(extra_param not in allowed_params)

        executor = ToolExecutor()

        # Define a schema that doesn't allow additional properties
        properties = {param: {"type": "string"} for param in allowed_params}
        schema = {
            "type": "object",
            "properties": properties,
            "required": allowed_params,
            "additionalProperties": False,
        }

        def handler(params):
            return {"result": "success"}

        tool_def = ToolDefinition(
            name=tool_name,
            description="Test tool",
            parameters_schema=schema,
            handler=handler,
        )

        executor.register_tool(tool_def)

        # Provide all required params plus an extra one
        params_with_extra = {param: "value" for param in allowed_params}
        params_with_extra[extra_param] = "extra_value"

        is_valid, error = executor.validate_parameters(tool_name, params_with_extra)

        # Property: Validation should fail when extra parameters are provided
        assert not is_valid, (
            f"Validation should fail when extra parameter '{extra_param}' is provided"
        )
        assert error is not None, "Error message should be provided"

    @given(
        tool_name=st.text(min_size=1, max_size=50),
        param_name=st.text(min_size=1, max_size=20),
        enum_values=st.lists(
            st.text(min_size=1, max_size=20),
            min_size=2,
            max_size=5,
            unique=True,
        ),
        invalid_value=st.text(min_size=1, max_size=20),
    )
    @settings(max_examples=50)
    def test_property_parameter_validation_rejects_invalid_enum(
        self, tool_name: str, param_name: str, enum_values: list, invalid_value: str
    ):
        """Property 8 (variant): Parameter validation - Values not in enum should be rejected.

        **Validates: Requirements 10.4**

        This property tests that when a parameter has an enum constraint,
        values not in the enum are rejected.
        """
        # Ensure invalid_value is not in enum_values
        assume(invalid_value not in enum_values)

        executor = ToolExecutor()

        # Define a schema with enum constraint
        schema = {
            "type": "object",
            "properties": {
                param_name: {
                    "type": "string",
                    "enum": enum_values
                }
            },
            "required": [param_name],
            "additionalProperties": False,
        }

        def handler(params):
            return {"result": "success"}

        tool_def = ToolDefinition(
            name=tool_name,
            description="Test tool",
            parameters_schema=schema,
            handler=handler,
        )

        executor.register_tool(tool_def)

        # Test with value not in enum
        is_valid, error = executor.validate_parameters(
            tool_name, {param_name: invalid_value}
        )

        # Property: Validation should fail when value is not in enum
        assert not is_valid, (
            f"Validation should fail when parameter '{param_name}' value '{invalid_value}' "
            f"is not in enum {enum_values}"
        )
        assert error is not None, "Error message should be provided"
