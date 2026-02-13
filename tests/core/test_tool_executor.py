"""Unit tests for ToolExecutor.

Requirements: 10.4, 10.5
"""

import pytest
from voice_agent.core.tool_executor import (
    ToolExecutor,
    ToolDefinition,
    ToolResult,
)


class TestToolExecutor:
    """Unit tests for ToolExecutor."""

    def test_register_tool_success(self):
        """Test successful tool registration."""
        executor = ToolExecutor()

        def handler(params):
            return {"result": "success"}

        tool_def = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters_schema={
                "type": "object",
                "properties": {"param1": {"type": "string"}},
                "required": ["param1"],
            },
            handler=handler,
        )

        executor.register_tool(tool_def)

        # Verify tool is registered
        assert "test_tool" in executor.list_tools()
        assert executor.get_tool("test_tool") == tool_def

    def test_register_tool_duplicate_name_raises_error(self):
        """Test that registering a tool with duplicate name raises ValueError."""
        executor = ToolExecutor()

        def handler(params):
            return {"result": "success"}

        tool_def1 = ToolDefinition(
            name="test_tool",
            description="First tool",
            parameters_schema={"type": "object"},
            handler=handler,
        )

        tool_def2 = ToolDefinition(
            name="test_tool",
            description="Second tool",
            parameters_schema={"type": "object"},
            handler=handler,
        )

        executor.register_tool(tool_def1)

        # Registering duplicate should raise ValueError
        with pytest.raises(ValueError, match="already registered"):
            executor.register_tool(tool_def2)

    def test_validate_parameters_valid_schema(self):
        """Test parameter validation with valid parameters."""
        executor = ToolExecutor()

        def handler(params):
            return {"result": "success"}

        tool_def = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0},
                },
                "required": ["name", "age"],
            },
            handler=handler,
        )

        executor.register_tool(tool_def)

        # Valid parameters
        is_valid, error = executor.validate_parameters(
            "test_tool", {"name": "John", "age": 30}
        )

        assert is_valid
        assert error is None

    def test_validate_parameters_missing_required(self):
        """Test parameter validation with missing required parameter."""
        executor = ToolExecutor()

        def handler(params):
            return {"result": "success"}

        tool_def = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
            handler=handler,
        )

        executor.register_tool(tool_def)

        # Missing 'age' parameter
        is_valid, error = executor.validate_parameters("test_tool", {"name": "John"})

        assert not is_valid
        assert error is not None
        assert "age" in error or "required" in error.lower()

    def test_validate_parameters_wrong_type(self):
        """Test parameter validation with wrong parameter type."""
        executor = ToolExecutor()

        def handler(params):
            return {"result": "success"}

        tool_def = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters_schema={
                "type": "object",
                "properties": {
                    "age": {"type": "integer"},
                },
                "required": ["age"],
            },
            handler=handler,
        )

        executor.register_tool(tool_def)

        # Wrong type: string instead of integer
        is_valid, error = executor.validate_parameters("test_tool", {"age": "thirty"})

        assert not is_valid
        assert error is not None

    def test_validate_parameters_unregistered_tool(self):
        """Test parameter validation for unregistered tool raises ValueError."""
        executor = ToolExecutor()

        with pytest.raises(ValueError, match="not registered"):
            executor.validate_parameters("nonexistent_tool", {})

    def test_execute_tool_success(self):
        """Test successful tool execution."""
        executor = ToolExecutor()

        def handler(params):
            return {"greeting": f"Hello, {params['name']}!"}

        tool_def = ToolDefinition(
            name="greet",
            description="Greet a person",
            parameters_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=handler,
        )

        executor.register_tool(tool_def)

        result = executor.execute_tool("greet", {"name": "Alice"})

        assert result.success
        assert result.data == {"greeting": "Hello, Alice!"}
        assert result.error is None
        assert result.execution_time_ms >= 0

    def test_execute_tool_unregistered(self):
        """Test executing unregistered tool returns error result."""
        executor = ToolExecutor()

        result = executor.execute_tool("nonexistent_tool", {})

        assert not result.success
        assert result.error is not None
        assert "not registered" in result.error
        assert result.data is None

    def test_execute_tool_invalid_parameters(self):
        """Test executing tool with invalid parameters returns error result."""
        executor = ToolExecutor()

        def handler(params):
            return {"result": "success"}

        tool_def = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters_schema={
                "type": "object",
                "properties": {"age": {"type": "integer"}},
                "required": ["age"],
            },
            handler=handler,
        )

        executor.register_tool(tool_def)

        # Invalid parameters (missing required 'age')
        result = executor.execute_tool("test_tool", {})

        assert not result.success
        assert result.error is not None
        assert "validation failed" in result.error.lower()
        assert result.data is None

    def test_execute_tool_handler_exception(self):
        """Test tool execution when handler raises exception."""
        executor = ToolExecutor()

        def handler(params):
            raise RuntimeError("Something went wrong")

        tool_def = ToolDefinition(
            name="failing_tool",
            description="A tool that fails",
            parameters_schema={
                "type": "object",
                "properties": {"param": {"type": "string"}},
                "required": ["param"],
            },
            handler=handler,
        )

        executor.register_tool(tool_def)

        result = executor.execute_tool("failing_tool", {"param": "value"})

        assert not result.success
        assert result.error is not None
        assert "RuntimeError" in result.error
        assert "Something went wrong" in result.error
        assert result.data is None
        assert result.execution_time_ms >= 0

    def test_execute_tool_with_context(self):
        """Test tool execution with context parameter."""
        executor = ToolExecutor()

        def handler(params):
            return {"result": f"Processed {params['data']}"}

        tool_def = ToolDefinition(
            name="process",
            description="Process data",
            parameters_schema={
                "type": "object",
                "properties": {"data": {"type": "string"}},
                "required": ["data"],
            },
            handler=handler,
        )

        executor.register_tool(tool_def)

        context = {"session_id": "123", "user_id": "456"}
        result = executor.execute_tool("process", {"data": "test"}, context=context)

        assert result.success
        assert result.data == {"result": "Processed test"}

    def test_handle_tool_error(self):
        """Test error handling returns user-friendly message."""
        executor = ToolExecutor()

        error = ValueError("Invalid input")
        error_msg = executor.handle_tool_error(error, "test_tool")

        assert "ValueError" in error_msg
        assert "Invalid input" in error_msg

    def test_get_tool_existing(self):
        """Test getting an existing tool."""
        executor = ToolExecutor()

        def handler(params):
            return {"result": "success"}

        tool_def = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters_schema={"type": "object"},
            handler=handler,
        )

        executor.register_tool(tool_def)

        retrieved = executor.get_tool("test_tool")
        assert retrieved == tool_def

    def test_get_tool_nonexistent(self):
        """Test getting a nonexistent tool returns None."""
        executor = ToolExecutor()

        retrieved = executor.get_tool("nonexistent_tool")
        assert retrieved is None

    def test_list_tools_empty(self):
        """Test listing tools when none are registered."""
        executor = ToolExecutor()

        tools = executor.list_tools()
        assert tools == []

    def test_list_tools_multiple(self):
        """Test listing multiple registered tools."""
        executor = ToolExecutor()

        def handler(params):
            return {"result": "success"}

        for i in range(3):
            tool_def = ToolDefinition(
                name=f"tool_{i}",
                description=f"Tool {i}",
                parameters_schema={"type": "object"},
                handler=handler,
            )
            executor.register_tool(tool_def)

        tools = executor.list_tools()
        assert len(tools) == 3
        assert "tool_0" in tools
        assert "tool_1" in tools
        assert "tool_2" in tools

    def test_unregister_tool_success(self):
        """Test successful tool unregistration."""
        executor = ToolExecutor()

        def handler(params):
            return {"result": "success"}

        tool_def = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters_schema={"type": "object"},
            handler=handler,
        )

        executor.register_tool(tool_def)
        assert "test_tool" in executor.list_tools()

        result = executor.unregister_tool("test_tool")
        assert result is True
        assert "test_tool" not in executor.list_tools()

    def test_unregister_tool_nonexistent(self):
        """Test unregistering nonexistent tool returns False."""
        executor = ToolExecutor()

        result = executor.unregister_tool("nonexistent_tool")
        assert result is False

    def test_tool_definition_with_metadata(self):
        """Test tool definition with metadata."""
        executor = ToolExecutor()

        def handler(params):
            return {"result": "success"}

        metadata = {"version": "1.0", "author": "test"}
        tool_def = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters_schema={"type": "object"},
            handler=handler,
            metadata=metadata,
        )

        executor.register_tool(tool_def)

        retrieved = executor.get_tool("test_tool")
        assert retrieved.metadata == metadata

    def test_complex_json_schema_validation(self):
        """Test parameter validation with complex JSON schema."""
        executor = ToolExecutor()

        def handler(params):
            return {"result": "success"}

        # Complex schema with nested objects and arrays
        tool_def = ToolDefinition(
            name="complex_tool",
            description="A tool with complex schema",
            parameters_schema={
                "type": "object",
                "properties": {
                    "user": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer", "minimum": 0, "maximum": 150},
                        },
                        "required": ["name", "age"],
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                },
                "required": ["user", "tags"],
            },
            handler=handler,
        )

        executor.register_tool(tool_def)

        # Valid complex parameters
        valid_params = {
            "user": {"name": "Alice", "age": 30},
            "tags": ["tag1", "tag2"],
        }
        is_valid, error = executor.validate_parameters("complex_tool", valid_params)
        assert is_valid
        assert error is None

        # Invalid: missing nested required field
        invalid_params = {
            "user": {"name": "Alice"},  # missing 'age'
            "tags": ["tag1"],
        }
        is_valid, error = executor.validate_parameters("complex_tool", invalid_params)
        assert not is_valid
        assert error is not None

        # Invalid: empty array (violates minItems)
        invalid_params2 = {
            "user": {"name": "Alice", "age": 30},
            "tags": [],  # empty array
        }
        is_valid, error = executor.validate_parameters("complex_tool", invalid_params2)
        assert not is_valid
        assert error is not None

    def test_tool_result_timestamp(self):
        """Test that ToolResult includes timestamp."""
        executor = ToolExecutor()

        def handler(params):
            return {"result": "success"}

        tool_def = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters_schema={
                "type": "object",
                "properties": {"param": {"type": "string"}},
                "required": ["param"],
            },
            handler=handler,
        )

        executor.register_tool(tool_def)

        result = executor.execute_tool("test_tool", {"param": "value"})

        assert result.timestamp is not None
        assert isinstance(result.timestamp, str)
        # Verify it's in ISO format (basic check)
        assert "T" in result.timestamp
