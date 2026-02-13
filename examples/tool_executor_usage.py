"""Example usage of ToolExecutor for managing and executing tools.

This example demonstrates:
1. Creating a ToolExecutor
2. Registering tools with JSON schema validation
3. Executing tools with parameter validation
4. Handling tool execution errors
5. Managing tool registry

Requirements: 10.4, 10.5
"""

from voice_agent.core.tool_executor import ToolExecutor, ToolDefinition


def example_basic_tool_registration():
    """Example: Basic tool registration and execution."""
    print("=== Basic Tool Registration ===\n")
    
    # Create a ToolExecutor
    executor = ToolExecutor()
    
    # Define a simple greeting tool
    def greet_handler(params):
        name = params.get("name", "Guest")
        return {"message": f"Hello, {name}!"}
    
    # Create tool definition with JSON schema
    greet_tool = ToolDefinition(
        name="greet",
        description="Greet a person by name",
        parameters_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        },
        handler=greet_handler
    )
    
    # Register the tool
    executor.register_tool(greet_tool)
    print(f"Registered tool: {greet_tool.name}")
    
    # Execute the tool
    result = executor.execute_tool("greet", {"name": "Alice"})
    print(f"Success: {result.success}")
    print(f"Result: {result.data}")
    print(f"Execution time: {result.execution_time_ms:.2f}ms\n")


def example_parameter_validation():
    """Example: Parameter validation with JSON schema."""
    print("=== Parameter Validation ===\n")
    
    executor = ToolExecutor()
    
    # Define a tool with strict parameter validation
    def calculate_handler(params):
        a = params["a"]
        b = params["b"]
        operation = params["operation"]
        
        if operation == "add":
            return {"result": a + b}
        elif operation == "subtract":
            return {"result": a - b}
        elif operation == "multiply":
            return {"result": a * b}
        elif operation == "divide":
            return {"result": a / b if b != 0 else None}
        
        return {"result": None}
    
    calculator_tool = ToolDefinition(
        name="calculator",
        description="Perform basic arithmetic operations",
        parameters_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"]
                }
            },
            "required": ["a", "b", "operation"]
        },
        handler=calculate_handler
    )
    
    executor.register_tool(calculator_tool)
    
    # Valid execution
    print("Valid parameters:")
    result = executor.execute_tool("calculator", {
        "a": 10,
        "b": 5,
        "operation": "add"
    })
    print(f"  Success: {result.success}")
    print(f"  Result: {result.data}\n")
    
    # Invalid execution - missing parameter
    print("Invalid parameters (missing 'b'):")
    result = executor.execute_tool("calculator", {
        "a": 10,
        "operation": "add"
    })
    print(f"  Success: {result.success}")
    print(f"  Error: {result.error}\n")
    
    # Invalid execution - wrong type
    print("Invalid parameters (wrong type for 'a'):")
    result = executor.execute_tool("calculator", {
        "a": "ten",
        "b": 5,
        "operation": "add"
    })
    print(f"  Success: {result.success}")
    print(f"  Error: {result.error}\n")


def example_error_handling():
    """Example: Handling tool execution errors."""
    print("=== Error Handling ===\n")
    
    executor = ToolExecutor()
    
    # Define a tool that may raise exceptions
    def risky_handler(params):
        value = params["value"]
        if value < 0:
            raise ValueError("Value must be non-negative")
        return {"result": value ** 2}
    
    risky_tool = ToolDefinition(
        name="square",
        description="Calculate square of a number",
        parameters_schema={
            "type": "object",
            "properties": {
                "value": {"type": "number"}
            },
            "required": ["value"]
        },
        handler=risky_handler
    )
    
    executor.register_tool(risky_tool)
    
    # Successful execution
    print("Valid input:")
    result = executor.execute_tool("square", {"value": 5})
    print(f"  Success: {result.success}")
    print(f"  Result: {result.data}\n")
    
    # Execution with error
    print("Invalid input (negative value):")
    result = executor.execute_tool("square", {"value": -5})
    print(f"  Success: {result.success}")
    print(f"  Error: {result.error}")
    print(f"  Execution time: {result.execution_time_ms:.2f}ms\n")


def example_complex_schema():
    """Example: Complex JSON schema with nested objects."""
    print("=== Complex Schema ===\n")
    
    executor = ToolExecutor()
    
    # Define a tool with complex nested schema
    def create_user_handler(params):
        user = params["user"]
        return {
            "user_id": "12345",
            "name": user["name"],
            "email": user["email"],
            "age": user["age"]
        }
    
    user_tool = ToolDefinition(
        name="create_user",
        description="Create a new user",
        parameters_schema={
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "email": {"type": "string", "format": "email"},
                        "age": {"type": "integer", "minimum": 0, "maximum": 150}
                    },
                    "required": ["name", "email", "age"]
                }
            },
            "required": ["user"]
        },
        handler=create_user_handler
    )
    
    executor.register_tool(user_tool)
    
    # Execute with valid nested parameters
    result = executor.execute_tool("create_user", {
        "user": {
            "name": "Alice",
            "email": "alice@example.com",
            "age": 30
        }
    })
    print(f"Success: {result.success}")
    print(f"Result: {result.data}\n")


def example_tool_management():
    """Example: Managing tool registry."""
    print("=== Tool Management ===\n")
    
    executor = ToolExecutor()
    
    # Register multiple tools
    def handler1(params):
        return {"result": "tool1"}
    
    def handler2(params):
        return {"result": "tool2"}
    
    tool1 = ToolDefinition(
        name="tool1",
        description="First tool",
        parameters_schema={"type": "object"},
        handler=handler1
    )
    
    tool2 = ToolDefinition(
        name="tool2",
        description="Second tool",
        parameters_schema={"type": "object"},
        handler=handler2,
        metadata={"version": "1.0", "author": "example"}
    )
    
    executor.register_tool(tool1)
    executor.register_tool(tool2)
    
    # List all tools
    print(f"Registered tools: {executor.list_tools()}")
    
    # Get specific tool
    retrieved = executor.get_tool("tool2")
    print(f"Tool 2 metadata: {retrieved.metadata}")
    
    # Unregister a tool
    executor.unregister_tool("tool1")
    print(f"After unregistering tool1: {executor.list_tools()}\n")


def example_appointment_booking_tool():
    """Example: Real-world appointment booking tool."""
    print("=== Appointment Booking Tool ===\n")
    
    executor = ToolExecutor()
    
    # Simulate appointment booking
    def book_appointment_handler(params):
        patient_name = params["patient_name"]
        date = params["date"]
        time = params["time"]
        reason = params.get("reason", "General checkup")
        
        # In a real system, this would call an API
        return {
            "appointment_id": "APT-12345",
            "patient_name": patient_name,
            "date": date,
            "time": time,
            "reason": reason,
            "status": "confirmed"
        }
    
    appointment_tool = ToolDefinition(
        name="book_appointment",
        description="Book a medical appointment",
        parameters_schema={
            "type": "object",
            "properties": {
                "patient_name": {"type": "string", "minLength": 1},
                "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
                "time": {"type": "string", "pattern": r"^\d{2}:\d{2}$"},
                "reason": {"type": "string"}
            },
            "required": ["patient_name", "date", "time"]
        },
        handler=book_appointment_handler,
        metadata={
            "category": "healthcare",
            "requires_auth": True
        }
    )
    
    executor.register_tool(appointment_tool)
    
    # Book an appointment
    result = executor.execute_tool("book_appointment", {
        "patient_name": "John Doe",
        "date": "2024-03-15",
        "time": "14:30",
        "reason": "Annual checkup"
    })
    
    print(f"Success: {result.success}")
    print(f"Appointment details: {result.data}")
    print(f"Execution time: {result.execution_time_ms:.2f}ms\n")


if __name__ == "__main__":
    print("ToolExecutor Usage Examples\n")
    print("=" * 50 + "\n")
    
    example_basic_tool_registration()
    example_parameter_validation()
    example_error_handling()
    example_complex_schema()
    example_tool_management()
    example_appointment_booking_tool()
    
    print("=" * 50)
    print("\nAll examples completed successfully!")
