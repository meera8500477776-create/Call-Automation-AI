"""Tool Executor for executing external API calls based on LLM decisions.

This module provides the ToolExecutor class that manages tool registration,
parameter validation, and execution of external API calls.

Requirements: 10.4, 10.5
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol
from datetime import datetime, timezone
import logging
from jsonschema import validate, ValidationError as JsonSchemaValidationError


logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """
    Defines a tool that can be executed by the ToolExecutor.
    
    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description of what the tool does
        parameters_schema: JSON schema defining valid parameters
        handler: Callable that executes the tool logic
        metadata: Additional metadata about the tool
    """
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """
    Result of a tool execution.
    
    Attributes:
        success: Whether the tool executed successfully
        data: Result data from the tool execution
        error: Error message if execution failed
        execution_time_ms: Time taken to execute the tool in milliseconds
        timestamp: When the tool was executed (ISO format)
    """
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ToolExecutor:
    """
    Executes external API calls based on LLM decisions.
    
    The ToolExecutor manages a registry of available tools, validates parameters
    against JSON schemas, and executes tools with proper error handling.
    
    Requirements: 10.4, 10.5
    """
    
    def __init__(self):
        """Initialize the ToolExecutor with an empty tool registry."""
        self._tools: Dict[str, ToolDefinition] = {}
        logger.info("ToolExecutor initialized")
    
    def register_tool(self, tool_def: ToolDefinition) -> None:
        """
        Register a new tool for execution.
        
        Args:
            tool_def: Tool definition containing name, schema, and handler
            
        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool_def.name in self._tools:
            raise ValueError(f"Tool '{tool_def.name}' is already registered")
        
        self._tools[tool_def.name] = tool_def
        logger.info(f"Registered tool: {tool_def.name}")
    
    def validate_parameters(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate parameters against the tool's JSON schema.
        
        Args:
            tool_name: Name of the tool to validate parameters for
            parameters: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_message)
            
        Raises:
            ValueError: If the tool is not registered
        """
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' is not registered")
        
        tool_def = self._tools[tool_name]
        
        try:
            validate(instance=parameters, schema=tool_def.parameters_schema)
            return True, None
        except JsonSchemaValidationError as e:
            error_msg = f"Parameter validation failed: {e.message}"
            logger.warning(f"Validation failed for tool '{tool_name}': {error_msg}")
            return False, error_msg
    
    def execute_tool(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool
            context: Optional context information (e.g., session_id, user_id)
            
        Returns:
            ToolResult containing execution outcome
        """
        start_time = datetime.now(timezone.utc)
        
        # Check if tool exists
        if tool_name not in self._tools:
            error_msg = f"Tool '{tool_name}' is not registered"
            logger.error(error_msg)
            return ToolResult(
                success=False,
                error=error_msg,
                execution_time_ms=0.0
            )
        
        # Validate parameters
        is_valid, validation_error = self.validate_parameters(tool_name, parameters)
        if not is_valid:
            logger.error(f"Parameter validation failed for tool '{tool_name}': {validation_error}")
            return ToolResult(
                success=False,
                error=validation_error,
                execution_time_ms=0.0
            )
        
        # Execute the tool
        tool_def = self._tools[tool_name]
        try:
            logger.info(f"Executing tool '{tool_name}' with parameters: {parameters}")
            result_data = tool_def.handler(parameters)
            
            end_time = datetime.now(timezone.utc)
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            logger.info(f"Tool '{tool_name}' executed successfully in {execution_time_ms:.2f}ms")
            return ToolResult(
                success=True,
                data=result_data,
                execution_time_ms=execution_time_ms
            )
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            error_msg = self.handle_tool_error(e, tool_name)
            logger.error(f"Tool '{tool_name}' execution failed: {error_msg}")
            
            return ToolResult(
                success=False,
                error=error_msg,
                execution_time_ms=execution_time_ms
            )
    
    def handle_tool_error(self, error: Exception, tool_name: str) -> str:
        """
        Handle tool execution errors and return a user-friendly error message.
        
        Args:
            error: The exception that occurred
            tool_name: Name of the tool that failed
            
        Returns:
            User-friendly error message
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Log the full error for debugging
        logger.exception(f"Tool '{tool_name}' raised {error_type}: {error_msg}")
        
        # Return a user-friendly error message
        return f"Tool execution failed: {error_type} - {error_msg}"
    
    def get_tool(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        Get a tool definition by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            ToolDefinition if found, None otherwise
        """
        return self._tools.get(tool_name)
    
    def list_tools(self) -> list[str]:
        """
        List all registered tool names.
        
        Returns:
            List of registered tool names
        """
        return list(self._tools.keys())
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            True if tool was unregistered, False if tool was not found
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")
            return True
        return False
