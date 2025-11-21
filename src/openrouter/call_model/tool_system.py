"""Tool system definitions for the call_model API.

This module implements the tool system with Pydantic-based validation,
supporting three tool types: RegularTool, GeneratorTool, and ManualTool.

Key features:
- Type-safe tool definitions using Pydantic models
- Automatic JSON schema generation from Pydantic models
- Decorator pattern for convenient tool registration
- Support for sync and async execute methods
- Generator tools with preliminary results
- Manual tools for user-controlled execution

Example:
    >>> from pydantic import BaseModel
    >>> from openrouter.call_model.tool_system import tool, RegularTool
    >>>
    >>> # Using decorator pattern
    >>> @tool
    >>> class WeatherTool(BaseModel):
    ...     '''Get current weather for a location.'''
    ...     location: str
    ...     unit: str = "celsius"
    ...
    ...     def execute(self, context):
    ...         return {"temp": 22, "unit": self.unit}
    >>>
    >>> # Using class-based pattern
    >>> class SearchParams(BaseModel):
    ...     query: str
    ...     max_results: int = 10
    >>>
    >>> class SearchTool(RegularTool[SearchParams, dict]):
    ...     name: str = "search"
    ...     description: str = "Search for information"
    ...
    ...     async def execute(self, params: SearchParams, context):
    ...         # Perform search
    ...         return {"results": [...]}
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    TypeVar,
    get_args,
    get_origin,
)

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from openrouter.call_model.types import ToolContext

# Type variables for generic tool definitions
T = TypeVar("T", bound=BaseModel)
"""Type variable for tool parameter models."""

R = TypeVar("R")
"""Type variable for tool result types."""


class ParsedToolCall(BaseModel):
    """Parsed tool call from API response.

    Attributes:
        id: Unique identifier for this tool call
        name: Name of the tool being called
        arguments: Parsed arguments dictionary (from JSON)

    Example:
        >>> call = ParsedToolCall(
        ...     id="call_123",
        ...     name="weather_tool",
        ...     arguments={"location": "San Francisco", "unit": "celsius"}
        ... )
    """

    id: str
    name: str
    arguments: dict[str, Any]


class ToolExecutionResult(BaseModel):
    """Result of tool execution.

    Attributes:
        tool_call_id: ID of the tool call that was executed
        tool_name: Name of the tool that was executed
        result: Final result sent to model
        preliminary_results: All yielded values from generator (excluding final)
        error: Error message if execution failed

    Example:
        >>> result = ToolExecutionResult(
        ...     tool_call_id="call_123",
        ...     tool_name="search",
        ...     result={"results": [...]},
        ...     preliminary_results=[{"status": "searching"}]
        ... )
    """

    tool_call_id: str
    tool_name: str
    result: Any = None
    preliminary_results: list[Any] | None = None
    error: str | None = None


class BaseTool(BaseModel, Generic[T, R]):
    """Base class for all tool definitions.

    This is the foundation for all tool types, providing common functionality
    like JSON schema generation from Pydantic parameter models.

    Type Parameters:
        T: Pydantic model defining the tool's input parameters
        R: Type of the tool's return value

    Attributes:
        name: Unique name for the tool
        description: Human-readable description of what the tool does

    Example:
        >>> class MyParams(BaseModel):
        ...     value: str
        >>>
        >>> class MyTool(BaseTool[MyParams, str]):
        ...     name: str = "my_tool"
        ...     description: str = "Does something"
    """

    name: str
    description: str = ""

    # Class variable to mark this as allowing arbitrary types
    # This is needed for storing the parameter model type
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def to_json_schema(self) -> dict[str, Any]:
        """Generate JSON schema for tool parameters.

        Extracts the parameter model type from the execute method signature
        and generates its JSON schema for use in API requests.

        Returns:
            dict[str, Any]: JSON schema for the tool's parameters

        Example:
            >>> schema = tool.to_json_schema()
            >>> assert "properties" in schema
            >>> assert "location" in schema["properties"]
        """
        # Get the parameter type from the execute method signature
        # This is more reliable than trying to extract from __orig_bases__
        # which gets complicated with Pydantic's metaclass
        if hasattr(self, "execute"):
            execute_method = self.execute
            # Get the signature
            sig = inspect.signature(execute_method)
            # For bound methods, 'self' is already excluded from signature
            # First parameter is 'params', second is 'context'
            params_list = list(sig.parameters.values())
            if len(params_list) >= 1:
                params_param = params_list[0]  # First param is 'params'
                if params_param.annotation != inspect.Parameter.empty:
                    param_type = params_param.annotation
                    # Check if it's a BaseModel subclass
                    if isinstance(param_type, type) and issubclass(
                        param_type, BaseModel
                    ):
                        return param_type.model_json_schema()

        # Fallback: return empty schema
        return {}


class RegularTool(BaseTool[T, R]):
    """Tool with synchronous or async execute method.

    Regular tools perform a single operation and return a result.
    The execute method can be either sync or async.

    Type Parameters:
        T: Pydantic model defining the tool's input parameters
        R: Type of the tool's return value

    Example:
        >>> class CalculatorParams(BaseModel):
        ...     a: float
        ...     b: float
        >>>
        >>> class AddTool(RegularTool[CalculatorParams, float]):
        ...     name: str = "add"
        ...     description: str = "Add two numbers"
        ...
        ...     def execute(self, params: CalculatorParams, context):
        ...         return params.a + params.b
    """

    def execute(self, params: T, context: ToolContext) -> R | Awaitable[R]:
        """Execute the tool with given parameters.

        This method must be overridden by subclasses. It can be either
        synchronous or asynchronous.

        Args:
            params: Validated parameter object
            context: Execution context with turn info, history, etc.

        Returns:
            Union[R, Awaitable[R]]: Tool result (sync or async)

        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError(f"Tool '{self.name}' must implement execute method")


class GeneratorTool(BaseTool[T, R]):
    """Tool with async generator execute method.

    Generator tools can yield preliminary results during execution,
    with the final yield being the actual result sent to the model.

    Type Parameters:
        T: Pydantic model defining the tool's input parameters
        R: Type of the tool's final return value

    Example:
        >>> class SearchParams(BaseModel):
        ...     query: str
        >>>
        >>> class SearchTool(GeneratorTool[SearchParams, dict]):
        ...     name: str = "search"
        ...     description: str = "Search with progress updates"
        ...
        ...     async def execute(self, params: SearchParams, context):
        ...         yield {"status": "searching"}
        ...         await asyncio.sleep(0.1)
        ...         yield {"status": "found 5 results"}
        ...         yield {"results": [...]}  # Final result
    """

    async def execute(
        self, params: T, context: ToolContext
    ) -> AsyncIterator[object | R]:
        """Execute the tool as an async generator.

        Yields preliminary events during execution, with the final yield
        being the actual result.

        Args:
            params: Validated parameter object
            context: Execution context with turn info, history, etc.

        Yields:
            Union[Any, R]: Preliminary events, then final result as last yield

        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError(f"Tool '{self.name}' must implement execute method")
        # Make this an async generator for type checking
        yield  # type: ignore[unreachable]


class ManualTool(BaseTool[T, R]):
    """Tool without execute method - requires manual handling.

    Manual tools don't have automatic execution. When the model calls them,
    the developer is responsible for handling the call and providing results.

    Type Parameters:
        T: Pydantic model defining the tool's input parameters
        R: Type of the expected return value

    Example:
        >>> class ConfirmParams(BaseModel):
        ...     action: str
        ...     details: str
        >>>
        >>> confirm_tool = ManualTool[ConfirmParams, bool](
        ...     name="confirm_action",
        ...     description="Ask user to confirm an action"
        ... )
        >>>
        >>> # Later, in user code:
        >>> tool_calls = await response.get_tool_calls()
        >>> for call in tool_calls:
        ...     if call.name == "confirm_action":
        ...         # Handle manually
        ...         user_confirmed = input(f"Confirm {call.arguments['action']}? ")
    """

    pass


def tool(cls: type[T]) -> type[RegularTool[T, dict[str, Any]]]:
    """Decorator to convert a Pydantic model to a tool.

    This decorator provides a convenient way to create tools from Pydantic
    models. The model's fields become the tool's parameters, and if the model
    has an execute method, it's used for execution.

    Args:
        cls: Pydantic BaseModel class to convert to a tool

    Returns:
        type[RegularTool]: Tool class with execute method

    Example:
        >>> @tool
        ... class WeatherTool(BaseModel):
        ...     '''Get weather for a location.'''
        ...     location: str
        ...     unit: str = "celsius"
        ...
        ...     def execute(self, context):
        ...         # Fetch weather data
        ...         return {"temp": 22, "unit": self.unit}
        >>>
        >>> weather = WeatherTool()
        >>> assert weather.name == "weathertool"
        >>> schema = weather.to_json_schema()
        >>> assert "location" in schema["properties"]
    """
    # Generate tool name from class name (lowercase)
    tool_name = cls.__name__.lower()
    # Use docstring as description
    tool_description = cls.__doc__ or ""

    class ToolImpl(RegularTool[T, dict[str, Any]]):  # type: ignore[type-var]  # pyright: ignore[reportGeneralTypeIssues]
        """Generated tool implementation."""

        name: str = Field(default=tool_name)
        description: str = Field(default=tool_description.strip())

        # Store reference to the parameter model
        _model_class: ClassVar[type[BaseModel]] = cls

        def execute(
            self,
            params: Any,  # Use Any since cls is runtime type
            context: ToolContext,
        ) -> dict[str, object] | Awaitable[dict[str, object]]:
            """Execute the tool using the model's execute method.

            Args:
                params: Validated parameter instance
                context: Execution context

            Returns:
                Union[dict[str, Any], Awaitable[dict[str, Any]]]: Result

            Raises:
                NotImplementedError: If the model doesn't have execute method
            """
            # Check if the parameter model has an execute method
            if hasattr(params, "execute"):
                execute_method = getattr(params, "execute")
                result = execute_method(context)

                # Handle both sync and async execute methods
                if inspect.iscoroutine(result):
                    return result  # type: ignore[return-value]
                return result  # type: ignore[return-value]

            raise NotImplementedError(
                f"Tool {self.name} must implement execute method on the parameter model"
            )

        def to_json_schema(self) -> dict[str, Any]:
            """Generate JSON schema from the parameter model.

            Returns:
                dict[str, Any]: JSON schema for the tool's parameters
            """
            return self._model_class.model_json_schema()

    return ToolImpl  # type: ignore[return-value]
