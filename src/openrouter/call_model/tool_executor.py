"""Tool executor for the call_model API.

This module implements tool execution logic with validation, timeout enforcement,
and comprehensive error handling. It supports regular tools, generator tools with
preliminary results, and async execution patterns.

Key features:
- Pydantic validation of tool inputs before execution
- 30-second timeout enforcement per tool execution (NFR-2.2.4)
- Support for both sync and async execute methods
- Generator tool support with preliminary result handling
- Comprehensive error wrapping with actionable context

Example:
    >>> from openrouter.call_model.tool_executor import execute_tool
    >>> from openrouter.call_model.tool_system import RegularTool
    >>>
    >>> tool = MyTool(name="example", description="Example tool")
    >>> tool_call = ParsedToolCall(
    ...     id="call_123",
    ...     name="example",
    ...     arguments={"param": "value"}
    ... )
    >>> context = ToolContext(number_of_turns=1, message_history=[])
    >>> result = await execute_tool(tool, tool_call, context)
    >>> print(result.result)
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from openrouter.call_model.exceptions import (
    ToolExecutionError,
    ToolValidationError,
)
from openrouter.call_model.tool_system import (
    BaseTool,
    GeneratorTool,
    ManualTool,
    ParsedToolCall,
    RegularTool,
    ToolExecutionResult,
)
from openrouter.call_model.types import ToolContext

# Default timeout for tool execution (30 seconds per NFR-2.2.4)
DEFAULT_TOOL_TIMEOUT = 30.0


def _get_parameter_model_from_tool(tool: BaseTool[Any, Any]) -> type[BaseModel] | None:
    """Extract the parameter model type from a tool's generic base.

    Args:
        tool: Tool instance to inspect

    Returns:
        type[BaseModel] | None: Parameter model class or None if not found
    """
    # For tools created with @tool decorator
    if hasattr(tool, "_model_class"):
        return tool._model_class  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]

    # For Pydantic v2 models with generics, check __pydantic_generic_metadata__
    # This works for classes like: class MyTool(RegularTool[ParamsModel, ResultType])
    for cls in type(tool).__mro__:
        if hasattr(cls, "__pydantic_generic_metadata__"):
            metadata = cls.__pydantic_generic_metadata__
            if isinstance(metadata, dict) and "args" in metadata:
                args = metadata["args"]
                if args and len(args) >= 1:
                    param_model = args[0]
                    # Check if it's a BaseModel subclass
                    if isinstance(param_model, type) and issubclass(
                        param_model, BaseModel
                    ):
                        return param_model

    return None


def _validate_tool_input(
    tool: BaseTool[Any, Any],
    tool_call: ParsedToolCall,
    arguments: dict[str, Any],
) -> Any:
    """Validate tool arguments against Pydantic schema.

    Args:
        tool: Tool being executed
        tool_call: Parsed tool call with metadata
        arguments: Arguments to validate

    Returns:
        Any: Validated parameter instance

    Raises:
        ToolValidationError: If validation fails
    """
    param_model = _get_parameter_model_from_tool(tool)

    if param_model is None:
        # No parameter model found, return raw arguments
        return arguments

    try:
        # Validate using Pydantic
        validated = param_model.model_validate(arguments)
        return validated
    except ValidationError as e:
        # Convert Pydantic errors to readable format
        error_messages = []
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            error_messages.append(f"{field}: {msg}")

        raise ToolValidationError(
            tool_name=tool_call.name, validation_errors=error_messages
        ) from e


async def _execute_regular_tool(
    tool: RegularTool[Any, Any],
    validated_params: Any,
    context: ToolContext,
) -> Any:
    """Execute a regular (non-generator) tool.

    Handles both sync and async execute methods.

    Args:
        tool: Tool to execute
        validated_params: Validated parameter instance
        context: Execution context

    Returns:
        Any: Tool execution result
    """
    result = tool.execute(validated_params, context)

    # Handle both sync and async execute methods
    if inspect.iscoroutine(result):
        return await result

    return result


async def _execute_generator_tool(
    tool: GeneratorTool[Any, Any],
    validated_params: Any,
    context: ToolContext,
    on_preliminary_result: Callable[[str, Any], None] | None = None,
    tool_call_id: str = "",
) -> tuple[Any, list[Any]]:
    """Execute a generator tool and collect preliminary and final results.

    Args:
        tool: Generator tool to execute
        validated_params: Validated parameter instance
        context: Execution context
        on_preliminary_result: Optional callback for preliminary results
        tool_call_id: ID of the tool call (for callback)

    Returns:
        tuple[Any, list[Any]]: (final_result, preliminary_results)

    Raises:
        ValueError: If generator doesn't emit any values
    """
    preliminary_results: list[Any] = []
    last_value: Any = None
    has_emitted = False

    # Execute generator and collect all yields
    async for value in tool.execute(validated_params, context):
        has_emitted = True
        preliminary_results.append(value)
        last_value = value

        # Notify callback of preliminary result (all except final)
        if on_preliminary_result:
            on_preliminary_result(tool_call_id, value)

    # Generator must emit at least one value
    if not has_emitted:
        raise ValueError(
            f"Generator tool '{tool.name}' completed without emitting any values"
        )

    # Last emitted value is the final result
    final_result = last_value

    # Remove final result from preliminary list
    if preliminary_results:
        preliminary_results.pop()

    return final_result, preliminary_results


async def execute_tool(
    tool: BaseTool[Any, Any],
    tool_call: ParsedToolCall,
    context: ToolContext,
    *,
    timeout: float = DEFAULT_TOOL_TIMEOUT,
    on_preliminary_result: Callable[[str, Any], None] | None = None,
) -> ToolExecutionResult:
    """Execute a tool with validation, timeout, and error handling.

    This is the main entry point for tool execution. It handles:
    - Input validation against Pydantic schemas (FR-1.3.4)
    - Timeout enforcement (NFR-2.2.4)
    - Both sync and async execute methods
    - Generator tools with preliminary results (FR-1.3.9)
    - Comprehensive error wrapping

    Args:
        tool: Tool to execute
        tool_call: Parsed tool call with arguments
        context: Execution context with turn info, history, etc.
        timeout: Maximum execution time in seconds (default: 30.0)
        on_preliminary_result: Optional callback for generator preliminary results

    Returns:
        ToolExecutionResult: Result with success or error information

    Example:
        >>> result = await execute_tool(
        ...     tool=my_tool,
        ...     tool_call=ParsedToolCall(
        ...         id="call_123",
        ...         name="my_tool",
        ...         arguments={"param": "value"}
        ...     ),
        ...     context=ToolContext(number_of_turns=1, message_history=[])
        ... )
        >>> if result.error:
        ...     print(f"Tool failed: {result.error}")
        ... else:
        ...     print(f"Tool result: {result.result}")
    """
    # Manual tools cannot be executed automatically
    if isinstance(tool, ManualTool):
        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=None,
            error="Manual tool requires user intervention",
        )

    try:
        # Validate input against Pydantic schema (FR-1.3.4)
        validated_params = _validate_tool_input(tool, tool_call, tool_call.arguments)

        # Execute with timeout (NFR-2.2.4)
        async with asyncio.timeout(timeout):
            if isinstance(tool, GeneratorTool):
                # Handle generator tool with preliminary results (FR-1.3.9)
                final_result, preliminary_results = await _execute_generator_tool(
                    tool=tool,
                    validated_params=validated_params,
                    context=context,
                    on_preliminary_result=on_preliminary_result,
                    tool_call_id=tool_call.id,
                )

                return ToolExecutionResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=final_result,
                    preliminary_results=preliminary_results,
                )

            elif isinstance(tool, RegularTool):
                # Handle regular tool (sync or async)
                result = await _execute_regular_tool(
                    tool=tool, validated_params=validated_params, context=context
                )

                return ToolExecutionResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=result,
                )

            else:
                # Should not reach here
                return ToolExecutionResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=None,
                    error=f"Unknown tool type: {type(tool).__name__}",
                )

    except asyncio.TimeoutError:
        # Tool execution exceeded timeout (NFR-2.2.4)
        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=None,
            error=f"Tool execution timed out after {timeout} seconds",
        )

    except ToolValidationError as e:
        # Validation error - return with error message
        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=None,
            error=str(e),
        )

    except Exception as e:
        # Wrap all other errors in ToolExecutionError
        error_context = {
            "input": tool_call.arguments,
            "tool_call_id": tool_call.id,
            "context": context,
        }

        tool_error = ToolExecutionError(
            tool_name=tool_call.name,
            error=e,
            context=cast(dict[str, object], error_context),
        )

        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=None,
            error=str(tool_error),
        )


def find_tool_by_name(
    tools: list[BaseTool[Any, Any]], name: str
) -> BaseTool[Any, Any] | None:
    """Find a tool by name in the tools list.

    Args:
        tools: List of available tools
        name: Name of the tool to find

    Returns:
        BaseTool[Any, Any] | None: Tool if found, None otherwise

    Example:
        >>> tools = [tool1, tool2, tool3]
        >>> tool = find_tool_by_name(tools, "weather")
        >>> if tool:
        ...     print(f"Found tool: {tool.name}")
    """
    for tool in tools:
        if tool.name == name:
            return tool
    return None
