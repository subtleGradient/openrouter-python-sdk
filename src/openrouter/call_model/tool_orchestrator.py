"""Tool orchestrator for multi-round tool execution.

This module implements the tool orchestration logic that manages multi-round
conversations with tool execution. It coordinates between API calls, tool
execution, and building updated conversation context.

Key features:
- Multi-round tool orchestration (FR-1.3.6, FR-1.3.7)
- Configurable max rounds (int or callable) with 10 hard limit
- Message history building for context (FR-1.3.8)
- Graceful error handling during orchestration (FR-1.3.5)
- Integration with tool executor for actual execution

Example:
    >>> from openrouter.call_model.tool_orchestrator import execute_tool_loop
    >>>
    >>> result = await execute_tool_loop(
    ...     send_request=lambda input, tools: api.send(input),
    ...     initial_input={"model": "gpt-4", "messages": [...]},
    ...     tools=[weather_tool, calculator_tool],
    ...     max_rounds=3
    ... )
    >>> print(result.final_response)
    >>> print(f"Executed {len(result.all_responses)} rounds")
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, cast

from openrouter.call_model.exceptions import MaxToolRoundsExceededError
from openrouter.call_model.tool_executor import execute_tool, find_tool_by_name
from openrouter.call_model.tool_system import (
    BaseTool,
    ManualTool,
    ParsedToolCall,
    ToolExecutionResult,
)
from openrouter.call_model.types import ToolContext

# Default maximum rounds for tool execution (FR-1.3.6)
DEFAULT_MAX_ROUNDS = 5

# Hard maximum rounds enforced by the system (FR-1.3.7)
HARD_MAX_ROUNDS = 10


class ToolOrchestrationResult:
    """Result of the tool execution loop.

    This class contains all the data generated during multi-round tool
    orchestration, including all API responses, tool execution results,
    and the final conversation state.

    Attributes:
        final_response: The last API response received
        all_responses: List of all API responses from all rounds
        tool_execution_results: List of all tool execution results
        conversation_input: Final conversation input state
        rounds_executed: Number of tool execution rounds performed

    Example:
        >>> result = await execute_tool_loop(...)
        >>> print(f"Completed in {result.rounds_executed} rounds")
        >>> for response in result.all_responses:
        ...     print(f"Response: {response['id']}")
    """

    def __init__(
        self,
        final_response: dict[str, object],
        all_responses: list[dict[str, object]],
        tool_execution_results: list[ToolExecutionResult],
        conversation_input: dict[str, object],
        rounds_executed: int,
    ):
        """Initialize the orchestration result.

        Args:
            final_response: The last API response received
            all_responses: List of all API responses from all rounds
            tool_execution_results: List of all tool execution results
            conversation_input: Final conversation input state
            rounds_executed: Number of tool execution rounds performed
        """
        self.final_response = final_response
        self.all_responses = all_responses
        self.tool_execution_results = tool_execution_results
        self.conversation_input = conversation_input
        self.rounds_executed = rounds_executed


def extract_tool_calls_from_response(
    response: dict[str, object],
) -> list[ParsedToolCall]:
    """Extract tool calls from an API response.

    Args:
        response: API response dictionary

    Returns:
        list[ParsedToolCall]: List of parsed tool calls, empty if none found

    Example:
        >>> response = {"message": {"tool_calls": [...]}}
        >>> tool_calls = extract_tool_calls_from_response(response)
        >>> print(f"Found {len(tool_calls)} tool calls")
    """
    tool_calls: list[ParsedToolCall] = []

    # Extract message from response
    message = response.get("message")
    if not isinstance(message, dict):
        return tool_calls

    # Extract tool_calls array from message
    raw_tool_calls = message.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return tool_calls

    # Parse each tool call
    for raw_call in raw_tool_calls:
        if not isinstance(raw_call, dict):
            continue

        # Extract required fields
        call_id = raw_call.get("id")
        function = raw_call.get("function")

        if not isinstance(call_id, str) or not isinstance(function, dict):
            continue

        name = function.get("name")
        arguments = function.get("arguments")

        if not isinstance(name, str):
            continue

        # Handle arguments that may be either dict or JSON string
        if isinstance(arguments, str):
            # API sends arguments as JSON string, parse it
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, skip this tool call
                continue
        elif not isinstance(arguments, dict):
            # If arguments is neither string nor dict, skip
            continue

        # Create ParsedToolCall
        tool_calls.append(ParsedToolCall(id=call_id, name=name, arguments=arguments))

    return tool_calls


def response_has_tool_calls(response: dict[str, object]) -> bool:
    """Check if a response contains tool calls.

    Args:
        response: API response dictionary

    Returns:
        bool: True if response has tool calls, False otherwise

    Example:
        >>> if response_has_tool_calls(response):
        ...     print("Response requires tool execution")
    """
    tool_calls = extract_tool_calls_from_response(response)
    return len(tool_calls) > 0


def has_executable_tools(
    tool_calls: list[ParsedToolCall], tools: list[BaseTool[Any, Any]]
) -> bool:
    """Check if any tool calls can proceed (not ALL are manual-only).

    This returns False only if ALL tool calls are manual tools. Unknown tools
    or regular tools will return True (we'll generate errors or execute them).

    Args:
        tool_calls: List of tool calls from the API
        tools: List of available tool implementations

    Returns:
        bool: True if at least one tool call is not a manual tool

    Example:
        >>> if has_executable_tools(tool_calls, available_tools):
        ...     # Can proceed with automatic execution/error generation
        ...     await execute_tools(...)
    """
    for tool_call in tool_calls:
        tool = find_tool_by_name(tools, tool_call.name)
        # If tool not found OR tool is not manual, we can proceed
        if not tool or not isinstance(tool, ManualTool):
            return True
    return False


def build_tool_context(
    round_number: int,
    conversation_input: dict[str, object],
    previous_tool_results: list[ToolExecutionResult],
    response: dict[str, object],
) -> ToolContext:
    """Build a ToolContext for tool execution.

    Constructs the context object that gets passed to tool execute methods,
    including turn number, message history, model info, and previous results.

    Args:
        round_number: Current round number (1-indexed)
        conversation_input: Current conversation input state
        previous_tool_results: Results from previous tool executions
        response: Current API response

    Returns:
        ToolContext: Context object for tool execution

    Example:
        >>> context = build_tool_context(
        ...     round_number=2,
        ...     conversation_input=input_dict,
        ...     previous_tool_results=[...],
        ...     response=api_response
        ... )
        >>> print(f"Turn {context['number_of_turns']}")
    """
    # Extract message history from conversation input
    messages = conversation_input.get("messages")
    message_history: list[dict[str, object]] = []
    if isinstance(messages, list):
        message_history = [msg for msg in messages if isinstance(msg, dict)]

    # Extract model information
    model = conversation_input.get("model")
    model_str: str | None = str(model) if model is not None else None

    models = conversation_input.get("models")
    models_list: list[str] | None = None
    if isinstance(models, list):
        models_list = [str(m) for m in models if m is not None]

    # Convert tool results to dict format
    previous_results: list[dict[str, object]] | None = None
    if previous_tool_results:
        previous_results = [
            {
                "tool_call_id": result.tool_call_id,
                "tool_name": result.tool_name,
                "result": result.result,
                "error": result.error,
            }
            for result in previous_tool_results
        ]

    # Extract request ID from response
    request_id = response.get("id")
    request_id_str: str | None = str(request_id) if request_id is not None else None

    return ToolContext(
        number_of_turns=round_number,
        message_history=message_history,
        model=model_str,
        models=models_list,
        previous_tool_results=previous_results,
        request_id=request_id_str,
    )


async def execute_tool_loop(
    send_request: Callable[
        [dict[str, object], list[dict[str, object]]],
        Awaitable[dict[str, object]],
    ],
    initial_input: dict[str, object],
    tools: list[BaseTool[Any, Any]],
    api_tools: list[dict[str, object]],
    *,
    max_rounds: int | Callable[[ToolContext], bool] | None = None,
    on_preliminary_result: Callable[[str, Any], None] | None = None,
) -> ToolOrchestrationResult:
    """Execute multi-round tool orchestration loop.

    This is the main orchestration function that manages the loop of:
    1. Send request to API
    2. Check for tool calls in response
    3. Execute tools
    4. Build updated input with results
    5. Repeat until no more tool calls or max rounds reached

    Args:
        send_request: Function to send a request and get a response
        initial_input: Starting input for the conversation
        tools: Enhanced tools with execute methods
        api_tools: Converted tools in API format (JSON Schema)
        max_rounds: Max rounds (int, default: 5, max: 10) or callable
        on_preliminary_result: Optional callback for preliminary results

    Returns:
        ToolOrchestrationResult: Result containing final response and all data

    Raises:
        MaxToolRoundsExceededError: If max rounds exceeded (FR-1.3.7)

    Example:
        >>> result = await execute_tool_loop(
        ...     send_request=lambda input, tools: api.send(input),
        ...     initial_input={"model": "gpt-4", "messages": [...]},
        ...     tools=[weather_tool],
        ...     api_tools=[{"type": "function", ...}],
        ...     max_rounds=3
        ... )
        >>> print(f"Completed in {result.rounds_executed} rounds")
    """
    # Determine max rounds (FR-1.3.6)
    if max_rounds is None:
        effective_max_rounds = DEFAULT_MAX_ROUNDS
        max_rounds_callable: Callable[[ToolContext], bool] | None = None
    elif callable(max_rounds):
        # Callable - will be evaluated each round
        effective_max_rounds = HARD_MAX_ROUNDS
        max_rounds_callable = max_rounds
    else:
        # Integer - enforce hard maximum (FR-1.3.7)
        effective_max_rounds = min(int(max_rounds), HARD_MAX_ROUNDS)
        max_rounds_callable = None

    # Initialize tracking state
    all_responses: list[dict[str, object]] = []
    tool_execution_results: list[ToolExecutionResult] = []
    conversation_input = initial_input.copy()
    rounds_executed = 0

    # Initial API request
    response_raw = await send_request(conversation_input, api_tools)
    current_response = cast(dict[str, object], response_raw)
    all_responses.append(current_response)

    # Multi-round orchestration loop
    while response_has_tool_calls(current_response):
        # Extract tool calls from response
        tool_calls = extract_tool_calls_from_response(current_response)

        if not tool_calls:
            break

        # Check if any tools have executable implementations (FR-1.3.10)
        if not has_executable_tools(tool_calls, tools):
            # No executable tools - return for manual handling
            break

        # Increment round counter
        rounds_executed += 1

        # Build context for tool execution (FR-1.3.8)
        context = build_tool_context(
            round_number=rounds_executed,
            conversation_input=conversation_input,
            previous_tool_results=tool_execution_results,
            response=current_response,
        )

        # Check max rounds before executing (FR-1.3.7)
        if max_rounds_callable:
            # If callable returns False, stop
            # Note: We don't decrement for callable because the context
            # represents the round we just started, not the next one
            if not max_rounds_callable(context):
                raise MaxToolRoundsExceededError(
                    rounds=rounds_executed,
                    max_rounds=effective_max_rounds,
                )
        elif rounds_executed > effective_max_rounds:
            # Decrement since we incremented but aren't actually executing this round
            rounds_executed -= 1
            raise MaxToolRoundsExceededError(
                rounds=rounds_executed,
                max_rounds=effective_max_rounds,
            )

        # Execute all tool calls for this round
        round_results: list[ToolExecutionResult] = []

        for tool_call in tool_calls:
            tool = find_tool_by_name(tools, tool_call.name)

            if not tool:
                # Tool not found in definitions
                round_results.append(
                    ToolExecutionResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        result=None,
                        error=f'Tool "{tool_call.name}" not found in tool definitions',
                    )
                )
                continue

            if isinstance(tool, ManualTool):
                # Manual tool - skip automatic execution (FR-1.3.10)
                continue

            # Execute the tool with error handling (FR-1.3.5)
            result = await execute_tool(
                tool=tool,
                tool_call=tool_call,
                context=context,
                on_preliminary_result=on_preliminary_result,
            )
            round_results.append(result)

        # Add round results to overall results
        tool_execution_results.extend(round_results)

        # Update conversation with tool results for next round
        if round_results:
            # Build tool_calls list for assistant message
            tool_calls_for_message = []
            for result in round_results:
                # Find the original tool call to get arguments
                original_call = next(
                    (tc for tc in tool_calls if tc.id == result.tool_call_id), None
                )
                tool_calls_for_message.append(
                    {
                        "id": result.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": result.tool_name,
                            "arguments": json.dumps(original_call.arguments)
                            if original_call
                            else "{}",
                        },
                    }
                )

            # Add the assistant's message with tool calls
            assistant_message = {
                "role": "assistant",
                "content": current_response.get("content"),
                "tool_calls": tool_calls_for_message,
            }

            # Add tool results as tool messages
            tool_messages = [
                {
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": json.dumps(result.result)
                    if result.result is not None
                    else result.error or "",
                }
                for result in round_results
            ]

            # Update messages in conversation_input
            messages = conversation_input.get("messages", [])
            if isinstance(messages, list):
                # Create a new list to avoid modifying the original
                updated_messages = list(messages)
                updated_messages.append(assistant_message)
                updated_messages.extend(tool_messages)

                # Create updated input with new messages
                conversation_input = {
                    **conversation_input,
                    "messages": updated_messages,
                }

        # Send updated conversation to API for next round
        response_raw = await send_request(conversation_input, api_tools)
        current_response = cast(dict[str, object], response_raw)
        all_responses.append(current_response)

    # Return complete orchestration result
    return ToolOrchestrationResult(
        final_response=current_response,
        all_responses=all_responses,
        tool_execution_results=tool_execution_results,
        conversation_input=conversation_input,
        rounds_executed=rounds_executed,
    )


def summarize_tool_executions(results: list[ToolExecutionResult]) -> str:
    """Build a summary of tool executions for debugging/logging.

    Args:
        results: List of tool execution results to summarize

    Returns:
        str: Multi-line summary of all executions

    Example:
        >>> summary = summarize_tool_executions(results)
        >>> print(summary)
        ✅ weather_tool (call_123): SUCCESS
        ❌ calculator_tool (call_456): ERROR - Division by zero
    """
    lines: list[str] = []

    for result in results:
        if result.error:
            lines.append(
                f"❌ {result.tool_name} ({result.tool_call_id}): ERROR - {result.error}"
            )
        else:
            prelim_count = len(result.preliminary_results or [])
            prelim_info = (
                f" ({prelim_count} preliminary results)" if prelim_count > 0 else ""
            )
            lines.append(
                f"✅ {result.tool_name} ({result.tool_call_id}): SUCCESS{prelim_info}"
            )

    return "\n".join(lines)


def has_tool_execution_errors(results: list[ToolExecutionResult]) -> bool:
    """Check if any tool executions had errors.

    Args:
        results: List of tool execution results to check

    Returns:
        bool: True if any results have errors, False otherwise

    Example:
        >>> if has_tool_execution_errors(results):
        ...     print("Some tools failed!")
    """
    return any(result.error is not None for result in results)


def get_tool_execution_errors(results: list[ToolExecutionResult]) -> list[str]:
    """Get all tool execution error messages.

    Args:
        results: List of tool execution results

    Returns:
        list[str]: List of error messages from failed tools

    Example:
        >>> errors = get_tool_execution_errors(results)
        >>> for error in errors:
        ...     print(f"Error: {error}")
    """
    return [result.error for result in results if result.error is not None]
