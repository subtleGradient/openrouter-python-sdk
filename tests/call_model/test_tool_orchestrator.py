"""Tests for tool orchestrator functionality.

This test module verifies:
- Single round tool orchestration
- Multi-round tool orchestration (2-3 rounds)
- Max rounds enforcement (int and callable)
- Tool errors during orchestration
- Message history building
- Context passing between rounds
- Tool call extraction from responses
- Orchestration result structure
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel

from openrouter.call_model.exceptions import MaxToolRoundsExceededError
from openrouter.call_model.tool_orchestrator import (
    DEFAULT_MAX_ROUNDS,
    HARD_MAX_ROUNDS,
    ToolOrchestrationResult,
    build_tool_context,
    execute_tool_loop,
    extract_tool_calls_from_response,
    get_tool_execution_errors,
    has_executable_tools,
    has_tool_execution_errors,
    response_has_tool_calls,
    summarize_tool_executions,
)
from openrouter.call_model.tool_system import (
    GeneratorTool,
    ManualTool,
    ParsedToolCall,
    RegularTool,
    ToolExecutionResult,
)
from openrouter.call_model.types import ToolContext


# Test parameter models
class CounterParams(BaseModel):
    """Counter tool parameters."""

    count: int = 0


class CalculatorParams(BaseModel):
    """Calculator tool parameters."""

    a: float
    b: float
    operation: str = "add"


# Test tools
class CounterTool(RegularTool[CounterParams, dict[str, Any]]):
    """Tool that increments a counter."""

    name: str = "counter"
    description: str = "Increment a counter"

    def execute(self, params: CounterParams, context: ToolContext) -> dict[str, Any]:
        """Execute the counter."""
        return {
            "count": params.count + 1,
            "turn": context.get("number_of_turns", 0),
        }


class CalculatorTool(RegularTool[CalculatorParams, dict[str, Any]]):
    """Tool that performs calculations."""

    name: str = "calculator"
    description: str = "Perform math operations"

    def execute(self, params: CalculatorParams, context: ToolContext) -> dict[str, Any]:
        """Execute the calculation."""
        operations = {
            "add": params.a + params.b,
            "subtract": params.a - params.b,
            "multiply": params.a * params.b,
            "divide": params.a / params.b if params.b != 0 else None,
        }
        return {
            "result": operations.get(params.operation),
            "operation": params.operation,
        }


class ErrorTool(RegularTool[CounterParams, dict[str, Any]]):
    """Tool that always errors."""

    name: str = "error_tool"
    description: str = "Always fails"

    def execute(self, params: CounterParams, context: ToolContext) -> dict[str, Any]:
        """Raise an error."""
        raise ValueError("Intentional error for testing")


class SlowTool(RegularTool[CounterParams, dict[str, Any]]):
    """Tool that takes a long time."""

    name: str = "slow_tool"
    description: str = "Takes too long"

    async def execute(
        self, params: CounterParams, context: ToolContext
    ) -> dict[str, Any]:
        """Sleep for a long time."""
        await asyncio.sleep(100)
        return {"status": "done"}


class GeneratorCounterTool(GeneratorTool[CounterParams, dict[str, Any]]):
    """Generator tool that counts with preliminary results."""

    name: str = "generator_counter"
    description: str = "Count with progress updates"

    async def execute(
        self, params: CounterParams, context: ToolContext
    ) -> AsyncIterator[object | dict[str, Any]]:
        """Execute with preliminary results."""
        for i in range(params.count):
            yield {"status": "counting", "current": i}
            await asyncio.sleep(0.001)
        turn = context.get("number_of_turns")
        yield {"final_count": params.count, "turn": turn if turn is not None else 0}


class ManualCounterTool(ManualTool[CounterParams, dict[str, Any]]):
    """Manual tool that requires user handling."""

    name: str = "manual_counter"
    description: str = "Manual counter"


# Helper functions for creating mock responses
def create_text_response(text: str, request_id: str = "resp_123") -> dict[str, object]:
    """Create a mock API response with text content."""
    return {
        "id": request_id,
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
    }


def create_tool_call_response(
    tool_calls: list[dict[str, Any]], request_id: str = "resp_123"
) -> dict[str, object]:
    """Create a mock API response with tool calls."""
    return {
        "id": request_id,
        "message": {
            "role": "assistant",
            "content": [],
            "tool_calls": tool_calls,
        },
    }


def create_tool_call_dict(
    call_id: str, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Create a tool call dictionary in API format."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


# Tests for utility functions
class TestToolCallExtraction:
    """Tests for extracting tool calls from responses."""

    def test_extract_tool_calls_from_valid_response(self) -> None:
        """Test extracting tool calls from valid response."""
        response = create_tool_call_response(
            [
                create_tool_call_dict("call_1", "counter", {"count": 1}),
                create_tool_call_dict(
                    "call_2", "calculator", {"a": 5.0, "b": 3.0, "operation": "add"}
                ),
            ]
        )

        tool_calls = extract_tool_calls_from_response(response)

        assert len(tool_calls) == 2
        assert tool_calls[0].id == "call_1"
        assert tool_calls[0].name == "counter"
        assert tool_calls[0].arguments == {"count": 1}
        assert tool_calls[1].id == "call_2"
        assert tool_calls[1].name == "calculator"

    def test_extract_tool_calls_from_empty_response(self) -> None:
        """Test extracting from response with no tool calls."""
        response = create_text_response("Hello, world!")

        tool_calls = extract_tool_calls_from_response(response)

        assert len(tool_calls) == 0

    def test_extract_tool_calls_from_invalid_response(self) -> None:
        """Test extracting from malformed response."""
        response: dict[str, object] = {"message": "invalid"}

        tool_calls = extract_tool_calls_from_response(response)

        assert len(tool_calls) == 0

    def test_response_has_tool_calls_true(self) -> None:
        """Test checking if response has tool calls (positive case)."""
        response = create_tool_call_response(
            [create_tool_call_dict("call_1", "counter", {"count": 1})]
        )

        assert response_has_tool_calls(response) is True

    def test_response_has_tool_calls_false(self) -> None:
        """Test checking if response has tool calls (negative case)."""
        response = create_text_response("No tools here")

        assert response_has_tool_calls(response) is False


class TestExecutableToolCheck:
    """Tests for checking if tool calls have executable implementations."""

    def test_has_executable_tools_with_regular_tool(self) -> None:
        """Test has_executable_tools with regular tool."""
        tool_calls = [ParsedToolCall(id="call_1", name="counter", arguments={})]
        tools: list[Any] = [CounterTool()]

        assert has_executable_tools(tool_calls, tools) is True

    def test_has_executable_tools_with_manual_tool(self) -> None:
        """Test has_executable_tools with manual tool only."""
        tool_calls = [ParsedToolCall(id="call_1", name="manual_counter", arguments={})]
        tools: list[Any] = [ManualCounterTool()]

        assert has_executable_tools(tool_calls, tools) is False

    def test_has_executable_tools_with_mixed_tools(self) -> None:
        """Test has_executable_tools with mix of manual and regular."""
        tool_calls = [
            ParsedToolCall(id="call_1", name="manual_counter", arguments={}),
            ParsedToolCall(id="call_2", name="counter", arguments={}),
        ]
        tools: list[Any] = [ManualCounterTool(), CounterTool()]

        assert has_executable_tools(tool_calls, tools) is True

    def test_has_executable_tools_with_unknown_tool(self) -> None:
        """Test has_executable_tools when tool not found.

        Unknown tools should return True so we can generate error results.
        """
        tool_calls = [ParsedToolCall(id="call_1", name="unknown", arguments={})]
        tools: list[Any] = [CounterTool()]

        assert has_executable_tools(tool_calls, tools) is True


class TestContextBuilding:
    """Tests for building tool context."""

    def test_build_tool_context_basic(self) -> None:
        """Test building basic tool context."""
        conversation_input = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        }
        response = create_text_response("Response")

        context = build_tool_context(
            round_number=1,
            conversation_input=conversation_input,
            previous_tool_results=[],
            response=response,
        )

        assert context["number_of_turns"] == 1
        assert context["model"] == "gpt-4"
        assert len(context["message_history"]) == 2
        assert context["request_id"] == "resp_123"

    def test_build_tool_context_with_previous_results(self) -> None:
        """Test building context with previous tool results."""
        conversation_input = {"model": "gpt-4", "messages": []}
        response = create_text_response("Response")
        previous_results = [
            ToolExecutionResult(
                tool_call_id="call_1",
                tool_name="counter",
                result={"count": 5},
            )
        ]

        context = build_tool_context(
            round_number=2,
            conversation_input=conversation_input,
            previous_tool_results=previous_results,
            response=response,
        )

        assert context["number_of_turns"] == 2
        assert context["previous_tool_results"] is not None
        assert len(context["previous_tool_results"]) == 1

    def test_build_tool_context_with_models_list(self) -> None:
        """Test building context with models list."""
        conversation_input = {
            "models": ["gpt-4", "claude-3"],
            "messages": [],
        }
        response = create_text_response("Response")

        context = build_tool_context(
            round_number=1,
            conversation_input=conversation_input,
            previous_tool_results=[],
            response=response,
        )

        assert context["models"] == ["gpt-4", "claude-3"]
        assert context["model"] is None


class TestOrchestrationResult:
    """Tests for ToolOrchestrationResult class."""

    def test_orchestration_result_creation(self) -> None:
        """Test creating orchestration result."""
        final_response = create_text_response("Final")
        all_responses = [
            create_tool_call_response([]),
            create_text_response("Final"),
        ]
        tool_results = [
            ToolExecutionResult(
                tool_call_id="call_1", tool_name="counter", result={"count": 1}
            )
        ]
        conversation_input = {"model": "gpt-4", "messages": []}

        result = ToolOrchestrationResult(
            final_response=final_response,
            all_responses=all_responses,
            tool_execution_results=tool_results,
            conversation_input=conversation_input,
            rounds_executed=1,
        )

        assert result.final_response == final_response
        assert len(result.all_responses) == 2
        assert len(result.tool_execution_results) == 1
        assert result.rounds_executed == 1


class TestToolOrchestration:
    """Tests for multi-round tool orchestration."""

    @pytest.mark.asyncio
    async def test_single_round_orchestration(self) -> None:
        """Test single round of tool execution."""
        tools: list[Any] = [CounterTool()]
        api_tools = [{"type": "function", "function": {"name": "counter"}}]

        responses = [
            create_tool_call_response(
                [create_tool_call_dict("call_1", "counter", {"count": 0})]
            ),
            create_text_response("Count is now 1"),
        ]
        response_index = 0

        async def mock_send_request(
            _input: dict[str, object], _tools: list[dict[str, object]]
        ) -> dict[str, object]:
            nonlocal response_index
            response = responses[response_index]
            response_index += 1
            return response

        result = await execute_tool_loop(
            send_request=mock_send_request,
            initial_input={"model": "gpt-4", "messages": []},
            tools=tools,
            api_tools=api_tools,
        )

        assert result.rounds_executed == 1
        assert len(result.all_responses) == 2
        assert len(result.tool_execution_results) == 1
        assert result.tool_execution_results[0].tool_name == "counter"
        assert result.tool_execution_results[0].result is not None

    @pytest.mark.asyncio
    async def test_multi_round_orchestration(self) -> None:
        """Test multi-round tool execution (2 rounds)."""
        tools: list[Any] = [CounterTool()]
        api_tools = [{"type": "function", "function": {"name": "counter"}}]

        responses = [
            create_tool_call_response(
                [create_tool_call_dict("call_1", "counter", {"count": 0})]
            ),
            create_tool_call_response(
                [create_tool_call_dict("call_2", "counter", {"count": 1})]
            ),
            create_text_response("Count is now 2"),
        ]
        response_index = 0

        async def mock_send_request(
            _input: dict[str, object], _tools: list[dict[str, object]]
        ) -> dict[str, object]:
            nonlocal response_index
            response = responses[response_index]
            response_index += 1
            return response

        result = await execute_tool_loop(
            send_request=mock_send_request,
            initial_input={"model": "gpt-4", "messages": []},
            tools=tools,
            api_tools=api_tools,
        )

        assert result.rounds_executed == 2
        assert len(result.all_responses) == 3
        assert len(result.tool_execution_results) == 2

    @pytest.mark.asyncio
    async def test_max_rounds_int_enforcement(self) -> None:
        """Test max rounds enforcement with integer."""
        tools: list[Any] = [CounterTool()]
        api_tools = [{"type": "function", "function": {"name": "counter"}}]

        # Always return tool calls to trigger max rounds
        async def mock_send_request(
            _input: dict[str, object], _tools: list[dict[str, object]]
        ) -> dict[str, object]:
            return create_tool_call_response(
                [create_tool_call_dict("call_1", "counter", {"count": 0})]
            )

        with pytest.raises(MaxToolRoundsExceededError) as exc_info:
            await execute_tool_loop(
                send_request=mock_send_request,
                initial_input={"model": "gpt-4", "messages": []},
                tools=tools,
                api_tools=api_tools,
                max_rounds=3,
            )

        assert exc_info.value.max_rounds == 3
        assert exc_info.value.rounds == 3

    @pytest.mark.asyncio
    async def test_max_rounds_callable_enforcement(self) -> None:
        """Test max rounds enforcement with callable."""
        tools: list[Any] = [CounterTool()]
        api_tools = [{"type": "function", "function": {"name": "counter"}}]

        # Callable that stops after 2 turns
        def should_continue(context: ToolContext) -> bool:
            return context["number_of_turns"] < 2

        # Always return tool calls to trigger max rounds
        async def mock_send_request(
            _input: dict[str, object], _tools: list[dict[str, object]]
        ) -> dict[str, object]:
            return create_tool_call_response(
                [create_tool_call_dict("call_1", "counter", {"count": 0})]
            )

        with pytest.raises(MaxToolRoundsExceededError) as exc_info:
            await execute_tool_loop(
                send_request=mock_send_request,
                initial_input={"model": "gpt-4", "messages": []},
                tools=tools,
                api_tools=api_tools,
                max_rounds=should_continue,
            )

        assert exc_info.value.rounds == 2

    @pytest.mark.asyncio
    async def test_default_max_rounds(self) -> None:
        """Test default max rounds is enforced."""
        tools: list[Any] = [CounterTool()]
        api_tools = [{"type": "function", "function": {"name": "counter"}}]

        # Always return tool calls to trigger max rounds
        async def mock_send_request(
            _input: dict[str, object], _tools: list[dict[str, object]]
        ) -> dict[str, object]:
            return create_tool_call_response(
                [create_tool_call_dict("call_1", "counter", {"count": 0})]
            )

        with pytest.raises(MaxToolRoundsExceededError) as exc_info:
            await execute_tool_loop(
                send_request=mock_send_request,
                initial_input={"model": "gpt-4", "messages": []},
                tools=tools,
                api_tools=api_tools,
                # Don't specify max_rounds - should use default
            )

        assert exc_info.value.max_rounds == DEFAULT_MAX_ROUNDS

    @pytest.mark.asyncio
    async def test_hard_max_rounds_enforcement(self) -> None:
        """Test hard maximum of 10 rounds is enforced."""
        tools: list[Any] = [CounterTool()]
        api_tools = [{"type": "function", "function": {"name": "counter"}}]

        # Always return tool calls to trigger max rounds
        async def mock_send_request(
            _input: dict[str, object], _tools: list[dict[str, object]]
        ) -> dict[str, object]:
            return create_tool_call_response(
                [create_tool_call_dict("call_1", "counter", {"count": 0})]
            )

        # Request 100 rounds, but should be capped at 10
        with pytest.raises(MaxToolRoundsExceededError) as exc_info:
            await execute_tool_loop(
                send_request=mock_send_request,
                initial_input={"model": "gpt-4", "messages": []},
                tools=tools,
                api_tools=api_tools,
                max_rounds=100,
            )

        assert exc_info.value.max_rounds == HARD_MAX_ROUNDS
        assert exc_info.value.rounds == HARD_MAX_ROUNDS

    @pytest.mark.asyncio
    async def test_no_tool_calls_stops_immediately(self) -> None:
        """Test that orchestration stops when no tool calls returned."""
        tools: list[Any] = [CounterTool()]
        api_tools = [{"type": "function", "function": {"name": "counter"}}]

        async def mock_send_request(
            _input: dict[str, object], _tools: list[dict[str, object]]
        ) -> dict[str, object]:
            return create_text_response("No tools needed")

        result = await execute_tool_loop(
            send_request=mock_send_request,
            initial_input={"model": "gpt-4", "messages": []},
            tools=tools,
            api_tools=api_tools,
        )

        assert result.rounds_executed == 0
        assert len(result.all_responses) == 1
        assert len(result.tool_execution_results) == 0

    @pytest.mark.asyncio
    async def test_manual_tools_stop_orchestration(self) -> None:
        """Test that manual tools stop automatic orchestration."""
        tools: list[Any] = [ManualCounterTool()]
        api_tools = [{"type": "function", "function": {"name": "manual_counter"}}]

        async def mock_send_request(
            _input: dict[str, object], _tools: list[dict[str, object]]
        ) -> dict[str, object]:
            return create_tool_call_response(
                [create_tool_call_dict("call_1", "manual_counter", {"count": 0})]
            )

        result = await execute_tool_loop(
            send_request=mock_send_request,
            initial_input={"model": "gpt-4", "messages": []},
            tools=tools,
            api_tools=api_tools,
        )

        # Should stop after initial response because manual tool can't be executed
        assert result.rounds_executed == 0
        assert len(result.all_responses) == 1

    @pytest.mark.asyncio
    async def test_tool_error_captured_in_results(self) -> None:
        """Test that tool errors are captured in results."""
        tools: list[Any] = [ErrorTool()]
        api_tools = [{"type": "function", "function": {"name": "error_tool"}}]

        responses = [
            create_tool_call_response(
                [create_tool_call_dict("call_1", "error_tool", {"count": 0})]
            ),
            create_text_response("Error handled"),
        ]
        response_index = 0

        async def mock_send_request(
            _input: dict[str, object], _tools: list[dict[str, object]]
        ) -> dict[str, object]:
            nonlocal response_index
            response = responses[response_index]
            response_index += 1
            return response

        result = await execute_tool_loop(
            send_request=mock_send_request,
            initial_input={"model": "gpt-4", "messages": []},
            tools=tools,
            api_tools=api_tools,
        )

        assert result.rounds_executed == 1
        assert len(result.tool_execution_results) == 1
        assert result.tool_execution_results[0].error is not None
        assert "Intentional error" in result.tool_execution_results[0].error

    @pytest.mark.asyncio
    async def test_unknown_tool_error(self) -> None:
        """Test error when tool not found in definitions."""
        tools: list[Any] = [CounterTool()]
        api_tools = [{"type": "function", "function": {"name": "unknown_tool"}}]

        responses = [
            create_tool_call_response(
                [create_tool_call_dict("call_1", "unknown_tool", {"count": 0})]
            ),
            create_text_response("Tool not found"),
        ]
        response_index = 0

        async def mock_send_request(
            _input: dict[str, object], _tools: list[dict[str, object]]
        ) -> dict[str, object]:
            nonlocal response_index
            response = responses[response_index]
            response_index += 1
            return response

        result = await execute_tool_loop(
            send_request=mock_send_request,
            initial_input={"model": "gpt-4", "messages": []},
            tools=tools,
            api_tools=api_tools,
        )

        assert result.rounds_executed == 1
        assert len(result.tool_execution_results) == 1
        assert result.tool_execution_results[0].error is not None
        assert "not found" in result.tool_execution_results[0].error

    @pytest.mark.asyncio
    async def test_generator_tool_preliminary_results(self) -> None:
        """Test generator tool with preliminary results."""
        tools: list[Any] = [GeneratorCounterTool()]
        api_tools = [{"type": "function", "function": {"name": "generator_counter"}}]

        preliminary_results: list[tuple[str, Any]] = []

        def on_preliminary(call_id: str, result: Any) -> None:
            preliminary_results.append((call_id, result))

        responses = [
            create_tool_call_response(
                [create_tool_call_dict("call_1", "generator_counter", {"count": 3})]
            ),
            create_text_response("Counted to 3"),
        ]
        response_index = 0

        async def mock_send_request(
            _input: dict[str, object], _tools: list[dict[str, object]]
        ) -> dict[str, object]:
            nonlocal response_index
            response = responses[response_index]
            response_index += 1
            return response

        result = await execute_tool_loop(
            send_request=mock_send_request,
            initial_input={"model": "gpt-4", "messages": []},
            tools=tools,
            api_tools=api_tools,
            on_preliminary_result=on_preliminary,
        )

        assert result.rounds_executed == 1
        assert len(result.tool_execution_results) == 1
        # Should have preliminary results
        assert result.tool_execution_results[0].preliminary_results is not None
        assert len(result.tool_execution_results[0].preliminary_results) == 3
        # Callback should have been called
        assert len(preliminary_results) == 4  # 3 preliminary + 1 final

    @pytest.mark.asyncio
    async def test_context_passed_to_tools(self) -> None:
        """Test that context is properly passed to tool execute methods."""
        executed_contexts: list[ToolContext] = []

        class ContextCaptureTool(RegularTool[CounterParams, dict[str, Any]]):
            name: str = "context_capture"
            description: str = "Captures context"

            def execute(
                self, params: CounterParams, context: ToolContext
            ) -> dict[str, Any]:
                executed_contexts.append(context)
                return {"captured": True}

        tools: list[Any] = [ContextCaptureTool()]
        api_tools = [{"type": "function", "function": {"name": "context_capture"}}]

        responses = [
            create_tool_call_response(
                [create_tool_call_dict("call_1", "context_capture", {"count": 0})]
            ),
            create_tool_call_response(
                [create_tool_call_dict("call_2", "context_capture", {"count": 1})]
            ),
            create_text_response("Done"),
        ]
        response_index = 0

        async def mock_send_request(
            _input: dict[str, object], _tools: list[dict[str, object]]
        ) -> dict[str, object]:
            nonlocal response_index
            response = responses[response_index]
            response_index += 1
            return response

        result = await execute_tool_loop(
            send_request=mock_send_request,
            initial_input={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Test"}],
            },
            tools=tools,
            api_tools=api_tools,
        )

        assert result.rounds_executed == 2
        assert len(executed_contexts) == 2

        # Check first round context
        assert executed_contexts[0].get("number_of_turns") == 1
        assert executed_contexts[0].get("model") == "gpt-4"
        msg_history = executed_contexts[0].get("message_history")
        assert msg_history is not None and len(msg_history) == 1

        # Check second round context
        assert executed_contexts[1].get("number_of_turns") == 2
        prev_results = executed_contexts[1].get("previous_tool_results")
        assert prev_results is not None
        assert len(prev_results) == 1


class TestOrchestrationHelpers:
    """Tests for orchestration helper functions."""

    def test_summarize_tool_executions_success(self) -> None:
        """Test summarizing successful tool executions."""
        results = [
            ToolExecutionResult(
                tool_call_id="call_1",
                tool_name="counter",
                result={"count": 1},
            ),
            ToolExecutionResult(
                tool_call_id="call_2",
                tool_name="calculator",
                result={"result": 8},
            ),
        ]

        summary = summarize_tool_executions(results)

        assert "✅" in summary
        assert "counter" in summary
        assert "calculator" in summary
        assert "call_1" in summary
        assert "call_2" in summary

    def test_summarize_tool_executions_with_errors(self) -> None:
        """Test summarizing tool executions with errors."""
        results = [
            ToolExecutionResult(
                tool_call_id="call_1",
                tool_name="counter",
                result={"count": 1},
            ),
            ToolExecutionResult(
                tool_call_id="call_2",
                tool_name="error_tool",
                result=None,
                error="Something went wrong",
            ),
        ]

        summary = summarize_tool_executions(results)

        assert "✅" in summary
        assert "❌" in summary
        assert "ERROR" in summary
        assert "Something went wrong" in summary

    def test_summarize_tool_executions_with_preliminary(self) -> None:
        """Test summarizing with preliminary results."""
        results = [
            ToolExecutionResult(
                tool_call_id="call_1",
                tool_name="generator",
                result={"final": True},
                preliminary_results=[{"status": "working"}, {"status": "done"}],
            )
        ]

        summary = summarize_tool_executions(results)

        assert "2 preliminary results" in summary

    def test_has_tool_execution_errors_true(self) -> None:
        """Test detecting errors in results (positive case)."""
        results = [
            ToolExecutionResult(
                tool_call_id="call_1",
                tool_name="counter",
                result={"count": 1},
            ),
            ToolExecutionResult(
                tool_call_id="call_2",
                tool_name="error_tool",
                result=None,
                error="Error occurred",
            ),
        ]

        assert has_tool_execution_errors(results) is True

    def test_has_tool_execution_errors_false(self) -> None:
        """Test detecting errors in results (negative case)."""
        results = [
            ToolExecutionResult(
                tool_call_id="call_1",
                tool_name="counter",
                result={"count": 1},
            ),
            ToolExecutionResult(
                tool_call_id="call_2",
                tool_name="calculator",
                result={"result": 8},
            ),
        ]

        assert has_tool_execution_errors(results) is False

    def test_get_tool_execution_errors(self) -> None:
        """Test getting error messages from results."""
        results = [
            ToolExecutionResult(
                tool_call_id="call_1",
                tool_name="counter",
                result={"count": 1},
            ),
            ToolExecutionResult(
                tool_call_id="call_2",
                tool_name="error_tool",
                result=None,
                error="First error",
            ),
            ToolExecutionResult(
                tool_call_id="call_3",
                tool_name="another_error",
                result=None,
                error="Second error",
            ),
        ]

        errors = get_tool_execution_errors(results)

        assert len(errors) == 2
        assert "First error" in errors
        assert "Second error" in errors
