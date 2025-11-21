"""Tests for tool executor functionality.

This test module verifies:
- Successful tool execution (sync and async)
- Pydantic validation (valid and invalid inputs)
- Timeout enforcement (tools that run too long)
- Generator tools with preliminary results
- Error handling and wrapping
- ToolContext passing
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Union

import pytest
from pydantic import BaseModel, Field

from openrouter.call_model.exceptions import ToolValidationError
from openrouter.call_model.tool_executor import (
    DEFAULT_TOOL_TIMEOUT,
    execute_tool,
    find_tool_by_name,
)
from openrouter.call_model.tool_system import (
    GeneratorTool,
    ManualTool,
    ParsedToolCall,
    RegularTool,
)
from openrouter.call_model.types import ToolContext


# Test parameter models
class SimpleParams(BaseModel):
    """Simple parameter model for testing."""

    value: str
    count: int = 1


class WeatherParams(BaseModel):
    """Weather tool parameters."""

    location: str
    unit: str = Field(default="celsius", pattern="^(celsius|fahrenheit)$")


class MathParams(BaseModel):
    """Math operation parameters."""

    a: float
    b: float
    operation: str


# Test tools
class SyncTool(RegularTool[SimpleParams, dict[str, Any]]):
    """Synchronous test tool."""

    name: str = "sync_tool"
    description: str = "Synchronous tool for testing"

    def execute(self, params: SimpleParams, context: ToolContext) -> dict[str, Any]:
        """Execute synchronously."""
        return {
            "value": params.value,
            "count": params.count,
            "turn": context.get("number_of_turns", 0),
        }


class AsyncTool(RegularTool[SimpleParams, dict[str, Any]]):
    """Asynchronous test tool."""

    name: str = "async_tool"
    description: str = "Asynchronous tool for testing"

    async def execute(
        self, params: SimpleParams, context: ToolContext
    ) -> dict[str, Any]:
        """Execute asynchronously with small delay."""
        await asyncio.sleep(0.01)
        return {
            "value": params.value,
            "count": params.count,
            "async": True,
        }


class SlowTool(RegularTool[SimpleParams, dict[str, Any]]):
    """Tool that exceeds timeout."""

    name: str = "slow_tool"
    description: str = "Tool that takes too long"

    async def execute(
        self, params: SimpleParams, context: ToolContext
    ) -> dict[str, Any]:
        """Execute with delay that exceeds timeout."""
        await asyncio.sleep(60)  # Longer than default timeout
        return {"value": params.value}


class ErrorTool(RegularTool[SimpleParams, dict[str, Any]]):
    """Tool that raises an error."""

    name: str = "error_tool"
    description: str = "Tool that always fails"

    def execute(self, params: SimpleParams, context: ToolContext) -> dict[str, Any]:
        """Execute and raise error."""
        raise ValueError(f"Intentional error with value: {params.value}")


class GeneratorTestTool(GeneratorTool[SimpleParams, dict[str, Any]]):
    """Generator tool with preliminary results."""

    name: str = "generator_tool"
    description: str = "Generator tool for testing"

    async def execute(
        self, params: SimpleParams, context: ToolContext
    ) -> AsyncIterator[Union[Any, dict[str, Any]]]:
        """Execute as async generator yielding preliminary results."""
        # Yield preliminary results
        yield {"status": "starting", "value": params.value}
        await asyncio.sleep(0.01)

        for i in range(params.count):
            yield {"status": "processing", "iteration": i + 1}
            await asyncio.sleep(0.01)

        # Final result
        yield {
            "status": "complete",
            "value": params.value,
            "total_iterations": params.count,
        }


class EmptyGeneratorTool(GeneratorTool[SimpleParams, dict[str, Any]]):
    """Generator tool that doesn't emit any values."""

    name: str = "empty_generator"
    description: str = "Generator that emits nothing"

    async def execute(
        self, params: SimpleParams, context: ToolContext
    ) -> AsyncIterator[Union[Any, dict[str, Any]]]:
        """Execute but don't yield anything."""
        # Intentionally empty - violates contract
        if False:  # Make this an async generator
            yield {}
        return
        yield  # Unreachable but makes this an async generator


@pytest.mark.asyncio
class TestToolExecution:
    """Tests for basic tool execution."""

    async def test_sync_tool_execution(self) -> None:
        """Test executing a synchronous tool."""
        tool = SyncTool()
        tool_call = ParsedToolCall(
            id="call_123",
            name="sync_tool",
            arguments={"value": "test", "count": 5},
        )
        context: ToolContext = {"number_of_turns": 3, "message_history": []}

        result = await execute_tool(tool, tool_call, context)

        assert result.tool_call_id == "call_123"
        assert result.tool_name == "sync_tool"
        assert result.error is None
        assert result.result == {"value": "test", "count": 5, "turn": 3}
        assert result.preliminary_results is None

    async def test_async_tool_execution(self) -> None:
        """Test executing an asynchronous tool."""
        tool = AsyncTool()
        tool_call = ParsedToolCall(
            id="call_456",
            name="async_tool",
            arguments={"value": "async_test", "count": 2},
        )
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        result = await execute_tool(tool, tool_call, context)

        assert result.tool_call_id == "call_456"
        assert result.tool_name == "async_tool"
        assert result.error is None
        assert result.result == {"value": "async_test", "count": 2, "async": True}
        assert result.preliminary_results is None

    async def test_tool_receives_context(self) -> None:
        """Test that tools receive proper context."""
        tool = SyncTool()
        tool_call = ParsedToolCall(
            id="call_ctx",
            name="sync_tool",
            arguments={"value": "context_test"},
        )
        context: ToolContext = {
            "number_of_turns": 7,
            "message_history": [{"role": "user", "content": "test"}],
            "model": "gpt-4",
        }

        result = await execute_tool(tool, tool_call, context)

        assert result.error is None
        assert result.result["turn"] == 7  # Context was passed correctly


@pytest.mark.asyncio
class TestValidation:
    """Tests for Pydantic validation."""

    async def test_valid_input_passes_validation(self) -> None:
        """Test that valid inputs pass validation."""
        tool = SyncTool()
        tool_call = ParsedToolCall(
            id="call_valid",
            name="sync_tool",
            arguments={"value": "valid", "count": 10},
        )
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        result = await execute_tool(tool, tool_call, context)

        assert result.error is None
        assert result.result is not None

    async def test_invalid_input_fails_validation(self) -> None:
        """Test that invalid inputs fail validation."""
        tool = SyncTool()
        tool_call = ParsedToolCall(
            id="call_invalid",
            name="sync_tool",
            arguments={"value": "test", "count": "not_a_number"},  # Wrong type
        )
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        result = await execute_tool(tool, tool_call, context)

        assert result.error is not None
        assert "validation failed" in result.error.lower()
        assert "count" in result.error

    async def test_missing_required_field_fails_validation(self) -> None:
        """Test that missing required fields fail validation."""
        tool = SyncTool()
        tool_call = ParsedToolCall(
            id="call_missing",
            name="sync_tool",
            arguments={"count": 5},  # Missing required 'value' field
        )
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        result = await execute_tool(tool, tool_call, context)

        assert result.error is not None
        assert "validation failed" in result.error.lower()
        assert "value" in result.error

    async def test_field_validation_with_pattern(self) -> None:
        """Test field-level validation (e.g., regex patterns)."""

        class WeatherTool(RegularTool[WeatherParams, dict[str, Any]]):
            name: str = "weather"
            description: str = "Get weather"

            def execute(
                self, params: WeatherParams, context: ToolContext
            ) -> dict[str, Any]:
                return {"location": params.location, "unit": params.unit}

        tool = WeatherTool()
        tool_call = ParsedToolCall(
            id="call_pattern",
            name="weather",
            arguments={"location": "SF", "unit": "kelvin"},  # Invalid unit
        )
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        result = await execute_tool(tool, tool_call, context)

        assert result.error is not None
        assert "validation failed" in result.error.lower()


@pytest.mark.asyncio
class TestTimeout:
    """Tests for timeout enforcement."""

    async def test_tool_timeout_is_enforced(self) -> None:
        """Test that tool execution times out after specified duration."""
        tool = SlowTool()
        tool_call = ParsedToolCall(
            id="call_slow", name="slow_tool", arguments={"value": "test"}
        )
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        # Use short timeout for testing
        result = await execute_tool(tool, tool_call, context, timeout=0.1)

        assert result.error is not None
        assert "timed out" in result.error.lower()
        assert "0.1 seconds" in result.error

    async def test_fast_tool_does_not_timeout(self) -> None:
        """Test that fast tools complete before timeout."""
        tool = AsyncTool()
        tool_call = ParsedToolCall(
            id="call_fast", name="async_tool", arguments={"value": "fast"}
        )
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        # Use default timeout which should be plenty
        result = await execute_tool(tool, tool_call, context)

        assert result.error is None
        assert result.result is not None

    async def test_default_timeout_is_30_seconds(self) -> None:
        """Test that default timeout is 30 seconds."""
        assert DEFAULT_TOOL_TIMEOUT == 30.0


@pytest.mark.asyncio
class TestGeneratorTools:
    """Tests for generator tools with preliminary results."""

    async def test_generator_tool_execution(self) -> None:
        """Test generator tool with preliminary results."""
        tool = GeneratorTestTool()
        tool_call = ParsedToolCall(
            id="call_gen",
            name="generator_tool",
            arguments={"value": "gen_test", "count": 3},
        )
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        result = await execute_tool(tool, tool_call, context)

        assert result.error is None
        assert result.result == {
            "status": "complete",
            "value": "gen_test",
            "total_iterations": 3,
        }
        assert result.preliminary_results is not None
        # Should have 1 starting + 3 processing = 4 preliminary
        assert len(result.preliminary_results) == 4

        # Check preliminary results structure
        assert result.preliminary_results[0]["status"] == "starting"
        assert result.preliminary_results[1]["status"] == "processing"
        assert result.preliminary_results[1]["iteration"] == 1

    async def test_generator_with_callback(self) -> None:
        """Test generator tool with preliminary result callback."""
        tool = GeneratorTestTool()
        tool_call = ParsedToolCall(
            id="call_gen_cb",
            name="generator_tool",
            arguments={"value": "callback_test", "count": 2},
        )
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        # Collect preliminary results via callback
        collected_results: list[tuple[str, Any]] = []

        def on_preliminary(tool_call_id: str, result: Any) -> None:
            collected_results.append((tool_call_id, result))

        result = await execute_tool(
            tool, tool_call, context, on_preliminary_result=on_preliminary
        )

        assert result.error is None
        # Callback should receive all yields (including final)
        assert len(collected_results) == 4  # starting + 2 processing + final
        # All should have same tool_call_id
        assert all(tid == "call_gen_cb" for tid, _ in collected_results)

    async def test_empty_generator_fails(self) -> None:
        """Test that generator with no yields fails."""
        tool = EmptyGeneratorTool()
        tool_call = ParsedToolCall(
            id="call_empty",
            name="empty_generator",
            arguments={"value": "test"},
        )
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        result = await execute_tool(tool, tool_call, context)

        assert result.error is not None
        assert "without emitting any values" in result.error.lower()


@pytest.mark.asyncio
class TestErrorHandling:
    """Tests for error handling and wrapping."""

    async def test_tool_error_is_captured(self) -> None:
        """Test that tool execution errors are captured."""
        tool = ErrorTool()
        tool_call = ParsedToolCall(
            id="call_error",
            name="error_tool",
            arguments={"value": "error_test"},
        )
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        result = await execute_tool(tool, tool_call, context)

        assert result.error is not None
        assert "error_test" in result.error.lower()
        assert result.result is None

    async def test_manual_tool_returns_error(self) -> None:
        """Test that manual tools cannot be executed automatically."""
        tool = ManualTool[SimpleParams, dict[str, Any]](
            name="manual_tool", description="Manual tool"
        )
        tool_call = ParsedToolCall(
            id="call_manual", name="manual_tool", arguments={"value": "test"}
        )
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        result = await execute_tool(tool, tool_call, context)

        assert result.error is not None
        assert "manual tool" in result.error.lower()
        assert "user intervention" in result.error.lower()

    async def test_error_includes_context(self) -> None:
        """Test that errors include helpful context."""
        tool = ErrorTool()
        tool_call = ParsedToolCall(
            id="call_ctx_error",
            name="error_tool",
            arguments={"value": "context_error"},
        )
        context: ToolContext = {"number_of_turns": 5, "message_history": []}

        result = await execute_tool(tool, tool_call, context)

        assert result.error is not None
        # Error should mention the tool name
        assert "error_tool" in result.error.lower()


@pytest.mark.asyncio
class TestUtilityFunctions:
    """Tests for utility functions."""

    async def test_find_tool_by_name_finds_existing_tool(self) -> None:
        """Test finding a tool by name."""
        tool1 = SyncTool()
        tool2 = AsyncTool()
        tool3 = ErrorTool()
        tools = [tool1, tool2, tool3]

        found = find_tool_by_name(tools, "async_tool")

        assert found is not None
        assert found.name == "async_tool"
        assert isinstance(found, AsyncTool)

    async def test_find_tool_by_name_returns_none_for_missing(self) -> None:
        """Test that find_tool_by_name returns None for missing tool."""
        tool1 = SyncTool()
        tool2 = AsyncTool()
        tools = [tool1, tool2]

        found = find_tool_by_name(tools, "nonexistent_tool")

        assert found is None

    async def test_find_tool_by_name_with_empty_list(self) -> None:
        """Test finding tool in empty list."""
        found = find_tool_by_name([], "any_tool")

        assert found is None


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests combining multiple features."""

    async def test_full_workflow_with_validation_and_execution(self) -> None:
        """Test complete workflow: validation -> execution -> result."""

        class CalculatorParams(BaseModel):
            operation: str
            a: float
            b: float

        class CalculatorTool(RegularTool[CalculatorParams, float]):
            name: str = "calculator"
            description: str = "Perform math operations"

            def execute(self, params: CalculatorParams, context: ToolContext) -> float:
                ops = {
                    "add": params.a + params.b,
                    "subtract": params.a - params.b,
                    "multiply": params.a * params.b,
                    "divide": params.a / params.b if params.b != 0 else 0,
                }
                return ops.get(params.operation, 0)

        tool = CalculatorTool()
        tool_call = ParsedToolCall(
            id="call_calc",
            name="calculator",
            arguments={"operation": "add", "a": 10.5, "b": 5.5},
        )
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        result = await execute_tool(tool, tool_call, context)

        assert result.error is None
        assert result.result == 16.0

    async def test_multiple_tools_in_sequence(self) -> None:
        """Test executing multiple tools in sequence."""
        tools = [SyncTool(), AsyncTool(), GeneratorTestTool()]

        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        # Execute each tool
        results = []
        for i, tool in enumerate(tools):
            tool_call = ParsedToolCall(
                id=f"call_{i}",
                name=tool.name,
                arguments={"value": f"test_{i}", "count": 1},
            )
            result = await execute_tool(tool, tool_call, context)
            results.append(result)

        # All should succeed
        assert all(r.error is None for r in results)
        assert len(results) == 3

        # Generator should have preliminary results
        assert results[2].preliminary_results is not None
