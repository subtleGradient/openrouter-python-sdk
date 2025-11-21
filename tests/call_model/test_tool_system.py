"""Tests for tool system definitions.

This module tests the tool system including:
- BaseTool, RegularTool, GeneratorTool, ManualTool
- Pydantic model integration
- @tool decorator pattern
- JSON schema generation
- Type safety and validation
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel, Field

from openrouter.call_model.tool_system import (
    BaseTool,
    GeneratorTool,
    ManualTool,
    ParsedToolCall,
    RegularTool,
    ToolExecutionResult,
    tool,
)
from openrouter.call_model.types import ToolContext


# Test parameter models
class WeatherParams(BaseModel):
    """Parameters for weather tool."""

    location: str = Field(..., description="City name or coordinates")
    unit: str = Field(default="celsius", description="Temperature unit")


class CalculatorParams(BaseModel):
    """Parameters for calculator tool."""

    operation: str = Field(
        ..., description="Math operation: add, subtract, multiply, divide"
    )
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")


class SearchParams(BaseModel):
    """Parameters for search tool."""

    query: str = Field(..., description="Search query")
    max_results: int = Field(default=10, description="Maximum results to return")


# Test tool implementations
class WeatherTool(RegularTool[WeatherParams, dict[str, Any]]):
    """Synchronous weather tool."""

    def __init__(self) -> None:
        """Initialize weather tool."""
        super().__init__(
            name="weather", description="Get current weather for a location"
        )

    def execute(self, params: WeatherParams, context: ToolContext) -> dict[str, Any]:
        """Execute weather lookup."""
        return {
            "location": params.location,
            "temperature": 22,
            "unit": params.unit,
        }


from collections.abc import AsyncIterator, Awaitable


class AsyncCalculatorTool(RegularTool[CalculatorParams, float]):
    """Asynchronous calculator tool."""

    def __init__(self) -> None:
        """Initialize calculator tool."""
        super().__init__(name="calculator", description="Perform basic math operations")

    def execute(
        self, params: CalculatorParams, context: ToolContext
    ) -> Awaitable[float]:
        """Execute calculation asynchronously."""

        async def _execute() -> float:
            await asyncio.sleep(0.001)  # Simulate async work
            ops = {
                "add": params.a + params.b,
                "subtract": params.a - params.b,
                "multiply": params.a * params.b,
                "divide": params.a / params.b if params.b != 0 else 0.0,
            }
            return ops.get(params.operation, 0.0)

        return _execute()


class SearchGeneratorTool(GeneratorTool[SearchParams, dict[str, Any]]):
    """Generator tool with preliminary results."""

    def __init__(self) -> None:
        """Initialize search tool."""
        super().__init__(name="search", description="Search with progress updates")

    async def execute(
        self, params: SearchParams, context: ToolContext
    ) -> AsyncIterator[Any]:
        """Execute search with preliminary results."""
        # Yield preliminary status updates
        yield {"status": "searching", "query": params.query}
        await asyncio.sleep(0.001)
        yield {"status": "processing", "count": 5}
        await asyncio.sleep(0.001)
        # Final result
        yield {
            "results": [f"result_{i}" for i in range(params.max_results)],
            "total": params.max_results,
        }


class ConfirmTool(ManualTool[WeatherParams, bool]):
    """Manual tool requiring user intervention."""

    def __init__(self) -> None:
        """Initialize confirm tool."""
        super().__init__(
            name="confirm_action", description="Ask user to confirm an action"
        )


# Tests for ParsedToolCall
class TestParsedToolCall:
    """Tests for ParsedToolCall model."""

    def test_create_parsed_tool_call(self) -> None:
        """Test creating a ParsedToolCall instance."""
        call = ParsedToolCall(
            id="call_123",
            name="weather",
            arguments={"location": "San Francisco", "unit": "celsius"},
        )
        assert call.id == "call_123"
        assert call.name == "weather"
        assert call.arguments["location"] == "San Francisco"

    def test_parsed_tool_call_validation(self) -> None:
        """Test Pydantic validation on ParsedToolCall."""
        with pytest.raises(Exception):  # Pydantic validation error
            _ = ParsedToolCall(id="call_123")  # type: ignore[call-arg]


# Tests for ToolExecutionResult
class TestToolExecutionResult:
    """Tests for ToolExecutionResult model."""

    def test_create_execution_result(self) -> None:
        """Test creating a ToolExecutionResult instance."""
        result = ToolExecutionResult(
            tool_call_id="call_123",
            tool_name="weather",
            result={"temp": 22},
            preliminary_results=[{"status": "fetching"}],
        )
        assert result.tool_call_id == "call_123"
        assert result.tool_name == "weather"
        assert result.result == {"temp": 22}
        assert result.preliminary_results == [{"status": "fetching"}]
        assert result.error is None

    def test_execution_result_with_error(self) -> None:
        """Test ToolExecutionResult with error."""
        result = ToolExecutionResult(
            tool_call_id="call_123",
            tool_name="weather",
            error="Connection timeout",
        )
        assert result.error == "Connection timeout"
        assert result.result is None


# Tests for BaseTool
class TestBaseTool:
    """Tests for BaseTool base class."""

    def test_base_tool_attributes(self) -> None:
        """Test BaseTool has required attributes."""
        tool_instance = WeatherTool()
        assert hasattr(tool_instance, "name")
        assert hasattr(tool_instance, "description")
        assert hasattr(tool_instance, "to_json_schema")

    def test_json_schema_generation(self) -> None:
        """Test JSON schema generation from Pydantic model."""
        tool_instance = WeatherTool()
        schema = tool_instance.to_json_schema()

        assert "properties" in schema
        assert "location" in schema["properties"]
        assert "unit" in schema["properties"]
        assert schema["properties"]["location"]["type"] == "string"
        assert schema["properties"]["unit"]["default"] == "celsius"

    def test_json_schema_with_field_descriptions(self) -> None:
        """Test JSON schema includes Field descriptions."""
        tool_instance = WeatherTool()
        schema = tool_instance.to_json_schema()

        assert (
            schema["properties"]["location"]["description"]
            == "City name or coordinates"
        )
        assert schema["properties"]["unit"]["description"] == "Temperature unit"


# Tests for RegularTool
class TestRegularTool:
    """Tests for RegularTool with execute method."""

    def test_sync_execute(self) -> None:
        """Test synchronous execute method."""
        tool_instance = WeatherTool()
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        params = WeatherParams(location="London", unit="fahrenheit")
        result = tool_instance.execute(params, context)

        assert isinstance(result, dict)
        assert result["location"] == "London"
        assert result["unit"] == "fahrenheit"
        assert result["temperature"] == 22

    @pytest.mark.asyncio
    async def test_async_execute(self) -> None:
        """Test asynchronous execute method."""
        tool_instance = AsyncCalculatorTool()
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        params = CalculatorParams(operation="add", a=5.0, b=3.0)
        result = await tool_instance.execute(params, context)

        assert isinstance(result, float)
        assert result == 8.0

    @pytest.mark.asyncio
    async def test_async_calculator_operations(self) -> None:
        """Test all calculator operations."""
        tool_instance = AsyncCalculatorTool()
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        # Test add
        result = await tool_instance.execute(
            CalculatorParams(operation="add", a=10.0, b=5.0), context
        )
        assert result == 15.0

        # Test subtract
        result = await tool_instance.execute(
            CalculatorParams(operation="subtract", a=10.0, b=5.0), context
        )
        assert result == 5.0

        # Test multiply
        result = await tool_instance.execute(
            CalculatorParams(operation="multiply", a=10.0, b=5.0), context
        )
        assert result == 50.0

        # Test divide
        result = await tool_instance.execute(
            CalculatorParams(operation="divide", a=10.0, b=5.0), context
        )
        assert result == 2.0

        # Test divide by zero
        result = await tool_instance.execute(
            CalculatorParams(operation="divide", a=10.0, b=0.0), context
        )
        assert result == 0.0

    def test_regular_tool_not_implemented(self) -> None:
        """Test that RegularTool raises NotImplementedError if execute not overridden."""

        class UnimplementedTool(RegularTool[WeatherParams, dict[str, Any]]):
            def __init__(self) -> None:
                super().__init__(
                    name="unimplemented", description="Tool without execute"
                )

        tool_instance = UnimplementedTool()
        context: ToolContext = {"number_of_turns": 1, "message_history": []}
        params = WeatherParams(location="Test")

        with pytest.raises(NotImplementedError, match="must implement execute method"):
            _ = tool_instance.execute(params, context)


# Tests for GeneratorTool
class TestGeneratorTool:
    """Tests for GeneratorTool with async generator execute."""

    @pytest.mark.asyncio
    async def test_generator_yields_preliminary_results(self) -> None:
        """Test generator tool yields preliminary results."""
        tool_instance = SearchGeneratorTool()
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        params = SearchParams(query="python", max_results=3)
        results: list[Any] = []

        async for event in tool_instance.execute(params, context):
            results.append(event)

        # Should have 3 yields: 2 preliminary + 1 final
        assert len(results) == 3

        # First preliminary result
        assert results[0]["status"] == "searching"
        assert results[0]["query"] == "python"

        # Second preliminary result
        assert results[1]["status"] == "processing"
        assert results[1]["count"] == 5

        # Final result
        assert "results" in results[2]
        assert len(results[2]["results"]) == 3
        assert results[2]["total"] == 3

    @pytest.mark.asyncio
    async def test_generator_tool_not_implemented(self) -> None:
        """Test that GeneratorTool raises NotImplementedError if execute not overridden."""

        class UnimplementedGeneratorTool(GeneratorTool[SearchParams, dict[str, Any]]):
            def __init__(self) -> None:
                super().__init__(
                    name="unimplemented", description="Generator without execute"
                )

        tool_instance = UnimplementedGeneratorTool()
        context: ToolContext = {"number_of_turns": 1, "message_history": []}
        params = SearchParams(query="test")

        with pytest.raises(NotImplementedError, match="must implement execute method"):
            async for _ in tool_instance.execute(params, context):
                pass


# Tests for ManualTool
class TestManualTool:
    """Tests for ManualTool without execute method."""

    def test_manual_tool_has_no_execute(self) -> None:
        """Test that ManualTool doesn't have execute method to call."""
        tool_instance = ConfirmTool()

        # ManualTool should not have a callable execute
        # It inherits from BaseTool but doesn't define execute
        assert not hasattr(tool_instance, "execute") or not callable(
            getattr(tool_instance, "execute", None)
        )

    def test_manual_tool_has_schema(self) -> None:
        """Test that ManualTool can still generate JSON schema."""
        tool_instance = ConfirmTool()
        schema = tool_instance.to_json_schema()

        assert "properties" in schema
        assert "location" in schema["properties"]


# Tests for @tool decorator
class TestToolDecorator:
    """Tests for @tool decorator pattern."""

    def test_decorator_creates_tool(self) -> None:
        """Test decorator converts Pydantic model to tool."""

        @tool
        class DecoratorWeatherTool(BaseModel):
            """Get current weather for a location."""

            location: str
            unit: str = "celsius"

            def execute(self, context: ToolContext) -> dict[str, Any]:
                return {"temp": 22, "unit": self.unit}

        # Decorator returns a tool class type
        # The class has default name/description set
        assert hasattr(DecoratorWeatherTool, "name")
        assert hasattr(DecoratorWeatherTool, "description")

        # Create instance - name is a class attribute with default
        tool_instance = DecoratorWeatherTool()

        # Check tool attributes
        assert tool_instance.name == "decoratorweathertool"
        assert tool_instance.description == "Get current weather for a location."

    def test_decorator_generates_schema(self) -> None:
        """Test decorator-created tool generates correct schema."""

        @tool
        class SchemaTestTool(BaseModel):
            """Test schema generation."""

            required_field: str
            optional_field: int = 42

            def execute(self, context: ToolContext) -> dict[str, Any]:
                return {"result": "ok"}

        tool_instance = SchemaTestTool()
        schema = tool_instance.to_json_schema()

        assert "properties" in schema
        assert "required_field" in schema["properties"]
        assert "optional_field" in schema["properties"]
        assert schema["properties"]["optional_field"]["default"] == 42

    def test_decorator_sync_execute(self) -> None:
        """Test decorator tool with sync execute."""

        @tool
        class SyncTool(BaseModel):
            """Sync tool."""

            value: int

            def execute(self, context: ToolContext) -> dict[str, Any]:
                return {"doubled": self.value * 2}

        tool_instance = SyncTool()
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        # Create parameter instance using the original model class
        params = tool_instance._model_class(value=5)
        result = tool_instance.execute(params, context)

        assert result == {"doubled": 10}

    @pytest.mark.asyncio
    async def test_decorator_async_execute(self) -> None:
        """Test decorator tool with async execute."""

        @tool
        class AsyncTool(BaseModel):
            """Async tool."""

            value: int

            async def execute(self, context: ToolContext) -> dict[str, Any]:
                await asyncio.sleep(0.001)
                return {"tripled": self.value * 3}

        tool_instance = AsyncTool()
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        params = tool_instance._model_class(value=5)
        result = await tool_instance.execute(params, context)  # type: ignore[misc]

        assert result == {"tripled": 15}

    def test_decorator_without_execute_raises(self) -> None:
        """Test decorator tool without execute method raises error."""

        @tool
        class NoExecuteTool(BaseModel):
            """Tool without execute."""

            value: int

        tool_instance = NoExecuteTool()
        context: ToolContext = {"number_of_turns": 1, "message_history": []}

        params = tool_instance._model_class(value=5)

        with pytest.raises(NotImplementedError, match="must implement execute method"):
            _ = tool_instance.execute(params, context)

    def test_decorator_uses_docstring(self) -> None:
        """Test decorator uses class docstring as description."""

        @tool
        class DocstringTool(BaseModel):
            """This is a detailed description of what this tool does."""

            value: str

            def execute(self, context: ToolContext) -> dict[str, Any]:
                return {}

        tool_instance = DocstringTool()
        assert (
            tool_instance.description
            == "This is a detailed description of what this tool does."
        )

    def test_decorator_empty_docstring(self) -> None:
        """Test decorator handles missing docstring gracefully."""

        @tool
        class NoDocstringTool(BaseModel):
            value: str

            def execute(self, context: ToolContext) -> dict[str, Any]:
                return {}

        tool_instance = NoDocstringTool()
        assert tool_instance.description == ""


# Integration tests
class TestToolSystemIntegration:
    """Integration tests for complete tool system."""

    @pytest.mark.asyncio
    async def test_multiple_tools_different_types(self) -> None:
        """Test using multiple tool types together."""
        context: ToolContext = {
            "number_of_turns": 1,
            "message_history": [],
            "model": "test-model",
        }

        # Regular sync tool
        weather_tool = WeatherTool()
        weather_result = weather_tool.execute(WeatherParams(location="NYC"), context)
        assert weather_result["location"] == "NYC"

        # Regular async tool
        calc_tool = AsyncCalculatorTool()
        calc_result = await calc_tool.execute(
            CalculatorParams(operation="multiply", a=6.0, b=7.0), context
        )
        assert calc_result == 42.0

        # Generator tool
        search_tool = SearchGeneratorTool()
        search_results: list[Any] = []
        async for event in search_tool.execute(
            SearchParams(query="test", max_results=2), context
        ):
            search_results.append(event)
        assert len(search_results) == 3  # 2 preliminary + 1 final

        # Manual tool (just check it exists)
        confirm_tool = ConfirmTool()
        assert confirm_tool.name == "confirm_action"

    def test_all_tools_generate_valid_schemas(self) -> None:
        """Test that all tool types generate valid JSON schemas."""
        tools = [
            WeatherTool(),
            AsyncCalculatorTool(),
            SearchGeneratorTool(),
            ConfirmTool(),
        ]

        for tool_instance in tools:
            schema = tool_instance.to_json_schema()
            assert isinstance(schema, dict)
            assert "properties" in schema

    @pytest.mark.asyncio
    async def test_tool_context_usage(self) -> None:
        """Test that tools can access context information."""

        class ContextAwareTool(RegularTool[WeatherParams, dict[str, Any]]):
            """Tool that uses context."""

            def __init__(self) -> None:
                super().__init__(name="context_aware", description="Uses turn context")

            def execute(
                self, params: WeatherParams, context: ToolContext
            ) -> dict[str, Any]:
                return {
                    "turn": context.get("number_of_turns", 0),
                    "model": context.get("model"),
                    "param": params.location,
                }

        tool_instance = ContextAwareTool()
        context: ToolContext = {
            "number_of_turns": 3,
            "message_history": [],
            "model": "gpt-4",
        }

        result = tool_instance.execute(WeatherParams(location="Test"), context)
        assert result["turn"] == 3
        assert result["model"] == "gpt-4"
        assert result["param"] == "Test"


# Type checking tests
class TestTypeAnnotations:
    """Tests to verify type annotations work correctly."""

    def test_tool_type_parameters(self) -> None:
        """Test that generic type parameters are preserved."""
        # This primarily tests that the code type-checks correctly
        weather_tool: RegularTool[WeatherParams, dict[str, Any]] = WeatherTool()
        calc_tool: RegularTool[CalculatorParams, float] = AsyncCalculatorTool()
        search_tool: GeneratorTool[SearchParams, dict[str, Any]] = SearchGeneratorTool()
        manual_tool: ManualTool[WeatherParams, bool] = ConfirmTool()

        # Verify they're the expected types
        assert isinstance(weather_tool, RegularTool)
        assert isinstance(calc_tool, RegularTool)
        assert isinstance(search_tool, GeneratorTool)
        assert isinstance(manual_tool, ManualTool)
