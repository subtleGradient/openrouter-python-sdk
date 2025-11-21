"""Integration tests for call_model main entry point.

Tests the call_model() function to ensure:
- Single API call made (FR-1.1.4)
- Request parameters passed correctly
- Tool conversion works (Pydantic to API format)
- ResponseWrapper is returned with correct configuration
- Integration with existing SDK beta.responses.send
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from openrouter.call_model import call_model, convert_tools_to_api_format  # pyright: ignore[reportAttributeAccessIssue]
from openrouter.call_model.response_wrapper import ResponseWrapper  # pyright: ignore[reportAttributeAccessIssue]
from openrouter.call_model.tool_system import RegularTool  # pyright: ignore[reportAttributeAccessIssue]


class WeatherParams(BaseModel):
    """Parameters for weather tool."""

    location: str
    unit: str = "celsius"


class WeatherTool(RegularTool[WeatherParams, dict[str, Any]]):
    """Tool for getting weather information."""

    name: str = "get_weather"
    description: str = "Get current weather for a location"

    async def execute(self, params: WeatherParams, context: Any) -> dict[str, Any]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Execute the weather tool."""
        return {"temp": 22, "unit": params.unit, "location": params.location}


class TestToolConversion:
    """Test convert_tools_to_api_format function."""

    def test_converts_single_tool_to_api_format(self) -> None:
        """Test converting a single tool to API format."""
        tool = WeatherTool()
        api_tools = convert_tools_to_api_format([tool])

        assert len(api_tools) == 1
        assert api_tools[0]["type"] == "function"
        assert api_tools[0]["function"]["name"] == "get_weather"
        assert (
            api_tools[0]["function"]["description"]
            == "Get current weather for a location"
        )
        assert "parameters" in api_tools[0]["function"]

        # Verify schema contains expected fields from WeatherParams
        schema = api_tools[0]["function"]["parameters"]
        assert "properties" in schema
        assert "location" in schema["properties"]
        assert "unit" in schema["properties"]

    def test_converts_multiple_tools(self) -> None:
        """Test converting multiple tools."""

        class CalculatorParams(BaseModel):
            a: float
            b: float

        class CalculatorTool(RegularTool[CalculatorParams, float]):
            name: str = "calculator"
            description: str = "Add two numbers"

            async def execute(self, params: CalculatorParams, context: Any) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
                return params.a + params.b

        tools = [WeatherTool(), CalculatorTool()]
        api_tools = convert_tools_to_api_format(tools)

        assert len(api_tools) == 2
        assert api_tools[0]["function"]["name"] == "get_weather"
        assert api_tools[1]["function"]["name"] == "calculator"

    def test_empty_tool_list(self) -> None:
        """Test with empty tool list."""
        api_tools = convert_tools_to_api_format([])
        assert api_tools == []


class TestCallModel:
    """Test call_model() entry point function."""

    @pytest.mark.asyncio
    async def test_returns_response_wrapper(self) -> None:
        """Test that call_model returns a ResponseWrapper instance."""
        # Create mock client with async generator
        mock_client = MagicMock()
        mock_client.beta = MagicMock()
        mock_client.beta.responses = MagicMock()

        async def mock_send_async(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            """Mock streaming response - proper async generator."""
            # Empty stream for this test
            if False:
                yield {}

        mock_client.beta.responses.send_async = mock_send_async

        request = {"model": "openai/gpt-4", "input": "Hello!"}

        # Call the function
        response = await call_model(client=mock_client, request=request)

        # Verify it returns a ResponseWrapper
        assert isinstance(response, ResponseWrapper)
        assert response.state.value == "initialized"

    @pytest.mark.asyncio
    async def test_passes_request_parameters(self) -> None:
        """Test that request parameters are passed through correctly."""
        mock_client = MagicMock()
        mock_client.beta = MagicMock()
        mock_client.beta.responses = MagicMock()

        # Track the call
        call_kwargs: dict[str, Any] = {}

        async def mock_send_async(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            """Capture call arguments."""
            nonlocal call_kwargs
            call_kwargs.update(kwargs)
            # Return empty stream
            if False:
                yield {}

        mock_client.beta.responses.send_async = mock_send_async

        request = {
            "model": "openai/gpt-4",
            "input": "Test message",
            "temperature": 0.7,
            "max_output_tokens": 100,
        }

        # Call without tools
        response = await call_model(client=mock_client, request=request)

        # Trigger stream initialization to make the API call
        await response._init_stream()  # pyright: ignore[reportPrivateUsage]

        # Verify request parameters were passed
        assert call_kwargs.get("stream") is True
        assert call_kwargs.get("model") == "openai/gpt-4"
        assert call_kwargs.get("input") == "Test message"
        assert call_kwargs.get("temperature") == 0.7
        assert call_kwargs.get("max_output_tokens") == 100

    @pytest.mark.asyncio
    async def test_converts_and_passes_tools(self) -> None:
        """Test that tools are converted to API format and passed correctly."""
        mock_client = MagicMock()
        mock_client.beta = MagicMock()
        mock_client.beta.responses = MagicMock()

        call_kwargs: dict[str, Any] = {}

        async def mock_send_async(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            """Capture call arguments."""
            nonlocal call_kwargs
            call_kwargs.update(kwargs)
            if False:
                yield {}

        mock_client.beta.responses.send_async = mock_send_async

        request = {"model": "openai/gpt-4", "input": "What's the weather?"}
        tools = [WeatherTool()]

        # Call with tools
        response = await call_model(
            client=mock_client,
            request=request,
            tools=tools,
        )

        # Trigger stream initialization
        await response._init_stream()  # pyright: ignore[reportPrivateUsage]

        # Verify tools were converted and passed
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["type"] == "function"
        assert call_kwargs["tools"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_sets_max_tool_rounds(self) -> None:
        """Test that max_tool_rounds is configured on ResponseWrapper."""
        mock_client = MagicMock()
        mock_client.beta = MagicMock()
        mock_client.beta.responses = MagicMock()

        async def mock_send_async(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            """Mock send."""
            if False:
                yield {}

        mock_client.beta.responses.send_async = mock_send_async

        request = {"model": "openai/gpt-4", "input": "Test"}
        tools = [WeatherTool()]

        # Call with custom max_tool_rounds
        response = await call_model(
            client=mock_client,
            request=request,
            tools=tools,
            max_tool_rounds=3,
        )

        # Verify max_tool_rounds is set on the wrapper
        assert response._max_tool_rounds == 3  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.asyncio
    async def test_single_api_call_multiple_init_attempts(self) -> None:
        """Test that only one API call is made even with multiple init attempts."""
        mock_client = MagicMock()
        mock_client.beta = MagicMock()
        mock_client.beta.responses = MagicMock()

        call_count = 0

        async def mock_send_async(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            """Track API calls."""
            nonlocal call_count
            call_count += 1
            # Simple response stream
            yield {"type": "response.output_text.delta", "delta": "Hello"}
            yield {"type": "response.done"}

        mock_client.beta.responses.send_async = mock_send_async

        request = {"model": "openai/gpt-4", "input": "Hello"}

        # Create response
        response = await call_model(client=mock_client, request=request)

        # Multiple init attempts
        await response._init_stream()  # pyright: ignore[reportPrivateUsage]
        await response._init_stream()  # pyright: ignore[reportPrivateUsage]
        await response._init_stream()  # pyright: ignore[reportPrivateUsage]

        # Verify only one call was made (idempotent init)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_validates_request_is_dict(self) -> None:
        """Test that request must be a dictionary."""
        mock_client = MagicMock()

        # Try with non-dict request
        with pytest.raises(ValueError, match="request must be a dictionary"):
            await call_model(
                client=mock_client,
                request="invalid",  # pyright: ignore[reportArgumentType]
            )

    @pytest.mark.asyncio
    async def test_does_not_modify_original_request(self) -> None:
        """Test that the original request dict is not modified."""
        mock_client = MagicMock()
        mock_client.beta = MagicMock()
        mock_client.beta.responses = MagicMock()

        async def mock_send_async(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            """Mock send."""
            if False:
                yield {}

        mock_client.beta.responses.send_async = mock_send_async

        # Original request without tools
        original_request = {"model": "openai/gpt-4", "input": "Test"}
        original_keys = set(original_request.keys())

        tools = [WeatherTool()]

        # Call with tools
        await call_model(
            client=mock_client,
            request=original_request,
            tools=tools,
        )

        # Verify original request was not modified
        assert set(original_request.keys()) == original_keys
        assert "tools" not in original_request

    @pytest.mark.asyncio
    async def test_tools_stored_on_wrapper(self) -> None:
        """Test that tools are correctly stored on the ResponseWrapper."""
        mock_client = MagicMock()
        mock_client.beta = MagicMock()
        mock_client.beta.responses = MagicMock()

        async def mock_send_async(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            """Mock send."""
            if False:
                yield {}

        mock_client.beta.responses.send_async = mock_send_async

        request = {"model": "openai/gpt-4", "input": "Test"}
        tools = [WeatherTool()]

        # Call with tools
        response = await call_model(
            client=mock_client,
            request=request,
            tools=tools,
        )

        # Verify tools are stored on the wrapper
        assert response._tools == tools  # pyright: ignore[reportPrivateUsage]
        assert len(response._tools) == 1  # pyright: ignore[reportPrivateUsage]
        assert isinstance(response._tools[0], WeatherTool)  # pyright: ignore[reportPrivateUsage]
