"""Main entry point for the call_model API.

This module provides the primary public interface for calling OpenRouter models
with automatic tool orchestration and multiple consumption patterns.

The call_model() function:
- Makes exactly one API call via beta.responses.send(stream=True)
- Converts Pydantic tools to API format using model_json_schema()
- Returns a ResponseWrapper for flexible consumption
- Integrates with tool_orchestrator for automatic tool execution

Example:
    Basic usage:

    >>> from openrouter import OpenRouter
    >>> from openrouter.call_model import call_model
    >>>
    >>> async def main():
    ...     client = OpenRouter(api_key="...")
    ...     response = await call_model(
    ...         client=client,
    ...         request={"model": "gpt-4", "input": "Hello!"}
    ...     )
    ...     text = await response.get_text()
    ...     print(text)

    With tools:

    >>> from pydantic import BaseModel
    >>> from openrouter.call_model import call_model, RegularTool
    >>>
    >>> class WeatherParams(BaseModel):
    ...     location: str
    ...     unit: str = "celsius"
    >>>
    >>> class WeatherTool(RegularTool[WeatherParams, dict]):
    ...     name: str = "get_weather"
    ...     description: str = "Get current weather"
    ...
    ...     async def execute(self, params: WeatherParams, context):
    ...         return {"temp": 22, "unit": params.unit}
    >>>
    >>> async def main():
    ...     client = OpenRouter(api_key="...")
    ...     response = await call_model(
    ...         client=client,
    ...         request={"model": "gpt-4", "input": "What's the weather in SF?"},
    ...         tools=[WeatherTool()],
    ...         max_tool_rounds=3
    ...     )
    ...     message = await response.get_message()  # Tools auto-executed!
    ...     print(message)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from openrouter.call_model.response_wrapper import ResponseWrapper

if TYPE_CHECKING:
    from openrouter.call_model.tool_system import BaseTool


def convert_tools_to_api_format(
    tools: list[BaseTool[Any, Any]],
) -> list[dict[str, Any]]:
    """Convert Pydantic-based tools to API format.

    Takes a list of BaseTool instances and converts them to the format
    expected by the OpenResponses API. Each tool's Pydantic schema is
    extracted using model_json_schema() and formatted as a function tool.

    Args:
        tools: List of BaseTool instances with Pydantic parameter models

    Returns:
        list[dict[str, Any]]: Tools in API format (type, function with schema)

    Example:
        >>> tools = [WeatherTool(), CalculatorTool()]
        >>> api_tools = convert_tools_to_api_format(tools)
        >>> assert api_tools[0]["type"] == "function"
        >>> assert "name" in api_tools[0]["function"]
        >>> assert "parameters" in api_tools[0]["function"]
    """
    api_tools: list[dict[str, Any]] = []

    for tool in tools:
        # Extract JSON schema from the tool's parameter model
        # The to_json_schema() method gets the schema from the generic type parameter
        parameters_schema = tool.to_json_schema()

        # Build API tool format matching ToolDefinitionJSON structure
        api_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters_schema,
            },
        }

        api_tools.append(api_tool)

    return api_tools


async def call_model(
    client: Any,  # OpenRouter client instance
    request: dict[str, Any],
    *,
    tools: list[BaseTool[Any, Any]] | None = None,
    max_tool_rounds: int | Callable[[Any], bool] | None = None,
    options: Any | None = None,  # RequestOptions
) -> ResponseWrapper:
    """Create a response with multiple consumption patterns.

    This is the main entry point for the call_model API. It creates a streaming
    response from the OpenResponses API and wraps it in a ResponseWrapper that
    provides multiple ways to consume the response.

    The function:
    1. Converts Pydantic tools to API format using model_json_schema()
    2. Makes exactly one API call via client.beta.responses.send(stream=True)
    3. Returns a ResponseWrapper configured with tools and orchestration
    4. The wrapper handles tool execution automatically when consuming results

    Args:
        client: OpenRouter client instance (from openrouter.OpenRouter)
        request: Request parameters (same as beta.responses.send)
                 Must include "model" and "input" or other required fields
        tools: Optional list of BaseTool instances for automatic execution
               Each tool must have a Pydantic parameter model and execute method
        max_tool_rounds: Maximum tool execution rounds (default: 5, max: 10)
                        Can be an int or a callable that returns bool
        options: Optional request options (timeout, headers, etc.)

    Returns:
        ResponseWrapper: Wrapper with multiple consumption methods:
                        - await response.get_message() - Full message with tools
                        - await response.get_text() - Text content only
                        - async for delta in response.get_text_stream() - Stream text
                        - async for event in response.get_full_stream() - All events

    Raises:
        ValueError: If request is missing required fields
        Exception: Any error from the API or tool execution

    Example:
        Simple text extraction:

        >>> response = await call_model(
        ...     client=client,
        ...     request={"model": "openai/gpt-4", "input": "Hello!"}
        ... )
        >>> text = await response.get_text()
        >>> print(text)

        With automatic tool execution:

        >>> response = await call_model(
        ...     client=client,
        ...     request={"model": "openai/gpt-4", "input": "What's the weather?"},
        ...     tools=[weather_tool, calculator_tool],
        ...     max_tool_rounds=3
        ... )
        >>> message = await response.get_message()  # Tools executed automatically
        >>> print(message["content"])

        Stream text deltas:

        >>> response = await call_model(
        ...     client=client,
        ...     request={"model": "openai/gpt-4", "input": "Write a story"}
        ... )
        >>> async for delta in response.get_text_stream():
        ...     print(delta, end="", flush=True)
    """
    # Validate request has required fields
    if not isinstance(request, dict):
        raise ValueError("request must be a dictionary")

    # Make a copy to avoid modifying the original
    api_request = dict(request)

    # Convert tools to API format if provided
    api_tools: list[dict[str, Any]] | None = None
    if tools:
        api_tools = convert_tools_to_api_format(tools)
        api_request["tools"] = api_tools

    # Create ResponseWrapper with the request and configuration
    # The wrapper will make the actual API call when first consumed
    wrapper = ResponseWrapper(
        client=client,
        request=api_request,
        tools=tools,
        max_tool_rounds=max_tool_rounds,
        options=options,
    )

    return wrapper
