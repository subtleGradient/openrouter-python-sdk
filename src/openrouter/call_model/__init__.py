"""High-level API for OpenRouter model interactions with tool orchestration.

This module provides a Pythonic interface for calling OpenRouter models with
automatic tool execution, streaming support, and multiple consumption patterns.

The call_model API is designed to:
- Provide a simple, high-level interface similar to TypeScript SDK
- Support automatic tool orchestration with validation
- Enable multiple consumption patterns (streaming, complete message, text-only)
- Allow stream reuse without additional API calls
- Follow Python conventions (snake_case, async/await, type hints)

Example:
    Basic usage with tools:

    >>> from openrouter import OpenRouter
    >>> from openrouter.call_model import call_model
    >>> from pydantic import BaseModel
    >>>
    >>> class WeatherParams(BaseModel):
    ...     location: str
    ...     unit: str = "celsius"
    >>>
    >>> async def main():
    ...     client = OpenRouter(api_key="...")
    ...     response = await call_model(
    ...         client=client,
    ...         request={"model": "gpt-4", "messages": [...]},
    ...         tools=[WeatherParams],
    ...         max_tool_rounds=5
    ...     )
    ...     text = await response.get_text()
    ...     print(text)

For more examples, see the examples/ directory.
"""

from .exceptions import (
    CallModelError,
    MaxToolRoundsExceededError,
    StreamInterruptedError,
    ToolExecutionError,
    ToolValidationError,
)
from .response_wrapper import ResponseWrapper
from .reusable_stream import ReusableStream
from .tool_system import (
    BaseTool,
    GeneratorTool,
    ManualTool,
    ParsedToolCall,
    RegularTool,
    ToolExecutionResult,
    tool,
)
from .types import (
    CachedData,
    EventType,
    ResponseState,
    StreamEvent,
    ToolCallId,
    ToolContext,
    ToolType,
)

# Note: tool_executor and tool_orchestrator will be added in PR 4.1 (Integration)

__all__ = [
    # Exception classes
    "CallModelError",
    "MaxToolRoundsExceededError",
    "StreamInterruptedError",
    "ToolExecutionError",
    "ToolValidationError",
    # Core classes
    "ResponseWrapper",
    "ReusableStream",
    # Tool system
    "BaseTool",
    "GeneratorTool",
    "ManualTool",
    "ParsedToolCall",
    "RegularTool",
    "ToolExecutionResult",
    "tool",
    # Type definitions
    "CachedData",
    "EventType",
    "ResponseState",
    "StreamEvent",
    "ToolCallId",
    "ToolContext",
    "ToolType",
]

# Module metadata
__version__ = "0.1.0"
__author__ = "OpenRouter"
