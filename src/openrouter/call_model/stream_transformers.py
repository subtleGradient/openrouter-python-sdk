"""Stream transformers for filtering and transforming SSE events.

This module provides utilities for transforming and filtering events from
streaming API responses. It includes transformers for extracting text deltas,
filtering tool events, and aggregating events into complete messages.

Key features:
- Text delta extraction for streaming text content
- Tool event filtering for tool-related events
- Message aggregation from streaming events
- Proper resource cleanup with async context managers

Example:
    >>> async def process_text_stream(stream: ReusableStream):
    ...     async for text_delta in extract_text_deltas(stream.create_iterator()):
    ...         print(text_delta, end="", flush=True)
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable
from typing import cast, override

# Event type constants
EVENT_TYPE_OUTPUT_TEXT_DELTA = "response.output_text.delta"
EVENT_TYPE_OUTPUT_ITEM_ADDED = "response.output_item.added"
EVENT_TYPE_OUTPUT_ITEM_DONE = "response.output_item.done"
EVENT_TYPE_FUNCTION_CALL_ARGS_DELTA = "response.function_call_arguments.delta"
EVENT_TYPE_FUNCTION_CALL_ARGS_DONE = "response.function_call_arguments.done"


async def extract_text_deltas(
    stream: AsyncIterator[dict[str, object]],
) -> AsyncIterator[str]:
    """Extract text deltas from response stream events.

    Filters stream events to yield only text content deltas, which can be
    used to stream text output to users in real-time.

    Args:
        stream: Source async iterator of stream events

    Yields:
        str: Text delta strings from response.output_text.delta events

    Example:
        >>> async for delta in extract_text_deltas(stream.create_iterator()):
        ...     print(delta, end="", flush=True)
    """
    async for event in stream:
        if not isinstance(event, dict):
            continue

        event_type = event.get("type")
        if event_type == EVENT_TYPE_OUTPUT_TEXT_DELTA:
            delta = event.get("delta")
            if delta and isinstance(delta, str):
                yield delta


async def extract_tool_events(
    stream: AsyncIterator[dict[str, object]],
) -> AsyncIterator[dict[str, object]]:
    """Extract tool-related events from response stream.

    Filters stream events to yield only tool/function call related events,
    useful for tracking tool execution progress.

    Args:
        stream: Source async iterator of stream events

    Yields:
        dict[str, Any]: Tool-related events (function_call_arguments.delta,
                       function_call_arguments.done, etc.)

    Example:
        >>> async for tool_event in extract_tool_events(stream.create_iterator()):
        ...     if tool_event["type"] == "response.function_call_arguments.done":
        ...         print(f"Tool {tool_event['name']} called")
    """
    async for event in stream:
        if not isinstance(event, dict):
            continue

        event_type = event.get("type")
        if not isinstance(event_type, str):
            continue

        if event_type.startswith("response.function_call") or event_type.startswith(
            "response.output_item"
        ):
            # Check if it's actually a function_call item
            if event_type == EVENT_TYPE_OUTPUT_ITEM_ADDED:
                item: object = event.get("item", {})
                if isinstance(item, dict) and item.get("type") == "function_call":
                    yield event
            elif event_type == EVENT_TYPE_OUTPUT_ITEM_DONE:
                item2: object = event.get("item", {})
                if isinstance(item2, dict) and item2.get("type") == "function_call":
                    yield event
            elif event_type.startswith("response.function_call"):
                yield event


async def build_message_from_stream(
    stream: AsyncIterator[dict[str, object]],
) -> dict[str, object]:
    """Aggregate stream events into a complete assistant message.

    Consumes all events from the stream and builds a complete message object
    with accumulated text content. This is useful when you need the full
    response but are consuming a stream.

    Args:
        stream: Source async iterator of stream events

    Returns:
        dict[str, Any]: Complete assistant message with role and content

    Example:
        >>> message = await build_message_from_stream(stream.create_iterator())
        >>> print(message["content"])
    """
    content_parts: list[str] = []
    has_started = False

    async for event in stream:
        if not isinstance(event, dict):
            continue

        event_type = event.get("type")

        if event_type == EVENT_TYPE_OUTPUT_ITEM_ADDED:
            item: object = event.get("item", {})
            if isinstance(item, dict) and item.get("type") == "message":
                has_started = True

        elif event_type == EVENT_TYPE_OUTPUT_TEXT_DELTA:
            if has_started:
                delta = event.get("delta")
                if delta and isinstance(delta, str):
                    content_parts.append(delta)

        elif event_type == EVENT_TYPE_OUTPUT_ITEM_DONE:
            item3: object = event.get("item", {})
            if isinstance(item3, dict) and item3.get("type") == "message":
                # Extract complete text from the done event
                content: object = item3.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            text: object = part.get("text")
                            if isinstance(text, str):
                                # Use the complete text from the done event
                                return {
                                    "role": "assistant",
                                    "content": text,
                                }

    # If we didn't get a complete message from done event, build from deltas
    return {
        "role": "assistant",
        "content": "".join(content_parts),
    }


class StreamTransformer:
    """Base class for stream transformers with context manager support.

    This provides a base for creating stream transformers that properly
    handle resource cleanup. Subclasses should implement __aiter__ to
    define their transformation logic.

    Attributes:
        _source: Source async iterator to transform
    """

    _source: AsyncIterator[dict[str, object]]

    def __init__(self, source: AsyncIterator[dict[str, object]]):
        """Initialize transformer with source stream.

        Args:
            source: Source async iterator to transform
        """
        self._source = source

    async def __aenter__(self) -> "StreamTransformer":
        """Enter async context manager.

        Returns:
            Self for use in context
        """
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit async context manager with cleanup.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        # Close source if it has aclose
        # AsyncIterator protocol doesn't define aclose, but many implementations have it
        if hasattr(self._source, "aclose"):
            aclose_method = cast(object, getattr(self._source, "aclose"))
            if callable(aclose_method):
                _ = await cast(Awaitable[object], aclose_method())

    def __aiter__(self) -> AsyncIterator[object]:
        """Return async iterator.

        Subclasses should override this to implement transformation logic.

        Returns:
            AsyncIterator yielding transformed events
        """
        raise NotImplementedError("Subclasses must implement __aiter__")


class TextDeltaTransformer(StreamTransformer):
    """Transform stream to extract only text deltas.

    This transformer filters events to yield only text content deltas,
    making it easy to stream text output in real-time.

    Example:
        >>> async with TextDeltaTransformer(stream.create_iterator()) as transformer:
        ...     async for delta in transformer:
        ...         print(delta, end="", flush=True)
    """

    @override
    async def __aiter__(self) -> AsyncIterator[str]:
        """Iterate over text deltas from the stream.

        Yields:
            str: Text delta strings
        """
        async for delta in extract_text_deltas(self._source):
            yield delta


class ToolEventTransformer(StreamTransformer):
    """Transform stream to extract only tool-related events.

    This transformer filters events to yield only tool/function call
    related events, useful for tracking tool execution.

    Example:
        >>> async with ToolEventTransformer(stream.create_iterator()) as transformer:
        ...     async for event in transformer:
        ...         print(f"Tool event: {event['type']}")
    """

    @override
    async def __aiter__(self) -> AsyncIterator[dict[str, object]]:
        """Iterate over tool events from the stream.

        Yields:
            dict[str, Any]: Tool-related events
        """
        async for event in extract_tool_events(self._source):
            yield event


class MessageAggregator(StreamTransformer):
    """Aggregate stream events into a complete message.

    This transformer consumes the entire stream and returns a complete
    assistant message. Note: This is not a true transformer as it returns
    a single value rather than an iterator.

    Example:
        >>> async with MessageAggregator(stream.create_iterator()) as aggregator:
        ...     message = await aggregator.get_message()
        ...     print(message["content"])
    """

    async def get_message(self) -> dict[str, object]:
        """Get the complete message from the stream.

        Returns:
            dict[str, Any]: Complete assistant message
        """
        return await build_message_from_stream(self._source)

    @override
    async def __aiter__(self) -> AsyncIterator[dict[str, object]]:
        """Not typically used for aggregator, but yields the final message.

        Yields:
            dict[str, Any]: The complete message (single yield)
        """
        message = await self.get_message()
        yield message
