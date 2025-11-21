"""Tests for stream transformers.

This module tests the stream transformation utilities that filter and
aggregate SSE events from API responses.
"""

from __future__ import annotations

from __future__ import annotations

from typing import Any

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore[assignment]

from openrouter.call_model.stream_transformers import (
    TextDeltaTransformer,
    ToolEventTransformer,
    MessageAggregator,
    extract_text_deltas,
    extract_tool_events,
    build_message_from_stream,
)


# Test fixtures for mock events


class AsyncListIterator:
    """Helper class to convert a list to an async iterator."""

    def __init__(self, items: list):
        self._items = items
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


@pytest.fixture
def text_delta_events():
    """Sample events with text deltas."""
    return [
        {
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "msg_1"},
        },
        {
            "type": "response.output_text.delta",
            "delta": "Hello",
        },
        {
            "type": "response.output_text.delta",
            "delta": " world",
        },
        {
            "type": "response.output_text.delta",
            "delta": "!",
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": "msg_1",
                "content": [{"type": "output_text", "text": "Hello world!"}],
            },
        },
    ]


@pytest.fixture
def tool_call_events():
    """Sample events with tool calls."""
    return [
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "callId": "call_1",
                "name": "get_weather",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "itemId": "call_1",
            "delta": '{"location"',
        },
        {
            "type": "response.function_call_arguments.delta",
            "itemId": "call_1",
            "delta": ': "Paris"}',
        },
        {
            "type": "response.function_call_arguments.done",
            "itemId": "call_1",
            "name": "get_weather",
            "arguments": '{"location": "Paris"}',
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "callId": "call_1",
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
            },
        },
    ]


@pytest.fixture
def mixed_events():
    """Sample events with both text and tool calls."""
    return [
        {
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "msg_1"},
        },
        {
            "type": "response.output_text.delta",
            "delta": "Let me check ",
        },
        {
            "type": "response.output_text.delta",
            "delta": "the weather.",
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": "msg_1",
                "content": [
                    {"type": "output_text", "text": "Let me check the weather."}
                ],
            },
        },
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "callId": "call_1",
                "name": "get_weather",
            },
        },
        {
            "type": "response.function_call_arguments.done",
            "itemId": "call_1",
            "name": "get_weather",
            "arguments": '{"location": "Paris"}',
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "callId": "call_1",
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
            },
        },
    ]


# Tests for extract_text_deltas


@pytest.mark.asyncio
async def test_extract_text_deltas_basic(text_delta_events):
    """Test extracting text deltas from stream."""
    stream = AsyncListIterator(text_delta_events)
    deltas = []

    async for delta in extract_text_deltas(stream):
        deltas.append(delta)

    assert deltas == ["Hello", " world", "!"]
    assert "".join(deltas) == "Hello world!"


@pytest.mark.asyncio
async def test_extract_text_deltas_empty_stream():
    """Test extracting from empty stream."""
    stream = AsyncListIterator([])
    deltas = []

    async for delta in extract_text_deltas(stream):
        deltas.append(delta)

    assert deltas == []


@pytest.mark.asyncio
async def test_extract_text_deltas_no_text_events():
    """Test extracting when no text delta events present."""
    events = [
        {"type": "response.output_item.added", "item": {"type": "message"}},
        {"type": "response.completed"},
    ]
    stream = AsyncListIterator(events)
    deltas = []

    async for delta in extract_text_deltas(stream):
        deltas.append(delta)

    assert deltas == []


@pytest.mark.asyncio
async def test_extract_text_deltas_filters_non_text():
    """Test that only text deltas are extracted."""
    events = [
        {"type": "response.output_text.delta", "delta": "Hello"},
        {"type": "response.function_call_arguments.delta", "delta": '{"key"}'},
        {"type": "response.output_text.delta", "delta": " world"},
        {"type": "response.completed"},
    ]
    stream = AsyncListIterator(events)
    deltas = []

    async for delta in extract_text_deltas(stream):
        deltas.append(delta)

    assert deltas == ["Hello", " world"]


# Tests for extract_tool_events


@pytest.mark.asyncio
async def test_extract_tool_events_basic(tool_call_events):
    """Test extracting tool events from stream."""
    stream = AsyncListIterator(tool_call_events)
    events = []

    async for event in extract_tool_events(stream):
        events.append(event)

    # Should get: added, delta x2, done, item.done
    assert len(events) == 5
    assert events[0]["type"] == "response.output_item.added"
    assert events[1]["type"] == "response.function_call_arguments.delta"
    assert events[2]["type"] == "response.function_call_arguments.delta"
    assert events[3]["type"] == "response.function_call_arguments.done"
    assert events[4]["type"] == "response.output_item.done"


@pytest.mark.asyncio
async def test_extract_tool_events_filters_text():
    """Test that text events are filtered out."""
    events = [
        {"type": "response.output_text.delta", "delta": "Hello"},
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "callId": "1"},
        },
        {"type": "response.output_text.delta", "delta": " world"},
        {
            "type": "response.function_call_arguments.done",
            "itemId": "1",
            "name": "tool",
            "arguments": "{}",
        },
    ]
    stream = AsyncListIterator(events)
    tool_events = []

    async for event in extract_tool_events(stream):
        tool_events.append(event)

    assert len(tool_events) == 2
    assert all(
        "function_call" in e["type"]
        or (
            e["type"] == "response.output_item.added"
            and e.get("item", {}).get("type") == "function_call"
        )
        for e in tool_events
    )


@pytest.mark.asyncio
async def test_extract_tool_events_filters_message_items():
    """Test that message items are filtered out, only function_call items included."""
    events = [
        {
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "msg_1"},
        },
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "callId": "call_1"},
        },
        {
            "type": "response.output_item.done",
            "item": {"type": "message", "id": "msg_1"},
        },
        {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "callId": "call_1"},
        },
    ]
    stream = AsyncListIterator(events)
    tool_events = []

    async for event in extract_tool_events(stream):
        tool_events.append(event)

    # Should only get the function_call items
    assert len(tool_events) == 2
    assert all(e.get("item", {}).get("type") == "function_call" for e in tool_events)


# Tests for build_message_from_stream


@pytest.mark.asyncio
async def test_build_message_from_stream_basic(text_delta_events):
    """Test building complete message from stream."""
    stream = AsyncListIterator(text_delta_events)
    message = await build_message_from_stream(stream)

    assert message["role"] == "assistant"
    assert message["content"] == "Hello world!"


@pytest.mark.asyncio
async def test_build_message_from_stream_uses_done_event():
    """Test that complete text from done event is used."""
    events = [
        {
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "msg_1"},
        },
        # These deltas might be partial
        {"type": "response.output_text.delta", "delta": "Hel"},
        {"type": "response.output_text.delta", "delta": "lo"},
        # Done event has complete text
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": "msg_1",
                "content": [{"type": "output_text", "text": "Hello world!"}],
            },
        },
    ]
    stream = AsyncListIterator(events)
    message = await build_message_from_stream(stream)

    # Should use complete text from done event, not concatenated deltas
    assert message["content"] == "Hello world!"


@pytest.mark.asyncio
async def test_build_message_from_stream_empty():
    """Test building message from empty stream."""
    stream = AsyncListIterator([])
    message = await build_message_from_stream(stream)

    assert message["role"] == "assistant"
    assert message["content"] == ""


@pytest.mark.asyncio
async def test_build_message_from_stream_no_done_event():
    """Test building message when no done event (uses deltas)."""
    events = [
        {
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "msg_1"},
        },
        {"type": "response.output_text.delta", "delta": "Hello"},
        {"type": "response.output_text.delta", "delta": " from"},
        {"type": "response.output_text.delta", "delta": " deltas"},
    ]
    stream = AsyncListIterator(events)
    message = await build_message_from_stream(stream)

    assert message["role"] == "assistant"
    assert message["content"] == "Hello from deltas"


# Tests for TextDeltaTransformer class


@pytest.mark.asyncio
async def test_text_delta_transformer_basic(text_delta_events):
    """Test TextDeltaTransformer class."""
    stream = AsyncListIterator(text_delta_events)
    transformer = TextDeltaTransformer(stream)

    deltas = []
    async for delta in transformer:
        deltas.append(delta)

    assert deltas == ["Hello", " world", "!"]


@pytest.mark.asyncio
async def test_text_delta_transformer_context_manager(text_delta_events):
    """Test TextDeltaTransformer as context manager."""
    stream = AsyncListIterator(text_delta_events)

    deltas = []
    async with TextDeltaTransformer(stream) as transformer:
        async for delta in transformer:
            deltas.append(delta)

    assert deltas == ["Hello", " world", "!"]


# Tests for ToolEventTransformer class


@pytest.mark.asyncio
async def test_tool_event_transformer_basic(tool_call_events):
    """Test ToolEventTransformer class."""
    stream = AsyncListIterator(tool_call_events)
    transformer = ToolEventTransformer(stream)

    events = []
    async for event in transformer:
        events.append(event)

    assert len(events) == 5
    assert events[0]["type"] == "response.output_item.added"


@pytest.mark.asyncio
async def test_tool_event_transformer_context_manager(tool_call_events):
    """Test ToolEventTransformer as context manager."""
    stream = AsyncListIterator(tool_call_events)

    events = []
    async with ToolEventTransformer(stream) as transformer:
        async for event in transformer:
            events.append(event)

    assert len(events) == 5


# Tests for MessageAggregator class


@pytest.mark.asyncio
async def test_message_aggregator_get_message(text_delta_events):
    """Test MessageAggregator.get_message()."""
    stream = AsyncListIterator(text_delta_events)
    aggregator = MessageAggregator(stream)

    message = await aggregator.get_message()

    assert message["role"] == "assistant"
    assert message["content"] == "Hello world!"


@pytest.mark.asyncio
async def test_message_aggregator_iterator(text_delta_events):
    """Test MessageAggregator as async iterator."""
    stream = AsyncListIterator(text_delta_events)
    aggregator = MessageAggregator(stream)

    messages = []
    async for message in aggregator:
        messages.append(message)

    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == "Hello world!"


@pytest.mark.asyncio
async def test_message_aggregator_context_manager(text_delta_events):
    """Test MessageAggregator as context manager."""
    stream = AsyncListIterator(text_delta_events)

    async with MessageAggregator(stream) as aggregator:
        message = await aggregator.get_message()

    assert message["role"] == "assistant"
    assert message["content"] == "Hello world!"


# Integration tests with mixed events


@pytest.mark.asyncio
async def test_mixed_stream_text_extraction(mixed_events):
    """Test extracting text from stream with both text and tool events."""
    stream = AsyncListIterator(mixed_events)
    deltas = []

    async for delta in extract_text_deltas(stream):
        deltas.append(delta)

    assert deltas == ["Let me check ", "the weather."]
    assert "".join(deltas) == "Let me check the weather."


@pytest.mark.asyncio
async def test_mixed_stream_tool_extraction(mixed_events):
    """Test extracting tools from stream with both text and tool events."""
    stream = AsyncListIterator(mixed_events)
    tool_events = []

    async for event in extract_tool_events(stream):
        tool_events.append(event)

    # Should get: function_call added, args.done, item.done
    assert len(tool_events) == 3
    tool_types = [e["type"] for e in tool_events]
    assert "response.output_item.added" in tool_types
    assert "response.function_call_arguments.done" in tool_types
    assert "response.output_item.done" in tool_types


@pytest.mark.asyncio
async def test_multiple_transformers_on_same_source():
    """Test that we can create multiple transformers from same source iterator.

    Note: Each transformer will consume the iterator, so this test uses
    separate iterators. In practice, you'd use ReusableStream for this.
    """
    events = [
        {"type": "response.output_text.delta", "delta": "Hello"},
        {"type": "response.output_text.delta", "delta": " world"},
    ]

    # First transformer
    stream1 = AsyncListIterator(events.copy())
    deltas1 = []
    async for delta in extract_text_deltas(stream1):
        deltas1.append(delta)

    # Second transformer on fresh iterator
    stream2 = AsyncListIterator(events.copy())
    deltas2 = []
    async for delta in extract_text_deltas(stream2):
        deltas2.append(delta)

    assert deltas1 == deltas2 == ["Hello", " world"]
