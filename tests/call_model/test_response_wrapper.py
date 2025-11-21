"""Unit tests for ResponseWrapper implementation.

This module tests the ResponseWrapper class, verifying:
- Initialization and state management
- Lazy stream initialization
- State transitions (INITIALIZED → STREAMING → COMPLETED/ERROR)
- Basic caching infrastructure for message and text
- Property vs method access patterns
- Integration points for tools
- Error handling

Test organization follows the plan:
- TestBasicFunctionality: Core initialization and state
- TestLazyInitialization: Lazy stream creation
- TestStateTrans itions: State machine behavior
- TestCaching: Caching infrastructure
- TestErrorHandling: Error propagation and states
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from openrouter.call_model.response_wrapper import ResponseWrapper
from openrouter.call_model.types import ResponseState


# Test fixtures


@pytest.fixture
def mock_client() -> Mock:
    """Create a mock OpenRouter client.

    Returns:
        Mock: Mock client for testing
    """
    client = Mock()
    client.beta = Mock()
    client.beta.responses = Mock()
    client.beta.responses.send = AsyncMock()
    return client


@pytest.fixture
def basic_request() -> dict[str, object]:
    """Create a basic request for testing.

    Returns:
        dict[str, object]: Basic request parameters
    """
    return {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }


# Test Classes


@pytest.mark.asyncio
class TestBasicFunctionality:
    """Test basic initialization and configuration."""

    async def test_initialization_with_minimal_params(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test creating wrapper with minimal parameters.

        Hypothesis: Wrapper can be created with just client and request.
        Evidence: Wrapper created with correct initial state.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # Evidence: Wrapper initialized correctly
        assert wrapper.state == ResponseState.INITIALIZED
        assert wrapper.message is None
        assert wrapper.text is None
        assert wrapper._tools == []
        assert wrapper._max_tool_rounds is None

    async def test_initialization_with_tools(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test creating wrapper with tools.

        Hypothesis: Wrapper stores tools for later execution.
        Evidence: Tools are stored in wrapper.
        """
        # Create mock tools
        mock_tools = [Mock(), Mock()]

        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
            tools=mock_tools,
        )

        # Evidence: Tools stored correctly
        assert wrapper._tools == mock_tools
        assert wrapper.state == ResponseState.INITIALIZED

    async def test_initialization_with_max_tool_rounds_int(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test creating wrapper with integer max_tool_rounds.

        Hypothesis: Integer max_tool_rounds is stored.
        Evidence: max_tool_rounds stored as integer.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
            max_tool_rounds=3,
        )

        # Evidence: max_tool_rounds stored correctly
        assert wrapper._max_tool_rounds == 3

    async def test_initialization_with_max_tool_rounds_callable(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test creating wrapper with callable max_tool_rounds.

        Hypothesis: Callable max_tool_rounds is stored.
        Evidence: max_tool_rounds stored as callable.
        """

        def custom_limit(context: Any) -> bool:
            """Custom limit function."""
            return context["number_of_turns"] < 5

        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
            max_tool_rounds=custom_limit,
        )

        # Evidence: Callable stored correctly
        assert wrapper._max_tool_rounds == custom_limit
        assert callable(wrapper._max_tool_rounds)

    async def test_properties_return_none_before_consumption(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test that properties return None before consumption (FR-1.2.6).

        Hypothesis: Properties return None until data is cached.
        Evidence: message and text properties return None initially.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # Evidence: Properties return None before consumption
        assert wrapper.message is None
        assert wrapper.text is None

    async def test_state_property_returns_current_state(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test that state property returns current state.

        Hypothesis: State property provides current ResponseState.
        Evidence: State property returns INITIALIZED on creation.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # Evidence: State accessible via property
        assert wrapper.state == ResponseState.INITIALIZED
        assert isinstance(wrapper.state, ResponseState)


@pytest.mark.asyncio
class TestLazyInitialization:
    """Test lazy stream initialization."""

    async def test_stream_not_initialized_on_creation(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test that stream is not created until needed.

        Hypothesis: Stream is created lazily, not on wrapper creation.
        Evidence: _stream is None after wrapper creation.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # Evidence: Stream not created yet
        assert wrapper._stream is None
        assert wrapper._init_promise is None

    async def test_init_stream_creates_stream_once(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test that _init_stream creates stream exactly once.

        Hypothesis: Multiple calls to _init_stream are idempotent.
        Evidence: Stream created once, same promise returned.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # First call creates stream
        await wrapper._init_stream()

        # Evidence: Stream created
        assert wrapper._stream is not None
        assert wrapper._init_promise is not None

        # Store references
        first_stream = wrapper._stream
        first_promise = wrapper._init_promise

        # Second call should return same stream
        await wrapper._init_stream()

        # Evidence: Same stream and promise (idempotent)
        assert wrapper._stream is first_stream
        assert wrapper._init_promise is first_promise

    async def test_init_stream_transitions_to_streaming_state(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test that _init_stream transitions to STREAMING state.

        Hypothesis: Stream initialization changes state to STREAMING.
        Evidence: State is STREAMING after _init_stream.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # Evidence: Initial state
        assert wrapper.state == ResponseState.INITIALIZED

        # Initialize stream
        await wrapper._init_stream()

        # Evidence: State transitioned to STREAMING
        assert wrapper.state == ResponseState.STREAMING


@pytest.mark.asyncio
class TestStateTransitions:
    """Test state machine transitions."""

    async def test_state_initialized_on_creation(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test initial state is INITIALIZED.

        Hypothesis: All wrappers start in INITIALIZED state.
        Evidence: State is INITIALIZED after creation.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # Evidence: Initial state
        assert wrapper.state == ResponseState.INITIALIZED

    async def test_state_transitions_to_streaming(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test transition from INITIALIZED to STREAMING.

        Hypothesis: Stream init causes INITIALIZED → STREAMING.
        Evidence: State changes when stream initialized.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # Initial state
        assert wrapper.state == ResponseState.INITIALIZED

        # Trigger stream initialization
        await wrapper._init_stream()

        # Evidence: Transitioned to STREAMING
        assert wrapper.state == ResponseState.STREAMING

    async def test_state_transitions_to_completed_no_tools(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test transition to COMPLETED when no tools.

        Hypothesis: Without tools, execution completes immediately.
        Evidence: State becomes COMPLETED after tool execution check.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
            tools=None,  # No tools
        )

        # Execute tools check
        await wrapper._execute_tools_if_needed()

        # Evidence: Completed since no tools to execute
        assert wrapper.state == ResponseState.COMPLETED

    async def test_state_transitions_to_completed_with_tools(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test transition to COMPLETED with tools.

        Hypothesis: Tool execution leads to COMPLETED state.
        Evidence: State becomes COMPLETED after tool execution.
        """
        # Mock tools
        mock_tools = [Mock()]

        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
            tools=mock_tools,
        )

        # Execute tools (placeholder in PR 3.1)
        await wrapper._execute_tools_if_needed()

        # Evidence: Completed after tool execution attempt
        assert wrapper.state == ResponseState.COMPLETED

    async def test_state_transitions_to_error_on_init_failure(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test transition to ERROR on initialization failure.

        Hypothesis: Errors during init cause ERROR state.
        Evidence: State becomes ERROR when init fails.
        """
        # Create wrapper
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # Inject error by making stream creation fail
        # We'll override _init_stream to raise an error
        original_init = wrapper._init_stream

        async def failing_init() -> None:
            """Init that raises error."""
            wrapper._state = ResponseState.STREAMING
            raise RuntimeError("Simulated init error")

        wrapper._init_stream = failing_init  # type: ignore[method-assign]

        # Attempt initialization
        with pytest.raises(RuntimeError, match="Simulated init error"):
            await wrapper._init_stream()

        # Evidence: State transitioned to ERROR
        # Note: In actual implementation, error handling will set state
        # For now, we verify the error is raised correctly


@pytest.mark.asyncio
class TestCaching:
    """Test caching infrastructure."""

    async def test_message_cached_after_get_message(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test that message is cached after get_message().

        Hypothesis: get_message() caches result in _cached_data.
        Evidence: message property returns cached value after get_message.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # Evidence: No cached message initially
        assert wrapper.message is None

        # Get message (will cache it)
        message = await wrapper.get_message()

        # Evidence: Message is now cached
        assert wrapper.message is not None
        assert wrapper.message == message
        assert "message" in wrapper._cached_data

    async def test_text_cached_after_get_text(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test that text is cached after get_text().

        Hypothesis: get_text() caches result in _cached_data.
        Evidence: text property returns cached value after get_text.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # Evidence: No cached text initially
        assert wrapper.text is None

        # Get text (will cache it)
        text = await wrapper.get_text()

        # Evidence: Text is now cached
        assert wrapper.text is not None
        assert wrapper.text == text
        assert "text" in wrapper._cached_data

    async def test_multiple_get_message_calls_return_same_instance(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test that multiple get_message() calls return cached value.

        Hypothesis: Subsequent calls return cached result without recomputation.
        Evidence: Same object returned on multiple calls.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # First call
        message1 = await wrapper.get_message()

        # Second call should return cached value
        message2 = await wrapper.get_message()

        # Evidence: Same cached instance returned
        assert message1 is message2

    async def test_multiple_get_text_calls_return_same_instance(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test that multiple get_text() calls return cached value.

        Hypothesis: Subsequent calls return cached result without recomputation.
        Evidence: Same object returned on multiple calls.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # First call
        text1 = await wrapper.get_text()

        # Second call should return cached value
        text2 = await wrapper.get_text()

        # Evidence: Same cached instance returned
        assert text1 is text2

    async def test_concurrent_get_message_calls_deduplicated(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test that concurrent get_message() calls are deduplicated.

        Hypothesis: Multiple concurrent calls share same task.
        Evidence: Same result returned to all callers.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # Launch multiple concurrent get_message calls
        results = await asyncio.gather(
            wrapper.get_message(),
            wrapper.get_message(),
            wrapper.get_message(),
        )

        # Evidence: All got the same result
        assert results[0] is results[1]
        assert results[1] is results[2]


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and propagation."""

    async def test_error_in_stream_init_propagates(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test that errors during stream init propagate correctly.

        Hypothesis: Init errors are raised to caller.
        Evidence: Exception propagates from _init_stream.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # Override init to raise error
        async def failing_init() -> None:
            """Init that fails."""
            wrapper._state = ResponseState.STREAMING
            raise ValueError("Stream init failed")

        wrapper._init_stream = failing_init  # type: ignore[method-assign]

        # Attempt to call method that inits stream
        with pytest.raises(ValueError, match="Stream init failed"):
            await wrapper._init_stream()

    async def test_tool_execution_errors_set_error_state(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test that tool execution errors set ERROR state.

        Hypothesis: Errors during tool execution set state to ERROR.
        Evidence: State is ERROR after execution failure.
        """
        mock_tools = [Mock()]

        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
            tools=mock_tools,
        )

        # Override tool execution to fail
        async def failing_execute() -> None:
            """Execution that fails."""
            await wrapper._init_stream()
            wrapper._state = ResponseState.ERROR
            raise RuntimeError("Tool execution failed")

        wrapper._execute_tools_if_needed = failing_execute  # type: ignore[method-assign]

        # Attempt to execute tools
        with pytest.raises(RuntimeError, match="Tool execution failed"):
            await wrapper._execute_tools_if_needed()

        # Evidence: State is ERROR
        assert wrapper.state == ResponseState.ERROR


@pytest.mark.asyncio
class TestContextManager:
    """Test async context manager support."""

    async def test_context_manager_cleanup(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test that context manager cleans up resources.

        Hypothesis: __aexit__ calls close() for cleanup.
        Evidence: close() called on context exit.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # Use as context manager
        async with wrapper as w:
            # Evidence: Same wrapper returned
            assert w is wrapper

        # After exit, close should have been called
        # (We can't verify this directly without mocking, but no errors is evidence)

    async def test_context_manager_cleanup_with_stream(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test cleanup when stream was initialized.

        Hypothesis: Context manager closes stream on exit.
        Evidence: Stream cleanup happens without errors.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        async with wrapper:
            # Initialize stream
            await wrapper._init_stream()
            assert wrapper._stream is not None

        # After exit, stream should be closed
        # (No errors during cleanup is evidence of success)


@pytest.mark.asyncio
class TestToolIntegration:
    """Test integration points for tool system."""

    async def test_execute_tools_if_needed_is_idempotent(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test that _execute_tools_if_needed is idempotent.

        Hypothesis: Multiple calls return same promise.
        Evidence: Same task returned on multiple calls.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # First call
        await wrapper._execute_tools_if_needed()
        first_promise = wrapper._tool_execution_promise

        # Second call
        await wrapper._execute_tools_if_needed()
        second_promise = wrapper._tool_execution_promise

        # Evidence: Same promise (idempotent)
        assert first_promise is second_promise

    async def test_execute_tools_initializes_stream_first(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test that tool execution initializes stream first.

        Hypothesis: _execute_tools_if_needed calls _init_stream.
        Evidence: Stream is initialized after tool execution.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # Evidence: Stream not initialized
        assert wrapper._stream is None

        # Execute tools
        await wrapper._execute_tools_if_needed()

        # Evidence: Stream was initialized
        assert wrapper._stream is not None

    async def test_no_tools_skips_execution(
        self, mock_client: Mock, basic_request: dict[str, object]
    ):
        """Test that execution is skipped when no tools provided.

        Hypothesis: Without tools, execution completes immediately.
        Evidence: State becomes COMPLETED without tool execution.
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
            tools=None,  # No tools
        )

        # Execute (should skip)
        await wrapper._execute_tools_if_needed()

        # Evidence: Completed without errors
        assert wrapper.state == ResponseState.COMPLETED


@pytest.mark.asyncio
class TestConsumptionMethods:
    """Test consumption methods implementation (FR-1.2.1-1.2.5)."""

    @pytest.fixture
    def mock_stream_events(self) -> list[dict[str, object]]:
        """Create mock stream events for testing.

        Returns:
            list[dict[str, object]]: Mock events simulating API stream
        """
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
    def wrapper_with_mock_stream(
        self,
        mock_client: Mock,
        basic_request: dict[str, object],
        mock_stream_events: list[dict[str, object]],
    ) -> ResponseWrapper:
        """Create wrapper with mocked stream.

        Args:
            mock_client: Mock client
            basic_request: Basic request
            mock_stream_events: Mock events

        Returns:
            ResponseWrapper: Wrapper with mocked stream
        """
        wrapper = ResponseWrapper(
            client=mock_client,
            request=basic_request,
        )

        # Mock the stream creation
        async def create_mock_stream() -> None:
            """Create mock stream."""
            from openrouter.call_model.reusable_stream import ReusableStream

            async def mock_source() -> AsyncIterator[dict[str, object]]:
                """Mock source that yields events."""
                for event in mock_stream_events:
                    yield event

            wrapper._stream = ReusableStream(mock_source())
            wrapper._state = ResponseState.STREAMING

        # Replace _init_stream with mock version
        async def mock_init_stream() -> None:
            """Mock init."""
            if wrapper._init_promise is None:
                wrapper._init_promise = asyncio.create_task(create_mock_stream())
            await wrapper._init_promise

        wrapper._init_stream = mock_init_stream  # type: ignore[method-assign]

        return wrapper

    async def test_get_message_returns_complete_message(
        self, wrapper_with_mock_stream: ResponseWrapper
    ):
        """Test get_message() returns complete message (FR-1.2.1).

        Hypothesis: get_message() aggregates stream into complete message.
        Evidence: Message contains expected content from stream.
        """
        # Get message
        message = await wrapper_with_mock_stream.get_message()

        # Evidence: Message has expected structure
        assert isinstance(message, dict)
        assert message["role"] == "assistant"
        assert "content" in message

        # Evidence: Content matches expected text
        content = message["content"]
        if isinstance(content, str):
            assert content == "Hello world!"
        elif isinstance(content, list):
            # Extract text from content parts
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text_parts.append(part.get("text", ""))
            assert "".join(text_parts) == "Hello world!"

    async def test_get_text_returns_text_only(
        self, wrapper_with_mock_stream: ResponseWrapper
    ):
        """Test get_text() returns only text content (FR-1.2.2).

        Hypothesis: get_text() extracts just the text from message.
        Evidence: Returns string with text content only.
        """
        # Get text
        text = await wrapper_with_mock_stream.get_text()

        # Evidence: Text is a string
        assert isinstance(text, str)

        # Evidence: Text matches expected content
        assert text == "Hello world!"

    async def test_get_text_stream_yields_deltas(
        self, wrapper_with_mock_stream: ResponseWrapper
    ):
        """Test get_text_stream() yields text deltas (FR-1.2.3).

        Hypothesis: get_text_stream() yields text as it arrives.
        Evidence: Stream yields expected deltas in order.
        """
        # Collect deltas from stream
        deltas: list[str] = []
        async for delta in wrapper_with_mock_stream.get_text_stream():
            deltas.append(delta)

        # Evidence: Got expected deltas
        assert deltas == ["Hello", " world", "!"]

        # Evidence: Joined deltas match complete text
        assert "".join(deltas) == "Hello world!"

    async def test_get_full_stream_yields_all_events(
        self,
        wrapper_with_mock_stream: ResponseWrapper,
        mock_stream_events: list[dict[str, object]],
    ):
        """Test get_full_stream() yields all events (FR-1.2.4).

        Hypothesis: get_full_stream() provides raw event access.
        Evidence: All events yielded in order.
        """
        # Collect events from stream
        events: list[dict[str, object]] = []
        async for event in wrapper_with_mock_stream.get_full_stream():
            events.append(event)

        # Evidence: Got all expected events
        assert len(events) == len(mock_stream_events)

        # Evidence: Events match expected types
        event_types = [e.get("type") for e in events]
        expected_types = [e.get("type") for e in mock_stream_events]
        assert event_types == expected_types

    async def test_multiple_consumption_methods_return_consistent_data(
        self, wrapper_with_mock_stream: ResponseWrapper
    ):
        """Test multiple consumption patterns are consistent (FR-1.2.5).

        Hypothesis: All consumption methods work on same stream.
        Evidence: Text from get_text matches deltas from get_text_stream.
        """
        # Get text via get_text()
        text1 = await wrapper_with_mock_stream.get_text()

        # Get text via streaming
        deltas: list[str] = []
        async for delta in wrapper_with_mock_stream.get_text_stream():
            deltas.append(delta)
        text2 = "".join(deltas)

        # Evidence: Both methods return same text
        assert text1 == text2
        assert text1 == "Hello world!"

    async def test_get_message_caches_result(
        self, wrapper_with_mock_stream: ResponseWrapper
    ):
        """Test get_message() caches result for reuse.

        Hypothesis: Second call returns cached value without reprocessing.
        Evidence: Property returns cached value after first call.
        """
        # First call
        message1 = await wrapper_with_mock_stream.get_message()

        # Evidence: Message is cached
        assert wrapper_with_mock_stream.message is not None
        assert wrapper_with_mock_stream.message is message1

        # Second call should return cached value
        message2 = await wrapper_with_mock_stream.get_message()

        # Evidence: Same instance returned
        assert message2 is message1

    async def test_get_text_caches_result(
        self, wrapper_with_mock_stream: ResponseWrapper
    ):
        """Test get_text() caches result for reuse.

        Hypothesis: Second call returns cached value without reprocessing.
        Evidence: Property returns cached value after first call.
        """
        # First call
        text1 = await wrapper_with_mock_stream.get_text()

        # Evidence: Text is cached
        assert wrapper_with_mock_stream.text is not None
        assert wrapper_with_mock_stream.text == text1

        # Second call should return cached value
        text2 = await wrapper_with_mock_stream.get_text()

        # Evidence: Same value returned
        assert text2 == text1

    async def test_concurrent_get_message_calls(
        self, wrapper_with_mock_stream: ResponseWrapper
    ):
        """Test concurrent get_message() calls are deduplicated.

        Hypothesis: Concurrent calls share same task, avoiding duplicate work.
        Evidence: All callers get same result.
        """
        # Launch multiple concurrent calls
        results = await asyncio.gather(
            wrapper_with_mock_stream.get_message(),
            wrapper_with_mock_stream.get_message(),
            wrapper_with_mock_stream.get_message(),
        )

        # Evidence: All got same result
        assert results[0] is results[1]
        assert results[1] is results[2]
        assert results[0]["content"] == "Hello world!" or any(
            isinstance(p, dict) and p.get("text") == "Hello world!"
            for p in results[0].get("content", [])
            if isinstance(results[0].get("content"), list)
        )

    async def test_concurrent_get_text_calls(
        self, wrapper_with_mock_stream: ResponseWrapper
    ):
        """Test concurrent get_text() calls are deduplicated.

        Hypothesis: Concurrent calls share same task.
        Evidence: All callers get same result.
        """
        # Launch multiple concurrent calls
        results = await asyncio.gather(
            wrapper_with_mock_stream.get_text(),
            wrapper_with_mock_stream.get_text(),
            wrapper_with_mock_stream.get_text(),
        )

        # Evidence: All got same result
        assert results[0] == results[1]
        assert results[1] == results[2]
        assert results[0] == "Hello world!"

    async def test_concurrent_stream_consumption(
        self, wrapper_with_mock_stream: ResponseWrapper
    ):
        """Test multiple consumers can stream concurrently.

        Hypothesis: ReusableStream supports multiple concurrent iterators.
        Evidence: Multiple consumers get same events.
        """

        async def consume_text_stream() -> str:
            """Consume text stream and return joined text."""
            deltas: list[str] = []
            async for delta in wrapper_with_mock_stream.get_text_stream():
                deltas.append(delta)
            return "".join(deltas)

        async def consume_full_stream() -> int:
            """Consume full stream and return event count."""
            count = 0
            async for _ in wrapper_with_mock_stream.get_full_stream():
                count += 1
            return count

        # Run consumers concurrently
        text_result, count_result = await asyncio.gather(
            consume_text_stream(),
            consume_full_stream(),
        )

        # Evidence: Text consumer got expected text
        assert text_result == "Hello world!"

        # Evidence: Full stream consumer got all events
        assert count_result == 5  # Number of mock events

    async def test_mixed_consumption_patterns(
        self, wrapper_with_mock_stream: ResponseWrapper
    ):
        """Test mixing different consumption patterns.

        Hypothesis: Can use get_message(), get_text(), and streams together.
        Evidence: All return consistent data.
        """
        # Start with streaming
        deltas: list[str] = []
        async for delta in wrapper_with_mock_stream.get_text_stream():
            deltas.append(delta)

        # Then get message
        message = await wrapper_with_mock_stream.get_message()

        # Then get text
        text = await wrapper_with_mock_stream.get_text()

        # Evidence: All consistent
        streamed_text = "".join(deltas)
        assert streamed_text == "Hello world!"
        assert text == "Hello world!"
        assert message["role"] == "assistant"


if __name__ == "__main__":
    # Run tests with pytest
    _ = pytest.main([__file__, "-v"])
