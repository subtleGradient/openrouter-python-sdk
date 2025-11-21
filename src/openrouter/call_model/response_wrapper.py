"""ResponseWrapper for multiple consumption patterns.

This module implements the ResponseWrapper class that provides a high-level
interface for consuming API responses in multiple ways. It supports:
- Complete message retrieval with tool execution
- Text-only extraction
- Streaming text deltas
- Full event stream access

Key features:
- Single API call with multiple consumption patterns (FR-1.1.3, FR-1.2.5)
- Lazy stream initialization and tool execution
- Cached results to avoid redundant computation
- State tracking through response lifecycle
- Properties for cached values, methods for operations (FR-1.2.6, FR-1.2.7)

Example:
    >>> async def process_response():
    ...     wrapper = ResponseWrapper(
    ...         client=client,
    ...         request={"model": "gpt-4", "messages": [...]},
    ...         tools=[weather_tool],
    ...         max_tool_rounds=3
    ...     )
    ...
    ...     # Multiple consumption patterns
    ...     message = await wrapper.get_message()  # Full message with tools
    ...     text = await wrapper.get_text()  # Just the text
    ...
    ...     # Properties for cached data
    ...     if wrapper.message:
    ...         print("Message already cached:", wrapper.message)
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from openrouter.call_model.reusable_stream import ReusableStream
from openrouter.call_model.types import CachedData, ResponseState

if TYPE_CHECKING:
    from openrouter.call_model.tool_system import BaseTool


class ResponseWrapper:
    """Wrapper providing multiple consumption patterns for a response.

    This class wraps an API response and provides multiple ways to consume it:
    - get_message(): Get complete message with tool execution
    - get_text(): Extract text content only
    - get_text_stream(): Stream text deltas as they arrive
    - get_full_stream(): Stream all SSE events

    All consumption methods work on the same underlying response, making
    exactly one API call regardless of how many consumption methods are used.

    Attributes:
        message: Cached complete message (None if not yet consumed)
        text: Cached text content (None if not yet consumed)
        state: Current state of the response wrapper

    Example:
        >>> wrapper = ResponseWrapper(
        ...     client=client,
        ...     request={"model": "gpt-4", "messages": [...]},
        ...     tools=[weather_tool]
        ... )
        >>>
        >>> # Get just the text
        >>> text = await wrapper.get_text()
        >>> print(text)
        >>>
        >>> # Later, get full message (no additional API call)
        >>> message = await wrapper.get_message()
        >>> print(message)
    """

    def __init__(
        self,
        client: Any,  # OpenRouter client instance
        request: dict[str, object],
        tools: list[BaseTool[Any, Any]] | None = None,
        max_tool_rounds: int | Callable[[Any], bool] | None = None,
        options: Any | None = None,  # RequestOptions
    ):
        """Initialize ResponseWrapper with request parameters.

        Args:
            client: OpenRouter client instance for making API calls
            request: Request parameters (same as beta.responses.send)
            tools: Optional list of enhanced tools with execute methods
            max_tool_rounds: Max rounds (int, default: 5, max: 10) or callable
            options: Request options (timeout, headers, etc.)

        Example:
            >>> wrapper = ResponseWrapper(
            ...     client=client,
            ...     request={"model": "gpt-4", "messages": [...]},
            ...     tools=[weather_tool, calculator_tool],
            ...     max_tool_rounds=3
            ... )
        """
        self._client = client
        self._request = request
        self._tools = tools or []
        self._max_tool_rounds = max_tool_rounds
        self._options = options

        # State tracking
        self._state = ResponseState.INITIALIZED
        self._cached_data: CachedData = {}

        # Lazy initialization promises
        self._stream: ReusableStream | None = None
        self._init_promise: asyncio.Task[None] | None = None
        self._tool_execution_promise: asyncio.Task[None] | None = None

        # Cached task promises for deduplication
        self._message_task: asyncio.Task[dict[str, object]] | None = None
        self._text_task: asyncio.Task[str] | None = None

    @property
    def message(self) -> dict[str, object] | None:
        """Cached complete message (None if not yet consumed).

        This property provides access to the cached message without triggering
        computation. Use get_message() to retrieve and cache the message.

        Returns:
            dict[str, object] | None: Cached message or None

        Example:
            >>> message = await wrapper.get_message()
            >>> # Later access without await
            >>> if wrapper.message:
            ...     print("Message is cached:", wrapper.message)
        """
        return self._cached_data.get("message")

    @property
    def text(self) -> str | None:
        """Cached text content (None if not yet consumed).

        This property provides access to the cached text without triggering
        computation. Use get_text() to retrieve and cache the text.

        Returns:
            str | None: Cached text or None

        Example:
            >>> text = await wrapper.get_text()
            >>> # Later access without await
            >>> if wrapper.text:
            ...     print("Text is cached:", wrapper.text)
        """
        return self._cached_data.get("text")

    @property
    def state(self) -> ResponseState:
        """Current state of the response wrapper.

        Returns:
            ResponseState: Current state (INITIALIZED, STREAMING, COMPLETED, ERROR)

        Example:
            >>> print(f"State: {wrapper.state}")
            >>> if wrapper.state == ResponseState.COMPLETED:
            ...     print("Response fully processed")
        """
        return self._state

    async def _init_stream(self) -> None:
        """Initialize the stream if not already started.

        This is idempotent - multiple calls will return the same promise.
        The stream is created lazily on first consumption method call.

        State transitions: INITIALIZED → STREAMING
        """
        if self._init_promise is not None:
            # Already initializing or initialized
            await self._init_promise
            return

        # Create initialization task
        async def do_init() -> None:
            """Perform stream initialization."""
            try:
                # Update state
                self._state = ResponseState.STREAMING

                # TODO: Create actual stream from client.beta.responses.send
                # For now, create a placeholder
                # In PR 4.1 (Integration), this will call the actual API

                # Placeholder for stream creation
                # This will be replaced with actual API call in integration PR
                from collections.abc import AsyncIterator

                async def placeholder_source() -> AsyncIterator[dict[str, object]]:
                    """Placeholder async generator."""
                    if False:  # noqa: SIM223
                        yield {}

                self._stream = ReusableStream(placeholder_source())

            except Exception as e:
                self._state = ResponseState.ERROR
                raise e

        self._init_promise = asyncio.create_task(do_init())
        await self._init_promise

    async def _execute_tools_if_needed(self) -> None:
        """Execute tools automatically if provided and needed.

        This is idempotent - multiple calls will return the same promise.
        Tool execution happens after stream initialization.

        Integrates with tool_orchestrator for actual execution.
        """
        if self._tool_execution_promise is not None:
            # Already executing or executed
            await self._tool_execution_promise
            return

        # Create execution task
        async def do_execute_tools() -> None:
            """Perform tool execution if needed."""
            try:
                # Ensure stream is initialized
                await self._init_stream()

                # TODO: Integrate with tool_orchestrator.execute_tool_loop
                # This will be implemented in PR 4.1 (Integration)
                # For now, just mark as completed

                # Check if we have tools that need execution
                if not self._tools:
                    # No tools, nothing to execute
                    self._state = ResponseState.COMPLETED
                    return

                # TODO: Call tool_orchestrator.execute_tool_loop here
                # For now, just transition to completed state
                self._state = ResponseState.COMPLETED

            except Exception as e:
                self._state = ResponseState.ERROR
                raise e

        self._tool_execution_promise = asyncio.create_task(do_execute_tools())
        await self._tool_execution_promise

    async def _build_message(self) -> dict[str, object]:
        """Build complete message from stream with tool execution.

        This is the internal implementation of message building. It:
        1. Initializes the stream if needed
        2. Executes tools if provided
        3. Aggregates stream events into complete message
        4. Caches the result

        Returns:
            dict[str, object]: Complete message with all content

        Raises:
            Exception: Any error during stream consumption or tool execution
        """
        # Execute tools if needed (includes stream init)
        await self._execute_tools_if_needed()

        # TODO: Aggregate stream events into message
        # This will use stream_transformers.build_message in PR 3.2
        # For now, return placeholder
        message: dict[str, object] = {
            "role": "assistant",
            "content": [],
        }

        return message

    async def _extract_text(self) -> str:
        """Extract text content from the response.

        This is the internal implementation of text extraction. It:
        1. Builds the complete message (which handles tools)
        2. Extracts text content from message

        Returns:
            str: Text content from the response

        Raises:
            Exception: Any error during message building
        """
        # Get complete message
        message = await self.get_message()

        # TODO: Extract text from message
        # This will use proper content extraction in PR 3.2
        # For now, return placeholder
        text = ""

        return text

    async def get_message(self) -> dict[str, object]:
        """Get complete message with tool execution.

        This method retrieves the full message from the response, automatically
        executing any tools that were called. The result is cached for future
        calls.

        Returns:
            dict[str, object]: Complete message with role, content, and tool results

        Raises:
            Exception: Any error during stream consumption or tool execution

        Example:
            >>> message = await wrapper.get_message()
            >>> print(f"Role: {message['role']}")
            >>> print(f"Content: {message['content']}")
        """
        # Check cache first (property access)
        if self.message is not None:
            return self.message

        # Check if already building
        if self._message_task is None:
            # Create task for building message
            self._message_task = asyncio.create_task(self._build_message())

        # Wait for message and cache it
        message = await self._message_task
        self._cached_data["message"] = message
        return message

    async def get_text(self) -> str:
        """Get just the text content from the response.

        This method extracts only the text content from the response, ignoring
        tool calls and other content types. The result is cached for future calls.

        Returns:
            str: Text content from the response

        Raises:
            Exception: Any error during message building

        Example:
            >>> text = await wrapper.get_text()
            >>> print(f"Response: {text}")
        """
        # Check cache first (property access)
        if self.text is not None:
            return self.text

        # Check if already extracting
        if self._text_task is None:
            # Create task for extracting text
            self._text_task = asyncio.create_task(self._extract_text())

        # Wait for text and cache it
        text = await self._text_task
        self._cached_data["text"] = text
        return text

    async def close(self) -> None:
        """Close the wrapper and clean up resources.

        This method ensures proper cleanup of the underlying stream and
        cancels any pending tasks.

        Example:
            >>> async with ResponseWrapper(...) as wrapper:
            ...     text = await wrapper.get_text()
            >>> # Cleanup happens automatically
        """
        if self._stream:
            await self._stream.close()

    async def __aenter__(self) -> ResponseWrapper:
        """Enter async context manager.

        Returns:
            Self for use in context

        Example:
            >>> async with ResponseWrapper(...) as wrapper:
            ...     text = await wrapper.get_text()
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
        await self.close()
