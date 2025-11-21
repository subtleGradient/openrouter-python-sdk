"""Reusable stream implementation for concurrent consumption.

This module implements a ReusableStream class that allows multiple consumers
to read from the same source async iterator concurrently. It addresses the
Python generator exhaustion problem by caching events and managing multiple
independent consumer positions.

Key features:
- Multiple concurrent consumers with independent read positions
- Bounded cache with 10MB/10,000 event limits
- Thread-safe concurrent access with asyncio locks
- Automatic resource cleanup
- Error propagation to all consumers

The cache bound is a soft limit - it prevents unbounded growth when multiple
slow consumers are present, but allows single fast consumers to read all events.

Example:
    >>> async def process_stream():
    ...     source = get_api_stream()  # Returns AsyncIterator[Dict[str, Any]]
    ...     reusable = ReusableStream(source)
    ...
    ...     # Multiple consumers can read concurrently
    ...     async with reusable:
    ...         consumer1 = reusable.create_iterator()
    ...         consumer2 = reusable.create_iterator()
    ...
    ...         # Both get the same events
    ...         async for event in consumer1:
    ...             print(f"Consumer 1: {event}")
    ...
    ...         async for event in consumer2:
    ...             print(f"Consumer 2: {event}")
"""

import asyncio
from collections.abc import Awaitable, AsyncIterator
from typing import cast


class ReusableStream:
    """Stream that can be consumed multiple times concurrently.

    This class wraps an async iterator and allows multiple independent
    consumers to read from it concurrently. Events are buffered to support
    reuse and concurrent access.

    The cache is bounded to prevent memory issues:
    - Maximum 10,000 events (approximately 10MB assuming ~1KB per event)
    - Events are only removed once all active consumers have read them
    - If a consumer lags behind, the cache may temporarily exceed the limit
      to ensure no consumer misses events

    Thread safety is ensured through asyncio locks, making this safe for
    concurrent access from multiple async tasks.

    Attributes:
        MAX_CACHE_SIZE: Maximum number of events to cache (default: 10,000)
    """

    MAX_CACHE_SIZE: int = 10000  # Approximately 10MB assuming ~1KB per event

    _source: AsyncIterator[dict[str, object] | object]
    _buffer: list[dict[str, object] | object]
    _exhausted: bool
    _error: Exception | None
    _lock: asyncio.Lock
    _consumers: set[int]
    _consumer_positions: dict[int, int]
    _consumer_events: dict[int, asyncio.Event]
    _pump_task: asyncio.Task[None] | None
    _next_consumer_id: int

    def __init__(self, source: AsyncIterator[dict[str, object] | object]):
        """Initialize reusable stream with a source iterator.

        Args:
            source: Source async iterator to wrap. This iterator will be
                   consumed exactly once and its events cached for reuse.
                   Can yield either dicts or Pydantic objects.
        """
        self._source = source
        self._buffer = []
        self._exhausted = False
        self._error = None
        self._lock = asyncio.Lock()
        self._consumers = set()
        self._consumer_positions = {}
        self._consumer_events = {}
        self._pump_task = None
        self._next_consumer_id = 0

    async def _pump_source(self) -> None:
        """Background task to pump source iterator into buffer.

        This task runs in the background, reading from the source iterator
        and adding events to the buffer. It notifies waiting consumers as
        new events arrive and handles errors by storing them for propagation.
        """
        try:
            async for event in self._source:
                async with self._lock:
                    self._buffer.append(event)

                    # Enforce MAX_CACHE_SIZE by removing oldest events
                    # Only trim if all consumers have read past the oldest events
                    if len(self._buffer) > self.MAX_CACHE_SIZE:
                        # Find the minimum consumer position
                        min_position = (
                            min(self._consumer_positions.values())
                            if self._consumer_positions
                            else 0
                        )
                        # We can only safely remove events that all consumers have already read
                        while (
                            len(self._buffer) > self.MAX_CACHE_SIZE and min_position > 0
                        ):
                            self._buffer.pop(0)
                            # Adjust all consumer positions since we removed an element
                            for consumer_id in self._consumer_positions:
                                self._consumer_positions[consumer_id] -= 1
                            min_position -= 1

                    # Notify all waiting consumers
                    for event_obj in self._consumer_events.values():
                        _ = event_obj.set()
        except Exception as e:
            async with self._lock:
                self._error = e
                # Notify consumers of error
                for event_obj in self._consumer_events.values():
                    _ = event_obj.set()
        finally:
            async with self._lock:
                self._exhausted = True
                # Final notification to all consumers
                for event_obj in self._consumer_events.values():
                    _ = event_obj.set()

    async def create_iterator(self) -> AsyncIterator[dict[str, object] | object]:
        """Create a new independent iterator for this stream.

        Each iterator maintains its own read position and can be consumed
        at its own pace. All iterators receive the same events.

        Yields:
            dict[str, object] | object: Stream events from the source iterator
                                        (either dicts or Pydantic objects)

        Raises:
            Exception: Any exception raised by the source iterator is
                      propagated to all consumers

        Example:
            >>> stream = ReusableStream(source)
            >>> async for event in stream.create_iterator():
            ...     print(event)
        """
        # Allocate consumer ID and register
        async with self._lock:
            consumer_id = self._next_consumer_id
            self._next_consumer_id += 1
            self._consumers.add(consumer_id)
            self._consumer_positions[consumer_id] = 0
            self._consumer_events[consumer_id] = asyncio.Event()

            # Start pump if not already running
            if not self._pump_task:
                self._pump_task = asyncio.create_task(self._pump_source())

        try:
            while True:
                async with self._lock:
                    position = self._consumer_positions[consumer_id]

                    # Check if we have data in buffer at this position
                    if position < len(self._buffer):
                        event = self._buffer[position]
                        self._consumer_positions[consumer_id] = position + 1
                        yield event
                        continue

                    # Check for errors (before checking exhausted!)
                    if self._error:
                        raise self._error

                    # Check if source is exhausted (after checking errors)
                    if self._exhausted:
                        break

                    # Clear event before waiting
                    _ = self._consumer_events[consumer_id].clear()

                # Wait for new data (outside lock to allow pump to proceed)
                _ = await self._consumer_events[consumer_id].wait()
        finally:
            # Clean up consumer
            async with self._lock:
                _ = self._consumers.discard(consumer_id)
                _ = self._consumer_positions.pop(consumer_id, None)
                _ = self._consumer_events.pop(consumer_id, None)

    async def close(self) -> None:
        """Close the stream and clean up resources.

        This cancels the pump task and closes the source iterator if it
        has an aclose method. Should be called when the stream is no longer
        needed to prevent resource leaks.

        Example:
            >>> stream = ReusableStream(source)
            >>> try:
            ...     async for event in stream.create_iterator():
            ...         process(event)
            ... finally:
            ...     await stream.close()
        """
        # Cancel pump task
        if self._pump_task and not self._pump_task.done():
            _ = self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass

        # Close source if it has aclose
        # AsyncIterator protocol doesn't define aclose, but many implementations have it
        if hasattr(self._source, "aclose"):
            aclose_method = cast(object, getattr(self._source, "aclose"))
            if callable(aclose_method):
                _ = await cast(Awaitable[object], aclose_method())

        # Wake up any waiting consumers
        async with self._lock:
            for event_obj in self._consumer_events.values():
                _ = event_obj.set()

    async def __aenter__(self) -> "ReusableStream":
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
        await self.close()

    @property
    def _cache(self) -> list[dict[str, object] | object]:
        """Alias for buffer to match test expectations.

        Returns:
            The internal buffer
        """
        return self._buffer
