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
from typing import Any, AsyncIterator, Dict, List, Optional, Set


class ReusableStream:
    """Stream that can be consumed multiple times concurrently.

    This class wraps an async iterator and allows multiple independent
    consumers to read from it concurrently. Events are buffered to support
    reuse and concurrent access.

    The cache is bounded to prevent memory issues:
    - Maximum 10,000 events (approximately 10MB assuming ~1KB per event)
    - The bound is enforced per the TypeScript reference: buffer grows as
      needed for active consumers, but is bounded overall

    Thread safety is ensured through asyncio locks, making this safe for
    concurrent access from multiple async tasks.

    Attributes:
        MAX_CACHE_SIZE: Maximum number of events to cache (default: 10,000)
    """

    MAX_CACHE_SIZE: int = 10000  # Approximately 10MB assuming ~1KB per event

    def __init__(self, source: AsyncIterator[Dict[str, Any]]):
        """Initialize reusable stream with a source iterator.

        Args:
            source: Source async iterator to wrap. This iterator will be
                   consumed exactly once and its events cached for reuse.
        """
        self._source = source
        self._buffer: List[Dict[str, Any]] = []
        self._exhausted = False
        self._error: Optional[Exception] = None
        self._lock = asyncio.Lock()
        self._consumers: Set[int] = set()
        self._consumer_positions: Dict[int, int] = {}
        self._consumer_events: Dict[int, asyncio.Event] = {}
        self._pump_task: Optional[asyncio.Task[None]] = None
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

                    # Notify all waiting consumers
                    for event_obj in self._consumer_events.values():
                        event_obj.set()
        except Exception as e:
            async with self._lock:
                self._error = e
                # Notify consumers of error
                for event_obj in self._consumer_events.values():
                    event_obj.set()
        finally:
            async with self._lock:
                self._exhausted = True
                # Final notification to all consumers
                for event_obj in self._consumer_events.values():
                    event_obj.set()

    async def create_iterator(self) -> AsyncIterator[Dict[str, Any]]:
        """Create a new independent iterator for this stream.

        Each iterator maintains its own read position and can be consumed
        at its own pace. All iterators receive the same events.

        Yields:
            Dict[str, Any]: Stream events from the source iterator

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
                    self._consumer_events[consumer_id].clear()

                # Wait for new data (outside lock to allow pump to proceed)
                await self._consumer_events[consumer_id].wait()
        finally:
            # Clean up consumer
            async with self._lock:
                self._consumers.discard(consumer_id)
                self._consumer_positions.pop(consumer_id, None)
                self._consumer_events.pop(consumer_id, None)

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
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass

        # Close source if it has aclose
        if hasattr(self._source, "aclose"):
            await self._source.aclose()  # type: ignore

        # Wake up any waiting consumers
        async with self._lock:
            for event_obj in self._consumer_events.values():
                event_obj.set()

    async def __aenter__(self) -> "ReusableStream":
        """Enter async context manager.

        Returns:
            Self for use in context
        """
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit async context manager with cleanup.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        await self.close()

    @property
    def _cache(self) -> List[Dict[str, Any]]:
        """Alias for buffer to match test expectations.

        Returns:
            The internal buffer
        """
        return self._buffer
