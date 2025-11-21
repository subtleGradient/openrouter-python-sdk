#!/usr/bin/env python3
"""Streaming: Watch AI Think in Real-Time

This example shows how to stream responses as they're generated.
Perfect for: Chat interfaces, real-time displays, user engagement

What you'll learn:
- Stream text as the AI generates it
- Display progressive results
- Build responsive UIs

Run this: python examples/call_model_streaming.py
"""

import asyncio
import os

from openrouter import OpenRouter
from openrouter.call_model import call_model


async def main():
    """Stream AI responses in real-time, word by word."""

    client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

    # Ask for a story - perfect for seeing streaming in action
    response = await call_model(
        client=client,
        request={
            "model": "openai/gpt-4",
            "input": "Tell me a very short story about a brave robot",
        },
    )

    print("AI is writing (streaming live):\n")

    # Stream each word as it arrives
    async for text_chunk in response.get_text_stream():
        print(text_chunk, end="", flush=True)

    print("\n\nStory complete.")


if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set")
        exit(1)

    print("Streaming Example\n" + "=" * 50 + "\n")
    asyncio.run(main())
