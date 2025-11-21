#!/usr/bin/env python3
"""Multiple Consumption Patterns: One Call, Many Ways to Use It

This example shows the power of the ResponseWrapper pattern.
Perfect for: Understanding flexibility, optimizing for your use case

What you'll learn:
- Make one API call, use it multiple ways
- Choose the right pattern for your needs
- Save time and money by reusing responses

Run this: python examples/call_model_multiple_patterns.py
"""

import asyncio
import os

from openrouter import OpenRouter
from openrouter.call_model import call_model


async def main():
    """One API call, consumed in multiple different ways."""

    client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

    # Make ONE API call
    print("📞 Making API call...\n")
    response = await call_model(
        client=client,
        request={
            "model": "openai/gpt-4",
            "input": "Explain quantum computing in one paragraph",
        },
    )
    print("✅ Got response! Now let's consume it different ways...\n")

    # Pattern 1: Just the text (simplest)
    print("=" * 60)
    print("PATTERN 1: Get just the text (quick and simple)")
    print("=" * 60)
    text = await response.get_text()
    print(f"Text only:\n{text}\n")

    # Pattern 2: Full message (with metadata)
    print("=" * 60)
    print("PATTERN 2: Get complete message (includes metadata)")
    print("=" * 60)
    message = await response.get_message()
    print(f"Role: {message.get('role')}")
    print(f"Content: {message.get('content')[:100]}...")
    print(f"Keys available: {list(message.keys())}\n")

    # Pattern 3: Access cached data (no API call!)
    print("=" * 60)
    print("PATTERN 3: Access cached data (instant, no API call)")
    print("=" * 60)
    print(f"Cached text is available: {response.text is not None}")
    print(f"Cached message is available: {response.message is not None}")
    print(f"Current state: {response.state.value}\n")

    # Pattern 4: Stream would have worked too!
    print("=" * 60)
    print("PATTERN 4: Could have streamed instead")
    print("=" * 60)
    print("We used get_text(), but could have used:")
    print("  async for chunk in response.get_text_stream():")
    print("      print(chunk, end='', flush=True)")
    print("\nAll from the SAME single API call! 🎉")


async def streaming_example():
    """Bonus: Show streaming version of same call."""

    client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

    print("\n" + "=" * 60)
    print("BONUS: Here's what streaming looks like")
    print("=" * 60 + "\n")

    response = await call_model(
        client=client, request={"model": "openai/gpt-4", "input": "Count to 5 slowly"}
    )

    print("Streaming output: ", end="")
    async for chunk in response.get_text_stream():
        print(chunk, end="", flush=True)
        await asyncio.sleep(0.1)  # Slow it down to see chunks
    print("\n")


if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY not set")
        exit(1)

    print("🎭 Multiple Consumption Patterns Example\n")
    print("This demonstrates the power of ResponseWrapper:")
    print("One API call, multiple ways to consume the result!\n")

    asyncio.run(main())
    asyncio.run(streaming_example())

    print("\n" + "=" * 60)
    print("Key Takeaway:")
    print("=" * 60)
    print("You don't have to choose your consumption pattern upfront!")
    print("Make the call, then decide how to use the response.")
    print("This saves API calls and gives you maximum flexibility. ✨")
