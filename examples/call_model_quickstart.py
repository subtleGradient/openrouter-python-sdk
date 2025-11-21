#!/usr/bin/env python3
"""Quick Start: Your First call_model() Call

This example shows the fastest path to success with call_model().
Perfect for: Getting started, prototyping, quick tests

What you'll learn:
- How to make your first AI call in 3 lines
- Get a text response instantly
- Handle errors gracefully

Run this: python examples/call_model_quickstart.py
"""

import asyncio
import os

from openrouter import OpenRouter
from openrouter.call_model import call_model


async def main():
    """Your first AI conversation in 3 simple steps."""

    # Step 1: Create your client
    client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

    # Step 2: Make the call
    response = await call_model(
        client=client,
        request={
            "model": "openai/gpt-4",
            "input": "Write a haiku about Python programming",
        },
    )

    # Step 3: Get your answer
    text = await response.get_text()
    print(f"AI says:\n{text}")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
        exit(1)

    print("🚀 Starting your first call_model() example...\n")
    asyncio.run(main())
    print("\n✅ Success! You just made your first AI call!")
