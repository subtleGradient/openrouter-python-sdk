#!/usr/bin/env python3
"""Tools: Give AI Superpowers

This example shows how to extend AI with custom functions (tools).
Perfect for: API integrations, data lookups, real-world actions

What you'll learn:
- Define tools the AI can use
- Let AI decide when to use them
- Get results with tool execution

Run this: python examples/call_model_with_tools.py
"""

import asyncio
import os
from typing import Any

from pydantic import BaseModel, Field
from openrouter import OpenRouter
from openrouter.call_model import call_model
from openrouter.call_model.tool_system import RegularTool


# Step 1: Define what data your tool needs
class WeatherParams(BaseModel):
    """Parameters for getting weather information."""

    location: str = Field(description="City name, e.g. 'San Francisco'")
    unit: str = Field(
        default="celsius", description="Temperature unit: celsius or fahrenheit"
    )


# Step 2: Create your tool
class WeatherTool(RegularTool[WeatherParams, dict[str, Any]]):
    """A tool that gets current weather for a location."""

    name: str = "get_weather"
    description: str = "Get the current weather for any city in the world"

    async def execute(self, params: WeatherParams, context: Any) -> dict[str, Any]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """This is where you'd call a real weather API.

        For this demo, we'll return simulated data.
        """
        # In a real app, you'd do:
        # weather_api = WeatherAPI()
        # return await weather_api.get(params.location, params.unit)

        # Demo: Return realistic fake data
        temps = {"San Francisco": 18, "London": 12, "Tokyo": 22}
        temp = temps.get(params.location, 20)

        if params.unit == "fahrenheit":
            temp = temp * 9 / 5 + 32

        return {
            "location": params.location,
            "temperature": temp,
            "unit": params.unit,
            "conditions": "partly cloudy",
            "humidity": 65,
        }


async def main():
    """Let AI use tools to answer questions."""

    client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

    # Create tool instance
    weather = WeatherTool()

    # Ask a question that requires the tool
    response = await call_model(
        client=client,
        request={
            "model": "openai/gpt-4",
            "input": "What's the weather like in San Francisco and London? Give me a comparison.",
        },
        tools=[weather],  # AI can use this tool
        max_tool_rounds=3,  # Allow up to 3 rounds of tool execution
    )

    print("🤖 AI is thinking (may use tools)...\n")

    # Get the final answer (tools auto-executed!)
    answer = await response.get_text()
    print(f"AI Answer:\n{answer}")


if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY not set")
        exit(1)

    print("🛠️  Tools Example\n" + "=" * 50 + "\n")
    print("This shows how AI can use tools to get real-time information.\n")
    asyncio.run(main())
    print("\n✅ Done! The AI used the weather tool to answer your question.")
