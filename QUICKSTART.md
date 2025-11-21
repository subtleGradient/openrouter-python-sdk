# Quick Start Guide: call_model()

This guide demonstrates OpenRouter's `call_model()` API with progressively complex examples.

## What You'll Learn

- Make your first AI call with minimal code
- Stream responses in real-time
- Integrate tools for extended functionality
- Use multiple consumption patterns
- Choose the right approach for your use case

## Prerequisites

```bash
# Install the SDK
pip install openrouter

# Set your API key
export OPENROUTER_API_KEY='your-key-here'
```

Get your API key at: https://openrouter.ai/keys

---

## Your First Call (30 seconds)

The simplest possible example:

```python
import asyncio
from openrouter import OpenRouter
from openrouter.call_model import call_model

async def main():
    client = OpenRouter(api_key="your-key-here")
    
    response = await call_model(
        client=client,
        request={"model": "openai/gpt-4", "input": "Say hello!"}
    )
    
    text = await response.get_text()
    print(text)

asyncio.run(main())
```

That's all you need for a basic AI call.

---

## Level Up: Streaming (2 minutes)

Want to see AI responses as they're generated? Use streaming:

```python
response = await call_model(
    client=client,
    request={"model": "openai/gpt-4", "input": "Tell me a story"}
)

# Stream each chunk as it arrives
async for chunk in response.get_text_stream():
    print(chunk, end="", flush=True)
```

**Perfect for:** Chat UIs, progress indicators, real-time displays

---

## Power Move: Tools (5 minutes)

Give AI the ability to call your functions:

```python
from pydantic import BaseModel
from openrouter.call_model.tool_system import RegularTool

# 1. Define your tool's input
class WeatherParams(BaseModel):
    location: str
    unit: str = "celsius"

# 2. Create the tool
class WeatherTool(RegularTool[WeatherParams, dict]):
    name = "get_weather"
    description = "Get current weather for a city"
    
    async def execute(self, params, context):
        # Your code here - call weather API, database, etc.
        return {"temp": 72, "conditions": "sunny"}

# 3. Use it!
response = await call_model(
    client=client,
    request={"model": "openai/gpt-4", "input": "What's the weather in NYC?"},
    tools=[WeatherTool()]
)

# AI automatically calls your tool and uses the result!
answer = await response.get_text()
```

**Perfect for:** API integrations, data lookups, taking actions

---

## Pro Tip: Multiple Consumption Patterns

The response object is flexible - use it however you need:

```python
# Make ONE API call
response = await call_model(client, request)

# Then consume it in any way:
text = await response.get_text()                    # Just the text
message = await response.get_message()              # Full message object
async for chunk in response.get_text_stream():...  # Stream it
async for event in response.get_full_stream():...  # All raw events

# Access cached data instantly (no extra API calls!)
if response.text:
    print(f"Already have: {response.text}")
```

**Key benefit:** One API call supports multiple consumption patterns, reducing API costs and improving performance.

---

## Complete Examples

Ready to dive deeper? Check out our runnable examples:

1. **[call_model_quickstart.py](examples/call_model_quickstart.py)** - Your first call (< 20 lines)
2. **[call_model_streaming.py](examples/call_model_streaming.py)** - Real-time streaming
3. **[call_model_with_tools.py](examples/call_model_with_tools.py)** - AI with superpowers
4. **[call_model_multiple_patterns.py](examples/call_model_multiple_patterns.py)** - All patterns in action

Run any example:
```bash
python examples/call_model_quickstart.py
```

---

## Common Patterns & Recipes

### Simple Question & Answer
```python
response = await call_model(client, {"model": "openai/gpt-4", "input": "Explain quantum computing"})
answer = await response.get_text()
```

### Chat with History
```python
response = await call_model(client, {
    "model": "openai/gpt-4",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "Tell me about Python"}
    ]
})
```

### Stream to UI
```python
async for chunk in (await call_model(client, request)).get_text_stream():
    update_ui(chunk)  # Your UI update function
```

### Multiple Tools
```python
response = await call_model(
    client,
    request,
    tools=[WeatherTool(), CalculatorTool(), SearchTool()],
    max_tool_rounds=5  # Allow up to 5 rounds of tool calls
)
```

---

## What's Happening Under the Hood?

```
call_model()
    ↓
Makes 1 API call to /responses (streaming)
    ↓
Returns ResponseWrapper
    ↓
You choose how to consume:
  ├── get_text() → Extract just text
  ├── get_message() → Get full message + metadata
  ├── get_text_stream() → Stream text chunks
  └── get_full_stream() → All raw SSE events
    ↓
Tools auto-execute if provided
    ↓
You get the final result!
```

**Architecture note:** The ResponseWrapper makes a single API call but provides multiple consumption methods through internal caching and stream management.

---

## Troubleshooting

### "OPENROUTER_API_KEY not set"
```bash
export OPENROUTER_API_KEY='sk-or-v1-...'
# or in Python:
client = OpenRouter(api_key="sk-or-v1-...")
```

### "call_model not found"
```bash
pip install --upgrade openrouter
```

### Tool not being called?
Make sure:
- Tool has `name` and `description` attributes
- Description clearly explains when to use it
- `max_tool_rounds` is > 0 (default is 5)

### Want synchronous code?
Wrap in `asyncio.run()`:
```python
import asyncio

def my_function():
    return asyncio.run(my_async_function())
```

---

## Next Steps

**Learn More:**
- [Full API Reference](https://openrouter.ai/docs/sdks/python/reference)
- [Tool System Guide](docs/tools.md)
- [Streaming Deep Dive](docs/streaming.md)

**Get Help:**
- [Discord Community](https://discord.gg/openrouter)
- [GitHub Issues](https://github.com/OpenRouterTeam/openrouter-python/issues)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/openrouter)

---

## Summary

You now have the foundation to build AI-powered features with the OpenRouter SDK.

Key principles:
- **Simple things are simple** - Basic calls require minimal code
- **Complex things are possible** - Tools, streaming, and custom logic are fully supported
- **You're in control** - Choose the consumption pattern that fits your requirements

---

*OpenRouter Python SDK Documentation*
