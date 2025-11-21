# 🚀 Quick Start Guide: call_model()

Welcome! This guide will get you from zero to hero with OpenRouter's `call_model()` API in just 5 minutes.

## What You'll Learn

- ✨ Make your first AI call in 3 lines of code
- 📡 Stream responses in real-time
- 🛠️ Give AI superpowers with tools
- 🎭 Use multiple consumption patterns
- 🎯 Choose the right approach for your needs

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

**That's it!** You just made your first AI call. 🎉

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

**The magic:** One API call, infinite flexibility. Save money, go fast.

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

**Key insight:** The ResponseWrapper makes ONE API call but gives you multiple ways to use the response. It's like ordering one pizza but being able to eat it multiple ways - by the slice, whole, cold, hot... you get the idea. 🍕

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

🎓 **Learn More:**
- [Full API Reference](https://openrouter.ai/docs/sdks/python/reference)
- [Tool System Guide](docs/tools.md)
- [Streaming Deep Dive](docs/streaming.md)

💬 **Get Help:**
- [Discord Community](https://discord.gg/openrouter)
- [GitHub Issues](https://github.com/OpenRouterTeam/openrouter-python/issues)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/openrouter)

🌟 **Share:**
- Built something cool? Share it in our Discord!
- Found a bug? Let us know on GitHub
- Have a question? Ask on Stack Overflow

---

## You're Ready!

You now know enough to build amazing AI-powered features. Start with a simple example, then gradually add complexity as you need it.

Remember the philosophy:
- **Simple things are simple** (3 lines for basic call)
- **Complex things are possible** (tools, streaming, custom logic)
- **You're in control** (choose your consumption pattern)

Happy building! 🚀

---

*Made with ❤️ by the OpenRouter team*
