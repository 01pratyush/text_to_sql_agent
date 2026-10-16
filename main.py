import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Windows consoles default to cp1252, which can't print some characters LLMs emit
sys.stdout.reconfigure(encoding="utf-8")

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

MODEL = "openai/gpt-oss-120b"
MAX_TOOL_ITERATIONS = 5  # safety limit for the inner (tool) loop

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Use the available tools when they help answer the question. "
    "Be concise."
)


# ---------------------------------------------------------------------------
# Tools: plain Python functions the model can ask us to run
# ---------------------------------------------------------------------------

def get_current_time() -> str:
    """Return the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


# Maps the tool name the model uses -> the Python function we run
AVAILABLE_TOOLS = {
    "get_current_time": get_current_time,
    "add_numbers": add_numbers,
}

# Tool definitions sent to the model (JSON Schema describing each function)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
    },
]


def run_tool(name: str, arguments: str) -> str:
    """Execute a tool call requested by the model and return the result as a string."""
    func = AVAILABLE_TOOLS.get(name)
    if func is None:
        return f"Error: unknown tool '{name}'"
    try:
        args = json.loads(arguments or "{}")
        return str(func(**args))
    except Exception as e:
        # Send errors back to the model so it can correct itself
        return f"Error running {name}: {e}"


# ---------------------------------------------------------------------------
# Inner loop: one user turn, possibly several LLM calls if tools are used
# ---------------------------------------------------------------------------

def run_agent_turn(messages: list) -> str:
    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0,
        )
        msg = response.choices[0].message

        # No tool calls -> the model has produced its final answer
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content

        # The model wants tools: record its request in history...
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        # ...then run each tool and append its result, linked by tool_call_id
        for tc in msg.tool_calls:
            print(f"  [tool call] {tc.function.name}({tc.function.arguments})")
            result = run_tool(tc.function.name, tc.function.arguments)
            print(f"  [tool result] {result}")
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )
        # Loop again so the model can read the tool results

    stop_msg = "Stopped: too many tool calls in one turn."
    messages.append({"role": "assistant", "content": stop_msg})
    return stop_msg


# ---------------------------------------------------------------------------
# Outer loop: the multi-turn chat. `messages` is the conversation memory.
# ---------------------------------------------------------------------------

def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Chat started. Type 'exit' to quit, 'history' to see the messages list.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "history":
            print(json.dumps(messages, indent=2))
            continue

        messages.append({"role": "user", "content": user_input})
        reply = run_agent_turn(messages)
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()
