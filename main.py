import sys
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# Windows consoles default to cp1252, which can't print some characters LLMs emit
sys.stdout.reconfigure(encoding="utf-8")

MODEL = "openai/gpt-oss-120b"
# Safety limit for the inner loop. Counts graph steps (node runs), not loop
# rounds: agent + tools = 2 steps per round, so 10 allows ~5 rounds
RECURSION_LIMIT = 10

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Use the available tools when they help answer the question. "
    "Be concise."
)


# ---------------------------------------------------------------------------
# Tools: @tool turns each function into a tool object and builds its JSON
# schema from the name, type hints and docstring (replaces the TOOLS list)
# ---------------------------------------------------------------------------

@tool
def get_current_time() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# parse_docstring=True reads the Args: section into per-parameter descriptions
@tool(parse_docstring=True)
def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return the sum.

    Args:
        a: First number
        b: Second number
    """
    return a + b


TOOLS = [get_current_time, add_numbers]


# LangChain chat model; reads GROQ_API_KEY from the environment.
# bind_tools converts the tool objects to JSON schemas and attaches them to every call
llm = ChatGroq(model=MODEL, temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# Nodes: the steps of the graph
# ---------------------------------------------------------------------------

def agent(state: MessagesState):
    """Call the LLM with the conversation so far (replaces the API call in the loop)."""
    # The system prompt is added on every call instead of being stored in state,
    # so changing it takes effect without touching saved conversations
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    # Returning a list *appends* to state["messages"] (the add_messages reducer)
    return {"messages": [response]}


# ToolNode runs every tool call in the last AIMessage and returns ToolMessages
# with matching tool_call_ids; tool errors are sent back to the model as text
# (replaces AVAILABLE_TOOLS, run_tool() and the tool-result loop)
tool_node = ToolNode(TOOLS)


# ---------------------------------------------------------------------------
# Graph: nodes + edges form the inner loop
#
#   START -> agent --(tool calls?)--> tools -> agent ...
#                 \--(no tool calls)--> END
# ---------------------------------------------------------------------------

builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # replaces `if not msg.tool_calls`
builder.add_edge("tools", "agent")                       # replaces "loop again"

# The checkpointer saves state after every step, keyed by thread_id
# (replaces keeping the messages list alive in main())
graph = builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Outer loop: the multi-turn chat. History lives in the checkpointer.
# ---------------------------------------------------------------------------

def main():
    config = {"configurable": {"thread_id": "chat-1"}, "recursion_limit": RECURSION_LIMIT}
    print("Chat started. Type 'exit' to quit, 'history' to see the messages list.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "history":
            # Read the saved state for this thread from the checkpointer
            for m in graph.get_state(config).values.get("messages", []):
                m.pretty_print()
            continue

        # Only the NEW message is sent; earlier ones are loaded from the checkpointer
        inputs = {"messages": [HumanMessage(content=user_input)]}
        try:
            # stream_mode="updates" yields each node's output as it runs,
            # as {node_name: {"messages": [...]}}
            for update in graph.stream(inputs, config, stream_mode="updates"):
                for node, output in update.items():
                    last = output["messages"][-1]
                    if node == "agent" and last.tool_calls:
                        for tc in last.tool_calls:
                            print(f"  [tool call] {tc['name']}({tc['args']})")
                    elif node == "tools":
                        for m in output["messages"]:
                            print(f"  [tool result] {m.content}")
                    elif node == "agent":
                        print(f"Bot: {last.content}\n")
        except GraphRecursionError:
            print("Bot: Stopped: too many tool calls in one turn.\n")


if __name__ == "__main__":
    main()
