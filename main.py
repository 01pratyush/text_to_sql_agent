import logging
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent_callbacks import LoggingCallbackHandler
from log_config import run_context, setup_logging
from sql_tools import SQL_TOOLS
from tracing import setup_tracing

log = logging.getLogger("cli")

load_dotenv()

# Windows consoles default to cp1252, which can't print some characters LLMs emit
sys.stdout.reconfigure(encoding="utf-8")

MODEL = "openai/gpt-oss-120b"
# Safety limit on the agent <-> tools cycle within one user turn. Counts graph
# steps (node runs), not rounds: agent + tools = 2 steps per round, so 25 allows
# ~12 rounds (exploring the schema and fixing failed queries takes several)
RECURSION_LIMIT = 25

SYSTEM_PROMPT = """You are a data analyst answering questions about a DuckDB database \
(the TPC-H benchmark: customers, orders, line items, parts, suppliers, nations, regions).

How to work:
1. If you don't already know the relevant tables, call list_tables, then describe_table
   for the ones you need. Use the exact column names you see; don't guess them.
2. Write one DuckDB SELECT query and run it with run_sql.
3. If the query fails or the result looks wrong, read the error, fix the query, and retry.
4. Results are capped at 50 rows, so aggregate in SQL or use ORDER BY ... LIMIT.

Answer format:
- Give the answer in plain language first.
- Then show the final SQL query you used in a ```sql code block.
- If the data can't answer the question, say so instead of guessing."""

# The tools live in sql_tools.py: list_tables, describe_table, run_sql
TOOLS = SQL_TOOLS


# LangChain chat model; reads GROQ_API_KEY from the environment.
# bind_tools converts the tool objects to JSON schemas and attaches them to every call
llm = ChatGroq(model=MODEL, temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# Nodes: the steps of the graph
# ---------------------------------------------------------------------------

def agent(state: MessagesState):
    """Call the LLM with the conversation so far."""
    # The system prompt is added on every call instead of being stored in state,
    # so changing it takes effect without touching saved conversations
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    # Returning a list *appends* to state["messages"] (the add_messages reducer)
    return {"messages": [response]}


# ToolNode runs every tool call in the last AIMessage and returns ToolMessages
# with matching tool_call_ids. By default it only catches invalid arguments
# (sent back to the model as text); any other exception a tool raises stops
# the graph, so the tools in sql_tools.py return expected errors (bad SQL,
# unknown table) as strings instead of raising
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
builder.add_conditional_edges("agent", tools_condition)  # "tools" if tool calls, else END
builder.add_edge("tools", "agent")                       # model reads the tool results

# The checkpointer saves state after every step, keyed by thread_id.
# InMemorySaver keeps it in RAM, so history is lost when the program exits
graph = builder.compile(checkpointer=InMemorySaver())


def make_config(thread_id: str) -> dict:
    """Run config for one conversation: its thread, the step limit, and the
    logging callbacks (passed down to every LLM and tool call in the run)."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
        "callbacks": [LoggingCallbackHandler()],
    }


# ---------------------------------------------------------------------------
# Outer loop: the multi-turn chat. History lives in the checkpointer.
# ---------------------------------------------------------------------------

def main():
    setup_logging()
    setup_tracing("text-to-sql-chat")
    config = make_config("chat-1")
    log.info("session_start", extra={"model": MODEL, "thread_id": "chat-1"})
    print("Chat started. Type 'exit' to quit, 'history' to see the messages list.")
    print("Tool calls are logged below each question (set LOG_LEVEL=WARNING to hide them).\n")

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
        # One run_id per turn: every LLM and tool event in this turn is tagged with it
        with run_context() as run_id:
            log.info("turn_start", extra={"question": user_input})
            try:
                # run_id in the metadata tags this turn's Phoenix trace, matching the logs
                result = graph.invoke(inputs, {**config, "metadata": {"run_id": run_id}})
                reply = result["messages"][-1].content
                log.info("turn_end", extra={"status": "ok"})
            except GraphRecursionError:
                reply = "Stopped: too many tool calls in one turn."
                log.warning("turn_end", extra={"status": "recursion_limit"})
            except Exception:
                log.exception("turn_end", extra={"status": "error"})
                reply = "Sorry, something went wrong (details are in logs/agent.jsonl)."
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()
