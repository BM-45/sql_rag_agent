from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langsmith import traceable
from dotenv import load_dotenv
import sqlite3
import json

load_dotenv()
from core.rag_retrieval import embed_model, collection


# Defining tools and addign traces to it.


@tool
@traceable(name="search_schema", run_type="tool")
def search_schema(question: str) -> str:
    """Search the database for relevant table schemas based on a natural language question.
    Use this tool FIRST to find which tables are available before writing SQL.
    Returns the top 3 matching table schemas."""

    if collection is None:
        return "ERROR: ChromaDB collection not initialized"

    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    schemas = []
    for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
        schemas.append(f"[distance: {distance:.4f}] {meta['original_context']}")

    return "\n\n".join(schemas)


@tool
@traceable(name="validate_sql", run_type="tool")
def validate_sql(query: str) -> str:
    """Validate that a SQL query is safe to execute.
    Use this BEFORE running any SQL query.
    Returns 'VALID' if safe, or an error message explaining what's wrong."""

    upper = query.upper().strip()

    if not upper.startswith("SELECT"):
        return "ERROR: Only SELECT statements are allowed. Your query starts with: " + upper.split()[0]

    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE", "EXEC", "--"]
    for kw in dangerous:
        if kw in upper:
            return f"ERROR: Dangerous keyword '{kw}' detected. Remove it and try again."
  
    if len(query.strip()) < 10:
        return "ERROR: Query is too short. Write a complete SQL query."

    if "FROM" not in upper:
        return "ERROR: Query missing FROM clause. SQL needs: SELECT ... FROM table_name"

    return f"VALID: Query passed all safety checks. Safe to execute: {query}"


@tool
@traceable(name="execute_sql", run_type="tool")
def execute_sql(query: str) -> str:
    """Execute a validated SQL query against the database.
    Only use this AFTER validate_sql returns 'VALID'.
    Returns the query results."""

    try:
        conn = sqlite3.connect("test_db.sqlite")
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()

        if not results:
            return "Query returned no results."

        return f"Columns: {columns}\nResults: {results[:20]}"

    except Exception as e:
        return f"SQL EXECUTION ERROR: {str(e)}. Fix the query and try again."


# state defining.

class SqlAgent(TypedDict):
    messages: Annotated[list, add_messages]


# Setting up LLM.

SYSTEM_PROMPT = """You are a SQL analytics agent. Your job is to answer questions about data by querying a database.

WORKFLOW — follow these steps in order:
1. Use search_schema to find relevant table schemas
2. Write a SQL query based on the schema
3. Use validate_sql to check the query is safe
4. Use execute_sql to run the query
5. Respond with the answer in natural language

RULES:
- ALWAYS search for schemas first, never guess table or column names
- ALWAYS validate before executing
- If validation fails, read the error message carefully and fix the query
- If execution fails, read the error and try a different approach
- Only use SELECT statements
- Use exact column and table names from the schema
- Maximum 3 retry attempts, then explain what went wrong"""

tools = [search_schema, validate_sql, execute_sql]
tool_map = {t.name: t for t in tools}

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    num_predict=1024
).bind_tools(tools)


# Defining workflow.

MAX_ITERATIONS = 10


@traceable(name="agent_node", run_type="chain")
def agent_node(state: SqlAgent) -> dict:
    """LLM reasons and decides what to do next."""
    messages = state["messages"]

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    tool_call_count = sum(
        1 for m in messages
        if hasattr(m, "tool_calls") and m.tool_calls
    )
    if tool_call_count >= MAX_ITERATIONS:
        return {
            "messages": [AIMessage(content="I've reached the maximum number of attempts. Please try rephrasing your question.")]
        }

    response = llm.invoke(messages)
    return {"messages": [response]}


@traceable(name="tool_node", run_type="chain")
def tool_node(state: SqlAgent) -> dict:
    """Execute whatever tool(s) the LLM chose."""
    last_message = state["messages"][-1]
    results = []

    for call in last_message.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]

        try:
            result = tool_map[tool_name].invoke(tool_args)
        except Exception as e:
            result = f"TOOL ERROR: {str(e)}. Try a different approach."

        results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call["id"]
            )
        )

    return {"messages": results}


# This step is routing.

@traceable(name="should_continue", run_type="chain")
def should_continue(state: SqlAgent) -> str:
    """Decide: does LLM want to use a tool, or is it done?"""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"


# Buidling the graph using langgraph framework.

sql_workflow = StateGraph(SqlAgent)

sql_workflow.add_node("agent", agent_node)
sql_workflow.add_node("tools", tool_node)

sql_workflow.add_edge(START, "agent")
sql_workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)
sql_workflow.add_edge("tools", "agent")

app = sql_workflow.compile()


# Runnnig it.

@traceable(name="ask_agent", run_type="chain")
def ask(question: str, verbose: bool = True) -> str:
    """Ask the agent a question and get an answer."""

    result = app.invoke({
        "messages": [HumanMessage(content=question)]
    })

    if verbose:
        print(f"\n{'─'*60}")
        print("CONVERSATION TRACE:")
        print(f"{'─'*60}")
        for msg in result["messages"]:
            if isinstance(msg, HumanMessage):
                print(f"\n USER: {msg.content}")
            elif isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"\nAGENT calls: {tc['name']}({json.dumps(tc['args'])})")
                elif msg.content:
                    print(f"\nAGENT answer: {msg.content}")
            elif isinstance(msg, ToolMessage):
                print(f"\nTOOL result: {msg.content[:150]}...")

    # Extract final answer
    final_answer = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
            final_answer = msg.content
            break

    if verbose:
        print(f"\n{'─'*60}")
        print(f"FINAL ANSWER: {final_answer}")
        print(f"{'─'*60}")

    return final_answer


if __name__ == "__main__":
    questions = [
        "Which city has highest population?",
    ]

    for q in questions:
        print(f"\n\n{'═'*60}")
        print(f"USER: {q}")
        print(f"{'═'*60}")
        answer = ask(q)