# SQL RAG Agent - Deterministic Pipeline with Intent Classification
# Supports business document context passed from API layer
#
# Flow:
#   classify -> GENERAL  -> general_answer -> END
#            -> METADATA -> metadata_answer -> END
#            -> SQL      -> retrieve -> generate -> validate -> execute -> format -> END

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langsmith import traceable
from dotenv import load_dotenv
import sqlite3
import re

load_dotenv()
from core.rag_retrieval import embed_model, collection


# State

class AgentState(TypedDict):
    question: str
    intent: str
    doc_context: str
    schema: str
    generated_sql: str
    is_valid: bool
    error_message: str
    retry_count: int
    query_results: str
    answer: str


# LLM

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    num_predict=512
)


# Node 1: Classify intent using LLM

@traceable(name="classify_question", run_type="chain")
def classify_question(state: AgentState) -> dict:
    question = state["question"]
    doc_context = state.get("doc_context", "")

    context_note = "No documents uploaded."
    if doc_context:
        context_note = f"User has uploaded documents with this content:\n{doc_context[:500]}"

    system_msg = f"""/no_think
You are a query intent classifier. Classify the user question into exactly one category.

Categories:
- SQL: questions that need data from a database (counts, averages, filters, comparisons, rankings)
- METADATA: questions about what tables, columns, or data is available
- DOCUMENT: questions that can be answered from the uploaded document content below
- GENERAL: greetings, help requests, or questions about your capabilities

{context_note}

Respond with ONLY one word: SQL or METADATA or DOCUMENT or GENERAL"""

    # Debug logging
    print(f"\n[CLASSIFIER] Question: {question}")
    print(f"[CLASSIFIER] Has doc_context: {bool(doc_context)}")
    print(f"[CLASSIFIER] Doc context preview: {doc_context[:200] if doc_context else 'EMPTY'}")

    response = llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=question)
    ])

    raw_response = response.content.strip()
    print(f"[CLASSIFIER] Raw LLM response: '{raw_response}'")

    intent = raw_response.upper()
    intent = re.sub(r'<think>.*?</think>', '', intent, flags=re.DOTALL).strip()
    print(f"[CLASSIFIER] After cleanup: '{intent}'")

    if intent not in ("SQL", "METADATA", "DOCUMENT", "GENERAL"):
        print(f"[CLASSIFIER] '{intent}' not in valid list, defaulting to SQL")
        intent = "SQL"

    if intent == "DOCUMENT" and not doc_context:
        print(f"[CLASSIFIER] DOCUMENT but no context, falling back to GENERAL")
        intent = "GENERAL"

    print(f"[CLASSIFIER] Final intent: {intent}")
    return {"intent": intent}

# Router

def route_question(state: AgentState) -> str:
    intent = state.get("intent", "SQL")
    if intent == "METADATA":
        return "metadata_answer"
    if intent == "GENERAL":
        return "general_answer"
    return "retrieve_schema"


# GENERAL branch

@traceable(name="general_answer", run_type="chain")
def general_answer(state: AgentState) -> dict:
    response = llm.invoke([
        SystemMessage(content="""/no_think
You are a SQL analytics agent. You help users query databases using natural language.
Answer briefly and helpfully."""),
        HumanMessage(content=state["question"])
    ])

    answer = response.content.strip()
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    return {"answer": answer}


# METADATA branch

@traceable(name="metadata_answer", run_type="chain")
def metadata_answer(state: AgentState) -> dict:
    try:
        conn = sqlite3.connect("test_db.sqlite")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = []

        for (name,) in cursor.fetchall():
            cursor.execute(f'PRAGMA table_info("{name}")')
            cols = [f"{row[1]} ({row[2]})" for row in cursor.fetchall()]
            cursor.execute(f'SELECT COUNT(*) FROM "{name}"')
            count = cursor.fetchone()[0]
            tables.append(f"{name} ({count} rows): {', '.join(cols)}")

        conn.close()
        table_info = "\n".join(tables)

        response = llm.invoke([
            SystemMessage(content="""/no_think
You are a helpful SQL agent. Given table metadata, answer the user's question about what data is available. Be concise."""),
            HumanMessage(content=f"User question: {state['question']}\n\nAvailable tables:\n{table_info}")
        ])

        answer = response.content.strip()
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        return {"answer": answer}

    except Exception as e:
        return {"answer": f"Could not fetch table info: {str(e)}"}


# SQL Node 1: Retrieve schema from ChromaDB

@traceable(name="retrieve_schema", run_type="chain")
def retrieve_schema(state: AgentState) -> dict:
    question = state["question"]
    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    schemas = []
    for meta in results["metadatas"][0]:
        schemas.append(meta["original_context"])

    return {"schema": "\n\n".join(schemas)}


# SQL Node 2: LLM generates SQL query (uses doc_context if available)

@traceable(name="generate_sql", run_type="chain")
def generate_sql(state: AgentState) -> dict:
    question = state["question"]
    schema = state["schema"]
    doc_context = state.get("doc_context", "")
    error = state.get("error_message", "")

    prompt = f"Schema:\n{schema}\n\nQuestion: {question}"

    if doc_context:
        prompt += f"\n\nBusiness rules from uploaded documents (use these for calculations and logic):\n{doc_context}"

    if error:
        prompt += f"\n\nPrevious attempt failed with error: {error}\nFix the query."

    response = llm.invoke([
        SystemMessage(content="""/no_think
You are a SQL query generator. Given a table schema and a question, generate ONLY a SQL SELECT query.
If business rules are provided, use them for calculations (formulas, definitions, date ranges).
No explanations. No markdown. Just the SQL query."""),
        HumanMessage(content=prompt)
    ])

    sql = response.content.strip()
    sql = re.sub(r'<think>.*?</think>', '', sql, flags=re.DOTALL).strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()

    return {"generated_sql": sql}


# SQL Node 3: Validate SQL (code only)

@traceable(name="validate_sql", run_type="chain")
def validate_sql(state: AgentState) -> dict:
    query = state["generated_sql"]
    upper = query.upper().strip()

    if not upper.startswith("SELECT"):
        return {
            "is_valid": False,
            "error_message": "Only SELECT statements allowed",
            "retry_count": state.get("retry_count", 0) + 1
        }

    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE"]
    for kw in dangerous:
        if kw in upper:
            return {
                "is_valid": False,
                "error_message": f"Dangerous keyword {kw} found",
                "retry_count": state.get("retry_count", 0) + 1
            }

    if "FROM" not in upper:
        return {
            "is_valid": False,
            "error_message": "Missing FROM clause",
            "retry_count": state.get("retry_count", 0) + 1
        }

    return {"is_valid": True, "error_message": ""}


# SQL Node 4: Execute SQL (no retry)

@traceable(name="execute_sql", run_type="chain")
def execute_sql(state: AgentState) -> dict:
    query = state["generated_sql"]

    try:
        conn = sqlite3.connect("test_db.sqlite")
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()

        if not results:
            return {"query_results": "No results found."}

        return {"query_results": f"Columns: {columns}\nResults: {results[:20]}"}

    except Exception as e:
        return {"query_results": f"SQL execution failed: {str(e)}"}


# SQL Node 5: Format answer using LLM

@traceable(name="format_answer", run_type="chain")
def format_answer(state: AgentState) -> dict:
    question = state["question"]
    sql = state["generated_sql"]
    results = state.get("query_results", "")
    doc_context = state.get("doc_context", "")

    if not results:
        return {"answer": "Could not get results."}

    prompt = f"Question: {question}\nSQL: {sql}\nResults: {results}\n\nGive a short natural language answer."

    if doc_context:
        prompt += f"\n\nBusiness context (use if relevant to explain the answer):\n{doc_context}"

    response = llm.invoke([
        SystemMessage(content="/no_think\nYou summarize SQL results into short natural language answers. No markdown."),
        HumanMessage(content=prompt)
    ])

    answer = response.content.strip()
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    return {"answer": answer}


# Validation routing

def after_validate(state: AgentState) -> str:
    if state["is_valid"]:
        return "execute_sql"
    if state.get("retry_count", 0) >= 1:
        return "format_answer"
    return "generate_sql"


# Build graph

sql_workflow = StateGraph(AgentState)

sql_workflow.add_node("classify_question", classify_question)
sql_workflow.add_node("general_answer", general_answer)
sql_workflow.add_node("metadata_answer", metadata_answer)
sql_workflow.add_node("retrieve_schema", retrieve_schema)
sql_workflow.add_node("generate_sql", generate_sql)
sql_workflow.add_node("validate_sql", validate_sql)
sql_workflow.add_node("execute_sql", execute_sql)
sql_workflow.add_node("format_answer", format_answer)

sql_workflow.add_edge(START, "classify_question")

sql_workflow.add_conditional_edges(
    "classify_question",
    route_question,
    {
        "general_answer": "general_answer",
        "metadata_answer": "metadata_answer",
        "retrieve_schema": "retrieve_schema"
    }
)

sql_workflow.add_edge("general_answer", END)
sql_workflow.add_edge("metadata_answer", END)
sql_workflow.add_edge("retrieve_schema", "generate_sql")
sql_workflow.add_edge("generate_sql", "validate_sql")

sql_workflow.add_conditional_edges(
    "validate_sql",
    after_validate,
    {
        "execute_sql": "execute_sql",
        "generate_sql": "generate_sql",
        "format_answer": "format_answer"
    }
)

sql_workflow.add_edge("execute_sql", "format_answer")
sql_workflow.add_edge("format_answer", END)

app = sql_workflow.compile()


# Run

@traceable(name="ask_agent", run_type="chain")
def ask(question: str, doc_context: str = "", verbose: bool = True) -> str:
    result = app.invoke({
        "question": question,
        "intent": "",
        "doc_context": doc_context,
        "schema": "",
        "generated_sql": "",
        "is_valid": False,
        "error_message": "",
        "retry_count": 0,
        "query_results": "",
        "answer": ""
    })

    if verbose:
        print(f"\nIntent: {result['intent']}")
        if result.get("doc_context"):
            print(f"Doc context: {result['doc_context'][:100]}...")
        if result.get("schema"):
            print(f"Schema: {result['schema'][:100]}...")
        if result.get("generated_sql"):
            print(f"SQL: {result['generated_sql']}")
        if result.get("query_results"):
            print(f"Results: {result['query_results'][:200]}")
        print(f"Answer: {result['answer']}")

    return result["answer"]


if __name__ == "__main__":
    questions = [
        "Hello, what can you do?",
        "What tables do you have access to?",
        "Which city has highest population?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"USER: {q}")
        print(f"{'='*60}")
        answer = ask(q)