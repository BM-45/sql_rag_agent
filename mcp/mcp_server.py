"""
MCP Server — SQL RAG Agent Tools
==================================
Exposes search_schema, execute_sql tools to Claude Desktop.

pip install mcp
python mcp_server.py

Then add to Claude Desktop config:
  ~/Library/Application Support/Claude/claude_desktop_config.json
"""

from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
import chromadb
import sqlite3
import re
import os

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────

mcp = FastMCP("SQL RAG Agent")

embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
PROJECT_DIR = "/Users/bmt/sql_rag_mcp_agent"

chroma_client = chromadb.PersistentClient(path=os.path.join(PROJECT_DIR, "chroma_db"))

DB_PATH = os.path.join(PROJECT_DIR, "test_db.sqlite")

try:
    collection = chroma_client.get_collection("table_schemas")
    print(f"✅ ChromaDB loaded: {collection.count()} schemas")
except:
    collection = None
    print("⚠️  ChromaDB collection not found. Run rag_retrieval.py first.")



# ─────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────

@mcp.tool()
def search_schema(question: str) -> str:
    """Search the database for relevant table schemas based on a natural language question.
    Use this to find which tables and columns are available before writing SQL."""

    if collection is None:
        return "ERROR: ChromaDB not initialized. Run rag_retrieval.py first."

    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    schemas = []
    for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
        schemas.append(f"[similarity: {1-distance:.4f}] {meta['original_context']}")

    return "\n\n".join(schemas) if schemas else "No matching schemas found."


@mcp.tool()
def execute_sql(query: str) -> str:
    """Validate and execute a SQL query against the SQLite database.
    Only SELECT statements are allowed. Returns the query results."""

    upper = query.upper().strip()

    # Validation
    if not upper.startswith("SELECT"):
        return f"ERROR: Only SELECT allowed. Your query starts with: {upper.split()[0]}"

    for kw in ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE"]:
        if kw in upper:
            return f"ERROR: Dangerous keyword '{kw}' detected."

    # Execute
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()

        if not results:
            return "Query returned no results."

        return f"Columns: {columns}\nResults: {results[:20]}"

    except Exception as e:
        return f"SQL ERROR: {str(e)}. Fix the query and try again."


@mcp.tool()
def list_tables() -> str:
    """List all tables in the database with their columns and row counts."""

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")

        output = []
        for (name,) in cursor.fetchall():
            cursor.execute(f'PRAGMA table_info("{name}")')
            cols = [f"{row[1]} ({row[2]})" for row in cursor.fetchall()]
            cursor.execute(f'SELECT COUNT(*) FROM "{name}"')
            count = cursor.fetchone()[0]
            output.append(f"{name} ({count} rows): {', '.join(cols)}")

        conn.close()
        return "\n".join(output) if output else "No tables found."

    except Exception as e:
        return f"ERROR: {str(e)}"


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()