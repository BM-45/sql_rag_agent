
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas import EvaluationDataset, SingleTurnSample
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import sqlite3
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()


# Setup - same models as agent
embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("table_schemas")

llm = ChatOllama(model="qwen3:4b", temperature=0, num_predict=512)

DB_PATH = "test_db.sqlite"


# Build test questions from your actual tables
def build_test_set():
    """Create test questions from actual database tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    test_cases = []
    for table in tables:
        cursor.execute(f'PRAGMA table_info("{table}")')
        columns = [row[1] for row in cursor.fetchall()]
        cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
        count = cursor.fetchone()[0]

        if count == 0:
            continue

        # Generate a simple test question per table
        col = columns[0]
        cursor.execute(f'SELECT "{col}" FROM "{table}" LIMIT 1')
        sample = cursor.fetchone()

        test_cases.append({
            "question": f"How many rows are in the {table} table?",
            "reference": f"There are {count} rows in the {table} table.",
            "reference_sql": f"SELECT COUNT(*) FROM {table}"
        })

        if len(columns) >= 2:
            test_cases.append({
                "question": f"Show all {columns[1]} from {table}",
                "reference": f"List of {columns[1]} values from {table}",
                "reference_sql": f"SELECT {columns[1]} FROM {table}"
            })

    conn.close()
    print(f"Built {len(test_cases)} test cases from {len(tables)} tables")
    return test_cases


# Run retrieval + re-ranking (same logic as agent)
def retrieve_with_rerank(question: str) -> str:
    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=10)

    candidates = results["metadatas"][0]
    pairs = [[question, meta["original_context"]] for meta in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top_schemas = [meta["original_context"] for _, meta in ranked[:3]]

    return "\n\n".join(top_schemas)


# Generate SQL and get answer (same logic as agent)
def generate_and_execute(question: str, schema: str) -> dict:
    # Generate SQL
    response = llm.invoke([
        SystemMessage(content="""/no_think
You are a SQL query generator. Given a table schema and a question, generate ONLY a SQL SELECT query.
No explanations. No markdown. Just the SQL query."""),
        HumanMessage(content=f"Schema:\n{schema}\n\nQuestion: {question}")
    ])

    sql = response.content.strip()
    sql = re.sub(r'<think>.*?</think>', '', sql, flags=re.DOTALL).strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()

    # Execute
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()
        query_results = f"Columns: {columns}\nResults: {results[:20]}"
    except Exception as e:
        query_results = f"SQL execution failed: {str(e)}"

    # Format answer
    response = llm.invoke([
        SystemMessage(content="/no_think\nYou summarize SQL results into short natural language answers. No markdown."),
        HumanMessage(content=f"Question: {question}\nSQL: {sql}\nResults: {query_results}\n\nGive a short natural language answer.")
    ])

    answer = response.content.strip()
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

    return {
        "sql": sql,
        "query_results": query_results,
        "answer": answer
    }


# Run evaluation
def run_evaluation():
    test_cases = build_test_set()

    if not test_cases:
        print("No test cases generated. Make sure test_db.sqlite has data.")
        return

    samples = []
    print(f"\nRunning {len(test_cases)} test cases...\n")

    for i, tc in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {tc['question']}")

        # Retrieve + re-rank
        retrieved_context = retrieve_with_rerank(tc["question"])

        # Generate + execute
        result = generate_and_execute(tc["question"], retrieved_context)

        print(f"  SQL: {result['sql'][:80]}")
        print(f"  Answer: {result['answer'][:80]}")

        samples.append(SingleTurnSample(
            user_input=tc["question"],
            response=result["answer"],
            reference=tc["reference"],
            retrieved_contexts=[retrieved_context]
        ))

    print(f"\nEvaluating with RAGAS metrics...")
    dataset = EvaluationDataset(samples=samples)

    try:
        results = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )

        print(f"\n{'='*50}")
        print("RAGAS EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Faithfulness:      {results['faithfulness']:.4f}")
        print(f"Answer Relevancy:  {results['answer_relevancy']:.4f}")
        print(f"Context Precision: {results['context_precision']:.4f}")
        print(f"Context Recall:    {results['context_recall']:.4f}")
        print(f"{'='*50}")

        # Save results
        results_dict = {
            "faithfulness": float(results['faithfulness']),
            "answer_relevancy": float(results['answer_relevancy']),
            "context_precision": float(results['context_precision']),
            "context_recall": float(results['context_recall']),
            "num_test_cases": len(test_cases)
        }

        with open("ragas_results.json", "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to ragas_results.json")

    except Exception as e:
        print(f"\nRAGAS evaluation error: {e}")
        print("Falling back to manual evaluation...\n")

        # Manual evaluation if RAGAS fails (needs OpenAI key)
        correct = 0
        total = len(samples)
        for i, (sample, tc) in enumerate(zip(samples, test_cases)):
            has_answer = "no results" not in sample.response.lower() and "failed" not in sample.response.lower()
            if has_answer:
                correct += 1
            print(f"  [{i+1}] {'PASS' if has_answer else 'FAIL'} | {tc['question'][:50]}")

        print(f"\n{'='*50}")
        print(f"MANUAL EVALUATION: {correct}/{total} ({correct/total*100:.1f}%) generated valid answers")
        print(f"{'='*50}")


if __name__ == "__main__":
    run_evaluation()