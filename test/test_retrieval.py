from core.data_loading import schema_to_questions
from core.rag_retrieval import embed_model, collection

# Test Retrieval
# ─────────────────────────────────────────────
print("\n\n" + "="*60)
print("TESTING RAG RETRIEVAL")
print("="*60)

def retrieve_schema(question: str, top_k: int = 3):
    """Given a natural language question, find the most relevant table schema."""
    query_embedding = embed_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    return results


# Test with actual questions from the dataset
# test_questions = [
#    # Pick questions from different schemas
#    dataset["train"][0]["question"],
#    dataset["train"][1000]["question"],
#    dataset["train"][5000]["question"],
#    # Also try a custom question
#]

#for question in test_questions:
#    print(f"\n{'─'*60}")
#    print(f"Question: {question}")

#    results = retrieve_schema(question, top_k=5)

#    for rank, (doc, meta, distance) in enumerate(zip(
#        results["documents"][0],
#        results["metadatas"][0],
#        results["distances"][0], 
#    )):
#        print(f"\n  Rank {rank+1} (distance: {distance:.4f}):")
#        print(f"  Embedded : {doc[:100]}...")
#        print(f"  Schema   : {meta['original_context'][:100]}...")
#        print(f"  Tables   : {meta['table_names']}")

print("\n\n" + "="*60)
print("RETRIEVAL ACCURACY TEST")
print("="*60)


flat_list = [
    {
        "context": schema,
        **q
    }
    for schema, questions in schema_to_questions.items()
    for q in questions
]

import random
random.seed(42)
samples = 80
test_indices = random.sample(range(len(flat_list)), samples)  # test random examples

correct_at_1 = 0
correct_at_3 = 0

for idx in test_indices:
    ex = flat_list[idx]
    expected_context = ex["context"]

    results = retrieve_schema(ex["question"], top_k=5)
    retrieved_contexts = [m["original_context"] for m in results["metadatas"][0]]

    if retrieved_contexts[0] == expected_context:
        correct_at_1 += 1
    if expected_context in retrieved_contexts:
        correct_at_3 += 1

print(f"\nRetrieval Accuracy @ 1: {correct_at_1}/{samples} ({correct_at_1/samples*100:.1f}%)")
print(f"Retrieval Accuracy @ 3: {correct_at_3}/{samples} ({correct_at_3/samples*100:.1f}%)")
print(f"\nThis tells you: when a user asks a question, how often does RAG")
print(f"find the correct table schema in the top 1 (or top 3) results.")