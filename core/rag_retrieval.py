import chromadb
from sentence_transformers import SentenceTransformer
from data_loading import data_loading

# These are always available for import
embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Try to get existing collection, don't recreate
try:
    collection = chroma_client.get_collection("table_schemas")
except:
    collection = chroma_client.create_collection(
        name="table_schemas",
        metadata={"description": "SQL table schemas for RAG retrieval", "hnsw:space": "cosine"}
    )

if __name__ == "__main__":
    print("Loading data into ChromaDB...")
    chroma_client.delete_collection("table_schemas")
    collection = chroma_client.create_collection(
        name="table_schemas",
        metadata={"description": "SQL table schemas for RAG retrieval", "hnsw:space": "cosine"}
    )
    data_loading(embed_model, collection)
    print(f"Done! Stored {collection.count()} schemas")