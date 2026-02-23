# FastAPI Backend
# /query  - ask a question
# /upload - upload a business document (chunks stored in memory)
# /documents - list uploaded documents

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from core.agent import app as agent_app
from core.rag_retrieval import embed_model
from sentence_transformers import util
import uuid
import os
import re
import torch

api = FastAPI()

LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "sql-rag-agent")


# In-memory document store
# { filename: { "chunks": [...], "embeddings": tensor } }
document_store = {}


# Parsing

def parse_pdf(file_path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def parse_txt(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


def get_relevant_chunks(question: str, top_k: int = 3) -> str:
    if not document_store:
        print(f"[CHUNKS] document_store is empty. No documents uploaded.")
        return ""

    question_embedding = embed_model.encode(question, convert_to_tensor=True)

    all_chunks = []
    for filename, data in document_store.items():
        print(f"[CHUNKS] Searching in: {filename} ({len(data['chunks'])} chunks)")
        scores = util.cos_sim(question_embedding, data["embeddings"])[0]
        for i, score in enumerate(scores):
            all_chunks.append({"text": data["chunks"][i], "source": filename, "score": score.item()})

    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = all_chunks[:top_k]

    for c in top_chunks:
        print(f"[CHUNKS] Score: {c['score']:.4f} | {c['text'][:80]}...")

    # Threshold too high - lower it to 0.1 for general questions
    if not top_chunks or top_chunks[0]["score"] < 0.1:
        print(f"[CHUNKS] All scores below threshold, returning empty")
        return ""

    context = "\n\n".join([f"[{c['source']}] {c['text']}" for c in top_chunks])
    return context


# Routes

class Query(BaseModel):
    question: str


@api.post("/query")
def query(req: Query):
    run_id = str(uuid.uuid4())

    # Get relevant document chunks for this question
    doc_context = get_relevant_chunks(req.question)

    result = agent_app.invoke(
        {
            "question": req.question,
            "intent": "",
            "doc_context": doc_context,
            "schema": "",
            "generated_sql": "",
            "is_valid": False,
            "error_message": "",
            "retry_count": 0,
            "query_results": "",
            "answer": ""
        },
        config={"run_id": run_id}
    )

    trace_url = f"https://smith.langchain.com/o/default/projects/p/{LANGSMITH_PROJECT}/r/{run_id}"
    return {"answer": result.get("answer", "No response generated"), "trace_url": trace_url}


@api.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    print(f"\n[UPLOAD] Received file: {file.filename}")
    

    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        ext = file.filename.lower().split(".")[-1]
        if ext == "pdf":
            text = parse_pdf(temp_path)
        elif ext in ("txt", "md", "csv"):
            text = parse_txt(temp_path)
        else:
            os.remove(temp_path)
            return {"error": f"Unsupported file type: {ext}. Use PDF or TXT."}

        chunks = chunk_text(text)
        if not chunks:
            os.remove(temp_path)
            return {"error": "No text extracted from document."}

        # Embed and store in memory
        embeddings = embed_model.encode(chunks, convert_to_tensor=True)
        document_store[file.filename] = {
            "chunks": chunks,
            "embeddings": embeddings
        }
        
        print(f"[UPLOAD] Stored {len(chunks)} chunks for {file.filename}")
        print(f"[UPLOAD] document_store keys: {list(document_store.keys())}")

        os.remove(temp_path)
        return {
            "message": f"Uploaded {file.filename}",
            "chunks_stored": len(chunks)
        }

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return {"error": str(e)}


@api.get("/documents")
def list_documents():
    docs = [{"name": k, "chunks": len(v["chunks"])} for k, v in document_store.items()]
    return {"documents": docs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)