from datasets import load_dataset, load_from_disk
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from parse_schema_for_embedding import parse_schema_for_embedding
import re
import os
import sqlite3
from collections import defaultdict


DOCS_PATH = "./documents"

def load_documents_to_chromadb(embedding_model, collection):
    """Load text/PDF documents into ChromaDB alongside schemas."""
    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)
        print(f"Created {DOCS_PATH}/ folder. Add your .txt or .pdf files there.")
        return

    files = [f for f in os.listdir(DOCS_PATH) if f.endswith((".txt", ".md", ".pdf"))]
    if not files:
        print("No documents found in documents/ folder.")
        return

    for filename in files:
        filepath = os.path.join(DOCS_PATH, filename)

        # Parse
        if filename.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            text = "".join([page.extract_text() or "" for page in reader.pages])
        else:
            with open(filepath, "r") as f:
                text = f.read()

        # Chunk
        words = text.split()
        chunks = []
        for i in range(0, len(words), 250):
            chunk = " ".join(words[i:i + 300])
            if chunk.strip():
                chunks.append(chunk.strip())

        # Embed and store
        for i, chunk in enumerate(chunks):
            embedding = embedding_model.encode([chunk]).tolist()
            collection.add(
                ids=[f"doc_{filename}_{i}"],
                documents=[chunk],
                embeddings=embedding,
                metadatas=[{
                    "type": "business_rule",
                    "source": filename,
                    "original_context": chunk
                }]
            )

        print(f"  Stored {len(chunks)} chunks from {filename}")




# 1. Download dataset once, cache locally


DATASET_CACHE_PATH = "./dataset_cache"

if os.path.exists(DATASET_CACHE_PATH):
    print("📁 Loading dataset from local cache...")
    dataset = load_from_disk(DATASET_CACHE_PATH)
else:
    print("⬇️  Downloading dataset (first time only)...")
    dataset = load_dataset("b-mc2/sql-create-context", split="train")
    dataset.save_to_disk(DATASET_CACHE_PATH)
    print(f"💾 Saved to {DATASET_CACHE_PATH}")



# 2. Filter generic table names


def is_meaningful_table(context: str) -> bool:
    """Filter out generic table names like table_name_15, table_12434380_1"""
    table_names = re.findall(r'CREATE TABLE (\w+)', context, re.IGNORECASE)
    
    for name in table_names:
        if re.match(r'^table_name_\d+$', name, re.IGNORECASE):
            return False
        if re.match(r'^table_\d+$', name, re.IGNORECASE):
            return False
        if re.match(r'^table_\d+_\d+$', name, re.IGNORECASE):
            return False
    
    return True


# 3. Collect schemas and QA pairs


schema_to_questions = defaultdict(list)
count = 0
for sample in dataset:
    if is_meaningful_table(sample["context"]):
        count += 1
        schema_to_questions[sample["context"]].append({
            "index": count,
            "question": sample["question"],
            "answer": sample["answer"]
        })

unique_schemas = list(schema_to_questions.keys())
print(f"Unique meaningful schemas: {len(unique_schemas)}")




# 4. Select schemas (single-table, unique names, 2+ columns)


def select_schemas(schemas: list, limit: int = 10) -> list:
    """Pick schemas with unique table names, single table, 2+ columns."""
    seen_tables = set()
    selected = []
    
    for schema in schemas:
        table_names = re.findall(r'CREATE TABLE (\w+)', schema, re.IGNORECASE)
        
        if len(table_names) != 1:
            continue
        
        if table_names[0].lower() in seen_tables:
            continue
        
        columns = re.findall(r'CREATE TABLE \w+\s*\(([^)]+)\)', schema, re.IGNORECASE)
        if columns:
            col_count = len([c for c in columns[0].split(',') if c.strip()])
            if col_count < 2:
                continue
        
        seen_tables.add(table_names[0].lower())
        selected.append(schema)
        
        if len(selected) == limit:
            break
    
    return selected

selected_schemas = select_schemas(unique_schemas, limit=10)
print(f"Selected schemas: {len(selected_schemas)}")


# 5. ChromaDB data loading (now uses selected_schemas)


def data_loading(embedding_model, collection):
    """Load selected schemas into ChromaDB."""
    BATCH_SIZE = 1
    for start in range(0, len(selected_schemas), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(selected_schemas))
        batch_schemas = selected_schemas[start:end]

        ids = []
        documents = []
        metadatas = []

        for i, schema in enumerate(batch_schemas):
            parsed = parse_schema_for_embedding(schema)
            idx = start + i

            ids.append(f"schema_{idx}")
            documents.append(parsed["embedding_text"])
            metadatas.append({
                "original_context": schema,
                "table_names": ", ".join(parsed["table_names"]),
                "num_tables": parsed["num_tables"]
            })

        embeddings = embedding_model.encode(documents).tolist()
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"  Stored {end}/{len(selected_schemas)} schemas")



# 5. SQLite creation with LLM-generated data


DB_PATH = "test_db.sqlite"

def clean_inserts(text: str) -> list[str]:
    """Extract INSERT statements from LLM output."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'```(?:sql)?', '', text)
    text = text.replace('```', '')
    inserts = re.findall(r'(INSERT\s+INTO\s+.+?;)', text, re.IGNORECASE | re.DOTALL)
    return inserts


def create_sqlite_db(schemas: list, db_path: str = DB_PATH):
    """Create SQLite tables and populate with LLM-generated data."""
    
    llm = ChatOllama(model="qwen3:4b", temperature=0.7, num_predict=2048)

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a data generator. Given a CREATE TABLE statement, "
         "generate exactly 10 INSERT statements with realistic, diverse data. "
         "Output ONLY the INSERT statements, nothing else. "
         "No explanations, no markdown, no code blocks."),
        ("human", "{schema}")
    ])

    chain = prompt | llm | StrOutputParser()

    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    created = 0

    for i, schema in enumerate(schemas):
        print(f"\n  [{i+1}/{len(schemas)}] {schema[:80]}...")

        # Create table(s)
        try:
            for statement in schema.split(';'):
                statement = statement.strip()
                if statement:
                    cursor.execute(statement)
        except Exception as e:
            print(f"    ❌ CREATE failed: {e}")
            continue

        # Generate INSERT statements via LLM
        try:
            raw_output = chain.invoke({"schema": schema})
            inserts = clean_inserts(raw_output)
            
            success = 0
            for stmt in inserts:
                try:
                    cursor.execute(stmt)
                    success += 1
                except Exception as e:
                    continue

            print(f"    ✅ {success} rows inserted")
            created += 1

        except Exception as e:
            print(f"    ❌ LLM failed: {e}")
            continue

    conn.commit()
    conn.close()
    print(f"\n✅ Database: {db_path} | Tables: {created}/{len(schemas)}")



# 6. Run only when called directly


if __name__ == "__main__":

    print(f"\n📦 Creating SQLite with LLM-generated data...")
    if os.path.exists(DB_PATH):
        print(f"✅ Database already exists: {DB_PATH}, skipping creation.")
    else:
        print(f"\n📦 Creating SQLite with {len(selected_schemas)} tables (LLM-generated data)...")
        create_sqlite_db(selected_schemas)