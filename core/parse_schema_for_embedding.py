# In this file, this is dataset based parsing for b-mc2/sql-create-context dataset.

#import data_loading
import re


def parse_schema_for_embedding(context: str) -> dict:
    """
    Convert CREATE TABLE statement into:
    - embeddable text (for vector search)
    - metadata (for filtering/passing to LLM)
    """
    # Extract table names and columns using regex.
    tables = re.findall(
        r'CREATE TABLE (\w+)\s*\(([^)]+)\)',
        context,
        re.IGNORECASE
    )

    table_descriptions = []     
    table_names = []

    for table_name, columns_str in tables:
        table_names.append(table_name)
        # Parse individual columns
        columns = []
        for col in columns_str.split(','):
            col = col.strip()
            if col:
                parts = col.split()
                if len(parts) >= 2:
                    columns.append(f"{parts[0]} ({parts[1]})")
                elif len(parts) == 1:
                    columns.append(parts[0])

        desc = f"Table: {table_name} | Columns: {', '.join(columns)}"
        table_descriptions.append(desc)

    # This is what gets embedded — semantic meaning of the table
    embedding_text = " | ".join(table_descriptions)

    return {
        "embedding_text": embedding_text,
        "table_names": table_names,
        "original_context": context,   # this goes to LLM as-is
        "num_tables": len(tables)
    }


# Test parsing
""" print("\n--- Parsed Schema Examples ---")

for schema in data_loading.unique_schemas[:3]:
    parsed = parse_schema_for_embedding(schema)
    print(f"\nOriginal : {schema[:120]}...")
    print(f"Embedded : {parsed['embedding_text']}")
    print(f"Tables   : {parsed['table_names']}")
 """