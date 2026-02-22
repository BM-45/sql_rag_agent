from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from data_loading import dataset

import random

import os
from dotenv import load_dotenv

# get values from .env file
load_dotenv()

llm = os.getenv("MODEL")

model = ChatOllama(model = llm, temperature = 0)


# This step is necessary to give to llm, because we need to define what it needs to do
# So need to give prompts to tell what it has to do.

prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """
    You are a sql expert. Given a database schema and a question,
    generate ONLY the SQL query that answers the question.
     
    Rules:
    - Output ONLY the SQL query, nothing else
    - No explanations, no markdown, no code blocks
    - Use exact column and table names from the schema
    """),

    ("human", 
    """
    Schema: {context}
    Question: {question}
    SQL Query: 
    """
    )
])

# Now chaining
chain = prompt | model | StrOutputParser()

# Sampling the data
numbers = random.sample(range(0, 78576), 6)


# Validation: Invoking and testing the models answer
count = 0
for id in numbers:
    groundTruth = dataset["train"][id]["answer"]
    generated = chain.invoke({"context": dataset["train"][id]["context"], "question": dataset["train"][id]["question"]})
    print(f"generated answer from llm: {generated}")
    print(f"Ground Truth answer: { groundTruth}")
    print(f"This is for dataset index for {id}")
    print("-------------------------------------------------------------------------------")
    if generated == groundTruth:
        count += 1
print(count)