from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from core.agent import app as agent_app
import uuid
import os

api = FastAPI()

LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "sql-rag-agent")

class Query(BaseModel):
    question: str

@api.post("/query")
def query(req: Query):
    run_id = str(uuid.uuid4())
    
    result = agent_app.invoke(
        {"messages": [HumanMessage(content=req.question)]},
        config={"run_id": run_id}
    )
    
    answer = "No response generated"
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
            answer = msg.content
            break
    
    trace_url = f"https://smith.langchain.com/o/default/projects/p/{LANGSMITH_PROJECT}/r/{run_id}"
    
    return {"answer": answer, "trace_url": trace_url}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)