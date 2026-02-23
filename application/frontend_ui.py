# Streamlit Frontend - Chat + Document Upload

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("Analytics Agent")

# Sidebar: upload documents

with st.sidebar:
    st.header("Upload Business Documents")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt", "md", "csv"])

    if uploaded_file and st.button("Upload"):
        with st.spinner("Uploading..."):
            try:
                resp = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                    timeout=60
                )
                data = resp.json()
                if "error" in data:
                    st.error(data["error"])
                else:
                    st.success(f"{data['chunks_stored']} chunks stored from {uploaded_file.name}")
            except:
                st.error("Backend not running. Run: python api.py")

    st.markdown("---")
    st.header("Stored Documents")
    try:
        resp = requests.get(f"{API_URL}/documents", timeout=5)
        docs = resp.json().get("documents", [])
        if docs:
            for doc in docs:
                st.markdown(f"- **{doc['name']}** ({doc['chunks']} chunks)")
        else:
            st.caption("No documents uploaded yet")
    except:
        st.caption("Backend not connected")


# Chat

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("trace_url"):
            st.markdown(f"[View Trace]({msg['trace_url']})")

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"question": question},
                    timeout=300
                )
                data = resp.json()
                answer = data["answer"]
                trace_url = data.get("trace_url", "")
            except:
                answer = "Error: Backend not running. Run: python api.py"
                trace_url = ""
        st.write(answer)
        if trace_url:
            st.markdown(f"[View Trace]({trace_url})")

    st.session_state.messages.append({"role": "assistant", "content": answer, "trace_url": trace_url})