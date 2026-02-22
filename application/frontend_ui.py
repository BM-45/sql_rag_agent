import streamlit as st
import requests

st.title("SQL Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("trace_url"):
            st.markdown(f"[🔍 View Trace]({msg['trace_url']})")

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post("http://localhost:8000/query", json={"question": question}, timeout=300)
                data = resp.json()
                answer = data["answer"]
                trace_url = data.get("trace_url", "")
            except:
                answer = "Error: Backend not running. Run: python api.py"
                trace_url = ""
        st.write(answer)
        if trace_url:
            st.markdown(f"[🔍 View Trace]({trace_url})")

    st.session_state.messages.append({"role": "assistant", "content": answer, "trace_url": trace_url})