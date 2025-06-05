# main.py

import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_core.documents import Document
import search
import rag
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Page config
st.set_page_config(page_title="Knowbl", page_icon="🔍", layout="wide")

# Sidebar: Config sliders
with st.sidebar:
    st.title("⚙️ Settings")
    num_results = st.slider("Web results to fetch", 1, 20, 5)
    rag_k = st.slider("RAG chunks per query", 1, 10, 3)
    st.markdown("---")
    st.write("Built by Krishna Thakar with ❤️")

# Store chat history in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize ChatOpenAI
chat_llm = ChatOpenAI(
    model_name="gpt-4.1-nano",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    temperature=0.0,
    streaming=True
)

# Async RAG search helper
async def perform_search_and_rag(query, num_results, rag_k):
    formatted_summary, raw_results = await search.search_web(query, num_results)

    # Get snippet & content
    search_results = []
    for r in raw_results:
        snippet = getattr(r, "summary", getattr(r, "snippet", "No summary available."))
        search_results.append((r.url, snippet))

    docs = []
    for r in raw_results:
        docs.extend(await search.get_web_content(r.url))

    vectorstore = await rag.create_rag_from_documents(docs)
    rag_docs = await rag.search_rag(query, vectorstore, k=rag_k)

    return formatted_summary, search_results, rag_docs

# Handle a user message
def handle_user_message(query):
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.spinner("🔍 Searching & indexing…"):
        formatted_summary, search_results, rag_docs = asyncio.run(
            perform_search_and_rag(query, num_results, rag_k)
        )
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": {
            "summary": formatted_summary,
            "search_results": search_results,
            "rag_docs": rag_docs
        }
    })

# Render interface
st.title("Knowbl 🧠")
st.subheader("Conversational Web + RAG Chat")

# Input field
user_input = st.chat_input("Ask me anything…")
if user_input:
    handle_user_message(user_input)

# Render history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        data = msg["content"]
        ac = st.chat_message("assistant")
        ac.markdown(f"**Summary:** {data['summary']}")

        cols = ac.columns([3, 1])
        with cols[1]:
            with st.expander("🔗 Sources", expanded=False):
                for url, snippet in data['search_results']:
                    st.markdown(f"- [{url}]({url})")
            with st.expander("📦 RAG Chunks", expanded=False):
                for i, d in enumerate(data['rag_docs'], 1):
                    snippet = d.page_content.strip().replace("\n", " ")
                    st.markdown(f"{i}. {snippet[:200]}…")
