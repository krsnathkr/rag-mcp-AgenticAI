import streamlit as st
import asyncio
import os
import time

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_core.documents import Document
from PyPDF2 import PdfReader

import search
import rag

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Page configuration
st.set_page_config(
    page_title="Knowbl",
    page_icon="🔍",
    layout="wide"
)

# Sidebar: Global settings
with st.sidebar:
    st.title("⚙️ Settings")
    num_results = st.slider("Web results to fetch", 1, 20, 5)
    rag_k = st.slider("RAG chunks per query", 1, 10, 3)
    st.markdown("---")
    st.write("Built by Krishna Thakar with ❤️")

# Caching document vectorstore for PDF/Text Q&A
@st.cache_resource(show_spinner=False)
def get_vectorstore_from_docs(_docs):
    return asyncio.run(rag.create_rag_from_documents(_docs))

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Instantiate a streaming Chat LLM
chat_llm = ChatOpenAI(
    model_name="gpt-4.1-nano",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    temperature=0.0,
    streaming=True
)

# Async function: web search + RAG
async def perform_search_and_rag(query, num_results, rag_k):
    # Web search for summary + raw results
    formatted_summary, raw_results = await search.search_web(query, num_results)

    # Prepare search results snippets
    search_results = []
    for r in raw_results:
        snippet = getattr(r, "summary", getattr(r, "snippet", "No summary available."))
        search_results.append((r.url, snippet))

    # Fetch and prepare docs for RAG
    docs = []
    for r in raw_results:
        docs.extend(await search.get_web_content(r.url))

    # Build RAG vectorstore and retrieve top-k chunks
    vectorstore = await rag.create_rag_from_documents(docs)
    rag_docs = await rag.search_rag(query, vectorstore, k=rag_k)

    return formatted_summary, search_results, rag_docs

# Handle a new user message in chat tab
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

# Create the two tabs
tab_chat, tab_doc = st.tabs([
    "💬 Web Search with MCP",
    "📄 Document Q&A"
])

# --- Tab 1: Conversational Web + RAG Chat ---
with tab_chat:
    st.title("Knowbl 🧠")
    st.subheader("Conversational Web + RAG Chat")

    # User input
    user_input = st.chat_input("Ask me anything…")
    if user_input:
        handle_user_message(user_input)

    # Render chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            data = msg["content"]
            ac = st.chat_message("assistant")
            ac.markdown(f"**Summary:** {data['summary']}")

            # Split layout: sources & RAG chunks
            cols = ac.columns([3, 1])
            with cols[1]:
                with st.expander("🔗 Sources", expanded=False):
                    for url, snippet in data['search_results']:
                        st.markdown(f"- [{url}]({url})")
                with st.expander("📦 RAG Chunks", expanded=False):
                    for i, d in enumerate(data['rag_docs'], 1):
                        snippet = d.page_content.strip().replace("\n", " ")
                        st.markdown(f"{i}. {snippet[:200]}…")

# --- Tab 2: Document Q&A with Summarization + Sources ---
with tab_doc:
    st.title("Knowbl 🧠")
    st.subheader("Document Q&A with Summarization")

    uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])
    if not uploaded_file:
        st.info("Please upload a document to get started.")
        st.stop()

    # Read uploaded document
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = "\n\n".join(page.extract_text() for page in reader.pages)
    else:
        text = uploaded_file.read().decode("utf-8")

    docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]

    # Index the document
    with st.spinner("📦 Indexing document…"):
        vectorstore = get_vectorstore_from_docs(docs)

    # Prepare QA chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": rag_k})
    qa = RetrievalQA.from_chain_type(
        llm=chat_llm,
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True
    )

    # Ask document question
    question = st.text_input("Ask a question about your document…")
    if question:
        with st.spinner("🤖 Generating answer…"):
            try:
                result = qa({"query": question})
                answer = result["result"]
                source_docs = result["source_documents"]
            except Exception as e:
                st.warning(f"Couldn't generate a summary: {e}")
                answer = None
                source_docs = asyncio.run(rag.search_rag(question, vectorstore, k=rag_k))

        # Display answer
        if answer:
            st.markdown("**Answer:**")
            st.write(answer)
        else:
            st.markdown("**Answer:** _(see raw sources below)_")

        # Collapsible sources
        with st.expander("Sources"):
            for i, d in enumerate(source_docs, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(d.page_content)
