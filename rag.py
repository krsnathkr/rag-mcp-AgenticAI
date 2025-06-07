# rag.py

import os, asyncio
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings

# ─── Pick ONE of these embedders ──────────────────────────────
# 1) OpenAI (fast + high quality, needs internet + API key)
from langchain.embeddings import OpenAIEmbeddings

# 2) Hugging Face all-MiniLM (local, no rate limits)
# from langchain.embeddings import HuggingFaceEmbeddings

# 3) Ollama + mistral-small (local, no rate limits)
# from langchain_ollama import OllamaEmbeddings
# ────────────────────────────────────────────────────────────────

async def create_rag(links: list[str]) -> FAISS:
    # ─── configure your embedding model ─────────────────────────
    model_name = os.getenv("MODEL", "text-embedding-ada-002")

    # ┌── OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        chunk_size=64,
    )

    # ┌── HuggingFaceEmbeddings
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # ┌── OllamaEmbeddings
    # embeddings = OllamaEmbeddings(model=model_name)

    # ─── fetch & chunk ───────────────────────────────────────────
    documents = []
    tasks = [search.get_web_content(url) for url in links]
    results = await asyncio.gather(*tasks)
    for docs in results:
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(documents)

    # ─── build & return FAISS index ─────────────────────────────
    return FAISS.from_documents(documents=split_docs, embedding=embeddings)


async def create_rag_from_documents(documents: list[Document]) -> FAISS:
    # same embedder as above
    model_name = os.getenv("MODEL", "text-embedding-ada-002")

    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        chunk_size=64,
    )


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(documents)
    return FAISS.from_documents(documents=split_docs, embedding=embeddings)

async def search_rag(query: str, vectorstore: FAISS, k: int = 3,) -> list[Document]:
    return vectorstore.similarity_search(query, k=k)
