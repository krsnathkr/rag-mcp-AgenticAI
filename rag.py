# rag.py

import os, asyncio
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings
import search  # depends on search.py from Commit 1

# Async: Build vectorstore from web URLs
async def create_rag(links: list[str]) -> FAISS:
    model_name = os.getenv("MODEL", "text-embedding-ada-002")

    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        chunk_size=64,
    )

    # Fetch content from each URL
    documents = []
    tasks = [search.get_web_content(url) for url in links]
    results = await asyncio.gather(*tasks)
    for docs in results:
        documents.extend(docs)

    # Split content into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(documents)

    return FAISS.from_documents(documents=split_docs, embedding=embeddings)

# Sync: Build vectorstore from already-available Document list
async def create_rag_from_documents(documents: list[Document]) -> FAISS:
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

# Query vectorstore and return top-k matching chunks
async def search_rag(query: str, vectorstore: FAISS, k: int = 3) -> list[Document]:
    return vectorstore.similarity_search(query, k=k)
