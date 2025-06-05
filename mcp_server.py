# mcp_server.py

import asyncio
from mcp.server.fastmcp import FastMCP
import rag
import search
import logging
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define the MCP agent
mcp = FastMCP(
    name="web_search", 
    version="1.0.0",
    description="Web search capability using Exa API and RAG to find relevant information from the internet in real-time."
)

@mcp.tool()
async def search_web_tool(query: str) -> str:
    logger.info(f"Searching web for query: {query}")
    formatted_results, raw_results = await search.search_web(query)
    
    if not raw_results:
        return "No search results found."
    
    urls = [result.url for result in raw_results if hasattr(result, 'url')]
    if not urls:
        return "No valid URLs found in search results."
        
    vectorstore = await rag.create_rag(urls)
    rag_results = await rag.search_rag(query, vectorstore)
    
    full_results = f"{formatted_results}\n\n### RAG Results:\n\n"
    full_results += '\n---\n'.join(doc.page_content for doc in rag_results)
    
    return full_results

@mcp.tool()
async def get_web_content_tool(url: str) -> str:
    try:
        documents = await asyncio.wait_for(search.get_web_content(url), timeout=15.0)
        if documents:
            return '\n\n'.join([doc.page_content for doc in documents])
        return "Unable to retrieve web content."
    except asyncio.TimeoutError:
        return "Timeout occurred while fetching web content. Please try again later."
    except Exception as e:
        return f"An error occurred while fetching web content: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="sse")
