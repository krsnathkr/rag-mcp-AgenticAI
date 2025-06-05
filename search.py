import asyncio
import os
from dotenv import load_dotenv
from typing import List, Tuple
from exa_py import Exa
from langchain_core.documents import Document
import requests
from bs4 import BeautifulSoup

load_dotenv(override=True)

#exa api setup
exa_api_key = os.getenv("EXA_API_KEY", "")
exa = Exa(api_key=exa_api_key)

#configuration
websearch_config = {
    "parameters": {
        "default_num_results": 5,
        "include_domains": []
    }
}

async def search_web(query: str, num_results: int = None) -> Tuple[str, list]:
    """Search the web using Exa API and return formatted and raw results."""
    try:
        search_args = {
            "num_results": num_results or websearch_config["parameters"]["default_num_results"]
        }

        search_results = exa.search_and_contents(
            query,
            summary={"query": "Main points and key takeaways"},
            **search_args
        )

        formatted_results = format_search_results(search_results)
        return formatted_results, search_results.results
    except Exception as e:
        return f"An error occurred while searching with Exa: {e}", []

def format_search_results(search_results):
    if not search_results.results:
        return "No results found."

    markdown_results = "### Search Results:\n\n"
    for idx, result in enumerate(search_results.results, 1):
        title = getattr(result, 'title', 'No title')
        url = result.url
        published_date = f" (Published: {result.published_date})" if getattr(result, 'published_date', None) else ""
        markdown_results += f"**{idx}.** [{title}]({url}){published_date}\n"
        if getattr(result, 'summary', None):
            markdown_results += f"> **Summary:** {result.summary}\n\n"
        else:
            markdown_results += "\n"
    return markdown_results

async def get_web_content(url: str) -> List[Document]:
    """Scrape a URL using requests and BeautifulSoup and return Document."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return [Document(page_content=text, metadata={"source": url})]
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []
