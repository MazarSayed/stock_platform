"""Market Analysis RAG tool for market analysis instructions.

This module provides LangChain tools for market analysis:
- market_analysis_rag_tool: Searches market analysis instructions using RAG
- web_search_tool: Performs web searches for real-time market information

These tools are used by the Market Insights Agent to provide comprehensive
market analysis and investment advice.
"""

from langchain_core.tools import tool
from ..rag.retriever import Retriever
from langchain_community.tools import TavilySearchResults

@tool
def market_analysis_rag_tool(query: str) -> str:
    """
    Search market analysis instructions and internet for stock market analysis and investment advice.
    
    Args:
        query: Question about market analysis, technical analysis, trading strategies, stocks, or investments
    
    Returns:
        Guidance based on market analysis instructions and real-time market information
    """
    # Get guidance from stored documents
    retriever = Retriever(document_type="market_analysis", top_k=3)
    return retriever.retrieve_with_context(query)



@tool
def web_search_tool(query: str, max_results: int = 5) -> str:
    """
    Search the internet for stock market analysis, investment advice, and market information.
    
    Args:
        query: Search query about stocks, markets, investments, or analysis
        max_results: Maximum number of search results to return (default: 5)
    
    Returns:
        Formatted search results with titles, URLs, and content snippets
    """
    try:
        # Initialize Tavily search tool
        search = TavilySearchResults(max_results=max_results)
        
        # Perform search
        results = search.invoke({"query": query})
        
        if not results:
            return "No search results found."
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "")
            
            formatted_results.append(
                f"[Result {i}]\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content[:300]}...\n"
            )
        
        return "\n".join(formatted_results)
    
    except Exception as e:
        return f"Web search error: {str(e)}"


