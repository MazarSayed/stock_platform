"""FAQ RAG tool for FAQ and user guide documents."""

from langchain_core.tools import tool
from rag.retriever import Retriever


@tool
def faq_rag_tool(query: str) -> str:
    """
    Search FAQ and user guide documents to answer questions.
    
    Args:
        query: User question about platform features, account management, or trading
    
    Returns:
        Answer based on FAQ and user guide content
    """
    retriever = Retriever(document_type="faq_rag", top_k=3)
    return retriever.retrieve_with_context(query)

