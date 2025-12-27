"""Retrieval mechanism module.

This module provides the Retriever class that handles document retrieval
using the vector store. It supports filtering by document type and formats
retrieved documents as context strings for use in RAG applications.
"""

from typing import List, Dict, Any, Optional
from .vectorstore import VectorStore
from .extractor import DocumentExtractor


class Retriever:
    """Handles document retrieval using vector store."""
    
    def __init__(
        self,
        vectorstore: Optional[VectorStore] = None,
        top_k: int = 5,
        document_type: Optional[str] = None
    ):
        """
        Initialize retriever.
        
        Args:
            vectorstore: VectorStore instance (will load if None)
            top_k: Number of top results to return
            document_type: Filter by document type (e.g., 'faq_rag', 'market_analysis')
        """
        self.vectorstore = vectorstore or VectorStore()
        if vectorstore is None:
            try:
                self.vectorstore.load()
            except FileNotFoundError:
                pass  # Index not created yet
        self.top_k = top_k
        self.document_type = document_type
    
    def retrieve(self, query: str, k: Optional[int] = None, document_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            k: Number of results (defaults to top_k)
            document_type: Filter by document type (overrides instance default)
        
        Returns:
            List of retrieved documents with metadata
        """
        k = k or self.top_k
        doc_type = document_type or self.document_type
        return self.vectorstore.search(query, k=k, document_type=doc_type)
    
    def retrieve_with_context(self, query: str, k: Optional[int] = None, document_type: Optional[str] = None) -> str:
        """
        Retrieve documents and format as context string.
        
        Args:
            query: Query text
            k: Number of results
            document_type: Filter by document type (overrides instance default)
        
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, k, document_type)
        
        if not results:
            return "No relevant information found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            doc_name = result['metadata'].get('document_name', 'Unknown')
            context_parts.append(
                f"[{doc_name}]\n"
                f"{result['text']}\n"
            )
        
        return "\n".join(context_parts)

