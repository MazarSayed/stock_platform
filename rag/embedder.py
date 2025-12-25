"""Embedding generation module."""

import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings


class Embedder:
    """Handles embedding generation for text chunks."""
    
    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize embedder.
        
        Args:
            model_name: Name of the embedding model
            openai_api_key: OpenAI API key (if None, uses environment variable)
        """
        init_params = {"model": model_name}
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key:
            init_params["openai_api_key"] = api_key
        
        self.embeddings: Embeddings = OpenAIEmbeddings(**init_params)
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)

