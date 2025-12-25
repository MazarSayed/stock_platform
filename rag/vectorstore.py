"""FAISS vector store module."""

import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from .embedder import Embedder


class VectorStore:
    """Manages FAISS vector store operations."""
    
    def __init__(
        self,
        vectorstore_path: str = "data/vectorstore",
        embedder: Optional[Embedder] = None
    ):
        """
        Initialize vector store.
        
        Args:
            vectorstore_path: Path to store FAISS index
            embedder: Embedder instance for generating embeddings
        """
        self.vectorstore_path = Path(vectorstore_path)
        self.vectorstore_path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or Embedder()
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension = 1536
    
    def create_index(self):
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text chunks
            metadatas: List of metadata dictionaries for each chunk
        """
        if self.index is None:
            self.create_index()
        
        # Generate embeddings
        embeddings = self.embedder.embed_documents(texts)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Store metadata
        self.metadata.extend(metadatas)
    
    def save(self):
        """Save the FAISS index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save. Create or load an index first.")
        
        # Save FAISS index
        index_path = self.vectorstore_path / "faiss_index.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = self.vectorstore_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load(self):
        """Load the FAISS index and metadata from disk."""
        index_path = self.vectorstore_path / "faiss_index.index"
        metadata_path = self.vectorstore_path / "metadata.json"
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []
    
    def search(self, query: str, k: int = 5, document_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            document_type: Filter by document type (e.g., 'faq_rag', 'market_analysis')
        
        Returns:
            List of results with text, metadata, and score
        """
        if self.index is None:
            raise ValueError("No index loaded. Load or create an index first.")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Search (get more results if filtering)
        search_k = k * 3 if document_type else k
        distances, indices = self.index.search(query_vector, search_k)
        
        # Format results with optional filtering
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                metadata = self.metadata[idx]
                
                # Filter by document type if specified
                if document_type and metadata.get("document_type") != document_type:
                    continue
                
                results.append({
                    "text": metadata.get("text", ""),
                    "metadata": {k: v for k, v in metadata.items() if k != "text"},
                    "score": float(distances[0][i])
                })
                
                # Stop when we have enough results
                if len(results) >= k:
                    break
        
        return results
    
    def build_from_chunks(self, chunks_path: str, document_configs: List[Dict[str, str]]):
        """
        Build vector store from chunk YAML files.
        
        Args:
            chunks_path: Path to directory containing chunk YAML files
            document_configs: List of config dicts with keys:
                - chunk_file: Name of chunk YAML file
                - document_name: Name to assign to documents
                - document_type: Type of document (e.g., 'faq_rag', 'market_analysis')
        """
        chunks_dir = Path(chunks_path)
        self.create_index()
        
        for config in document_configs:
            chunk_file = chunks_dir / config['chunk_file']
            if not chunk_file.exists():
                print(f"   {config['chunk_file']} not found, skipping...")
                continue
            
            # Load chunks from YAML
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunks_data = yaml.safe_load(f)
            
            chunks = chunks_data.get('chunks', [])
            if not chunks:
                continue
            
            # Add metadata to chunks
            texts = []
            metadatas = []
            for chunk in chunks:
                metadata = {k: v for k, v in chunk.items() if k != "text"}
                metadata["document_name"] = config['document_name']
                metadata["document_type"] = config['document_type']
                texts.append(chunk["text"])
                metadatas.append(metadata)
            
            # Add to vector store
            self.add_documents(texts, metadatas)
            print(f"   Added {len(chunks)} chunks from {config['chunk_file']}")

