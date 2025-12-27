"""FAISS vector store module.

This module provides the VectorStore class for managing FAISS-based vector
storage. It handles creating, saving, loading, and searching vector embeddings
with support for document type filtering and metadata management.
"""

import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
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
        self.version_info: Dict[str, Any] = {}
    
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
    
    def get_embedding_model_name(self) -> str:
        """Get the embedding model name from embedder."""
        try:
            return getattr(self.embedder.embeddings, 'model', 'unknown')
        except:
            return 'unknown'
    
    def get_next_version(self) -> int:
        """Get next version number."""
        version_info_path = self.vectorstore_path / "version_info.json"
        if version_info_path.exists():
            with open(version_info_path, 'r') as f:
                existing_info = json.load(f)
                return existing_info.get("version", 0) + 1
        return 1
    
    def save(self):
        """Save the FAISS index and metadata to disk with version info."""
        if self.index is None:
            raise ValueError("No index to save. Create or load an index first.")
        
        # Save FAISS index
        index_path = self.vectorstore_path / "faiss_index.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = self.vectorstore_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save version info
        version = self.get_next_version()
        model_name = self.get_embedding_model_name()
        self.version_info = {
            "version": version,
            "embedding_model": model_name,
            "dimension": self.dimension,
            "document_count": len(self.metadata),
            "created_at": datetime.now().isoformat()
        }
        
        version_info_path = self.vectorstore_path / "version_info.json"
        with open(version_info_path, 'w') as f:
            json.dump(self.version_info, f, indent=2)
        
        print(f"✓ Saved vector store (version {version}, model: {model_name}, documents: {len(self.metadata)})")
    
    def load(self):
        """Load the FAISS index and metadata from disk."""
        index_path = self.vectorstore_path / "faiss_index.index"
        metadata_path = self.vectorstore_path / "metadata.json"
        version_info_path = self.vectorstore_path / "version_info.json"
        
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
        
        # Load version info
        if version_info_path.exists():
            with open(version_info_path, 'r') as f:
                self.version_info = json.load(f)
            
            # Check embedding model compatibility
            stored_model = self.version_info.get("embedding_model", "unknown")
            current_model = self.get_embedding_model_name()
            if stored_model != current_model:
                print(f"⚠ Warning: Stored model ({stored_model}) differs from current ({current_model})")
        else:
            self.version_info = {}
    
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
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get current version information."""
        return self.version_info.copy() if self.version_info else {}

