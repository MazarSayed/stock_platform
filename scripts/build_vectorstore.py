"""Build FAISS vector store from all documents."""

from pathlib import Path
import sys
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.vectorstore import VectorStore


def main():
    """Build vector store from chunks."""
    load_dotenv()
    base_path = Path(__file__).parent.parent / "data" / "documents"
    chunks_path = base_path / "chunks"
    vectorstore_path = Path(__file__).parent.parent / "data" / "vectorstore"
    
    # Initialize vector store
    vectorstore = VectorStore(vectorstore_path=str(vectorstore_path))
    
    # Define document configurations
    document_configs = [
        {
            "chunk_file": "faq.yaml",
            "document_name": "faq.yaml",
            "document_type": "faq_rag"
        },
        {
            "chunk_file": "user_guide_chunks.yaml",
            "document_name": "user_guide.pdf",
            "document_type": "faq_rag"
        },
        {
            "chunk_file": "market_analysis_chunks.yaml",
            "document_name": "market_analysis_instructions.pdf",
            "document_type": "market_analysis"
        }
    ]
    
    print("Building vector store...")
    
    # Build from chunks
    vectorstore.build_from_chunks(str(chunks_path), document_configs)
    
    # Save vector store
    vectorstore.save()
    
    print(f"\n✓ Vector store saved to: {vectorstore_path}")
    print(f"  Total documents: {len(vectorstore.metadata)}")


if __name__ == "__main__":
    main()
