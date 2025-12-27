"""Extract and chunk PDFs using recursive chunking or vision-based extraction.

This script processes PDF documents and extracts text content, then chunks
them for vector storage. For FAQ documents, it uses GPT-4o vision for better
Q&A pair extraction. Outputs are saved as YAML files for vector store building.
"""

from pathlib import Path
import sys
import os
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.extractor import DocumentExtractor


def main():
    """Extract and chunk PDF documents."""
    load_dotenv()
    
    # Initialize extractor
    extractor = DocumentExtractor(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Use current folder structure
    base_path = Path(__file__).parent.parent / "data" / "documents"
    content_path = base_path / "content"
    chunks_path = base_path / "chunks"
    chunks_path.mkdir(parents=True, exist_ok=True)
    
    # PDF files to process from content folder
    pdf_files = [
        content_path / "faq.pdf",
        content_path / "user_guide.pdf",
        content_path / "market_analysis_instructions.pdf"
    ]
    
    print("Extracting and chunking PDFs...")
    
    # Process each PDF separately
    for pdf_file in pdf_files:
        if not pdf_file.exists():
            print(f"  Skipping {pdf_file.name} (not found)")
            continue
            
        print(f"\nProcessing: {pdf_file.name}")
        
        # Special handling for FAQ PDF
        if pdf_file.stem == "faq":
            # Use vision-based extraction for FAQ
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                print("  Using GPT-4o vision extraction...")
                chunks = extractor.extract_faq_with_vision(str(pdf_file), api_key)
                print(f"  Extracted {len(chunks)} Q&A pairs")
            else:
                # Fallback to text-based Q&A extraction
                print("  Using text-based Q&A extraction (vision API key not set)...")
                chunks = extractor.extract_and_chunk_pdf(str(pdf_file), use_qa_extraction=True)
                print(f"  Extracted {len(chunks)} Q&A pairs")
        else:
            # Regular chunking for other PDFs
            chunks = extractor.extract_and_chunk_pdf(str(pdf_file), use_qa_extraction=False)
            print(f"  Created {len(chunks)} chunks")
        
        # Save chunks with naming convention
        pdf_name = pdf_file.stem
        if pdf_name == "market_analysis_instructions":
            pdf_name = "market_analysis"
            output_filename = f"{pdf_name}_chunks.yaml"
        elif pdf_name == "faq":
            output_filename = "faq.yaml"
        else:
            output_filename = f"{pdf_name}_chunks.yaml"
        
        output_path = chunks_path / output_filename
        extractor.save_chunks_to_file(chunks, str(output_path))
        print(f"  Saved to: chunks/{output_filename}")
    
    print(f"\n✓ All chunks saved to chunks/ folder")


if __name__ == "__main__":
    main()
