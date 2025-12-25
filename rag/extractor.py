"""Document extraction and chunking module."""

import re
import yaml
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf
from io import BytesIO


class DocumentExtractor:
    """Handles document extraction and chunking."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize document extractor.
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            separators: List of separators for splitting text
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", " ", ""]
        )
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a document file.
        
        Args:
            file_path: Path to the document file
        
        Returns:
            Extracted text content
        """
        file_path_obj = Path(file_path)
        
        if file_path_obj.suffix.lower() == '.pdf':
            return self._extract_pdf_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path_obj.suffix}")
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Extracted text content
        """
        
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = pypdf.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
        
        Returns:
            List of text chunks
        """
        return self.text_splitter.split_text(text)
    
    def extract_and_chunk(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from document and chunk it.
        
        Args:
            file_path: Path to the document file
        
        Returns:
            List of chunks with metadata
        """
        text = self.extract_text(file_path)
        chunks = self.chunk_text(text)
        filename = Path(file_path).name
        
        return [
            {
                "text": chunk,
                "source": filename,
                "chunk_index": i,
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def extract_qa_pairs_from_text(self, text: str, source_filename: str = "faq.pdf") -> List[Dict[str, Any]]:
        """
        Extract Q&A pairs from FAQ text. Each Q&A becomes a chunk.
        
        Args:
            text: Text containing Q&A pairs
            source_filename: Filename to use as source
        
        Returns:
            List of chunks with Q&A pairs
        """
        qa_chunks = []
        
        # Pattern to match Q&A pairs: Q: question A: answer
        pattern = r'(?:Q|Question)[:\s]+(.*?)(?:A|Answer)[:\s]+(.*?)(?=(?:Q|Question)[:\s]|$)'
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        
        for i, match in enumerate(matches):
            question = match.group(1).strip()
            answer = match.group(2).strip()
            
            if question and answer:
                qa_text = f"Q: {question}\nA: {answer}"
                qa_chunks.append({
                    "text": qa_text,
                    "source": source_filename,
                    "chunk_index": i
                })
        
        # If no Q&A pattern found, try alternative patterns
        if not qa_chunks:
            pattern2 = r'(\d+)[\.\)]\s*(.*?)(?=\d+[\.\)]|$)'
            matches2 = re.finditer(pattern2, text, re.DOTALL | re.MULTILINE)
            
            for i, match in enumerate(matches2):
                content = match.group(2).strip()
                if content and len(content) > 20:
                    qa_chunks.append({
                        "text": content,
                        "source": source_filename,
                        "chunk_index": i
                    })
        
        return qa_chunks
    
    def extract_and_chunk_pdf(self, pdf_path: str, use_qa_extraction: bool = False) -> List[Dict[str, Any]]:
        """
        Extract and chunk a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            use_qa_extraction: If True, extract Q&A pairs instead of recursive chunking
        
        Returns:
            List of chunks with metadata
        """
        text = self._extract_pdf_text(pdf_path)
        filename = Path(pdf_path).name
        
        if use_qa_extraction:
            return self.extract_qa_pairs_from_text(text, filename)
        else:
            chunks = self.chunk_text(text)
            return [
                {"text": chunk, "source": filename, "chunk_index": i}
                for i, chunk in enumerate(chunks)
            ]
    
    def extract_faq_from_pdf(self, pdf_path: str, output_yaml_path: str):
        """
        Extract FAQ from PDF and save to YAML format.
        
        Args:
            pdf_path: Path to FAQ PDF file
            output_yaml_path: Path to save the YAML file
        """
        # Extract text from PDF
        text = self._extract_pdf_text(pdf_path)
        
        # Extract all Q&A pairs
        qa_pairs = self._extract_qa_pairs(text)
        
        # Simple flat structure - no categories
        faq_data = {"faqs": qa_pairs}
        
        # Save to YAML
        output_path = Path(output_yaml_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(faq_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    def extract_qa_pairs(self, text: str) -> List[Dict[str, str]]:
        """
        Extract Q&A pairs from text (simple pattern matching).
        
        Args:
            text: Text containing Q&A pairs
        
        Returns:
            List of dictionaries with 'content' field
        """
        qa_pairs = []
        
        # Simple pattern: Q: followed by A: (handles markdown formatting)
        pattern = r'Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)'
        
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            question = match.group(1).strip()
            answer = match.group(2).strip()       
            
            if question and answer:
                content = f"{question} \\n {answer}"
                qa_pairs.append({"content": content})
        
        return qa_pairs
    
    def save_chunks_to_file(self, chunks: List[Dict[str, Any]], output_path: str):
        """Save chunks to YAML file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump({"chunks": chunks}, f, default_flow_style=False, allow_unicode=True)
    
    def extract_faq_with_vision(self, pdf_path: str, api_key: str = None) -> List[Dict[str, Any]]:
        """
        Extract FAQ Q&A pairs from PDF using GPT-4o vision.
        
        Args:
            pdf_path: Path to FAQ PDF file
            api_key: OpenAI API key (optional, can use env var)
        
        Returns:
            List of chunks with Q&A pairs
        """
        from openai import OpenAI
        from pdf2image import convert_from_path
        
        # Import Pydantic models
        from models.models import QAPair, FAQExtraction
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Convert PDF to images
        print(f"Converting PDF to images: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=200)
        image_bytes_list = []
        for i, image in enumerate(images):
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            image_bytes_list.append(buffer.getvalue())
            print(f"  Page {i+1}: {len(image_bytes_list[-1])} bytes")
        
        # Extract Q&A from each page with context
        all_qa_pairs = []
        previous_chunk = None
        
        for i, image_bytes in enumerate(image_bytes_list):
            print(f"  Extracting Q&A from page {i + 1}...")
            
            # Convert to base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Build context
            context_text = "Extract all question-answer pairs from this FAQ page. Return them as structured Q&A pairs with 'question' and 'answer' fields."
            if previous_chunk:
                prev_q = previous_chunk.get('question', '')
                prev_a = previous_chunk.get('answer', '')
                context_text += f"\n\nIMPORTANT CONTEXT: The previous page ended with:\nQ: {prev_q}\nA: {prev_a}\n\n"
                context_text += "If you see an incomplete Q&A that continues from the previous page, merge them appropriately."
            
            # Extract with GPT-4o vision
            response = client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting FAQ question-answer pairs from PDF documents. Extract all Q&A pairs from the image. Look for questions marked with Q: or **Q:** and answers marked with A: or **A:**. If a Q&A pair spans across pages, merge them appropriately. Return them in structured format."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": context_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ],
                response_format=FAQExtraction
            )
            
            # Parse response
            qa_pairs = []
            if response.choices[0].message.parsed and response.choices[0].message.parsed.qa_pairs:
                qa_pairs = [{"question": qa.question, "answer": qa.answer} 
                           for qa in response.choices[0].message.parsed.qa_pairs]
            
            print(f"    Extracted {len(qa_pairs)} Q&A pairs")
            
            # Remove incomplete previous chunk if merged
            if previous_chunk and qa_pairs and all_qa_pairs:
                first_qa = qa_pairs[0]
                questions_match = first_qa['question'].lower().strip() == previous_chunk['question'].lower().strip()
                prev_incomplete = not previous_chunk['question'].endswith('?') or len(previous_chunk['answer'].split()) < 5
                
                if questions_match or prev_incomplete:
                    all_qa_pairs.pop()
                    print(f"    Removed incomplete Q&A (merged)")
            
            all_qa_pairs.extend(qa_pairs)
            previous_chunk = qa_pairs[-1] if qa_pairs else None
        
        # Convert to chunk format
        filename = Path(pdf_path).name
        return [
            {
                "text": f"Q: {qa['question']}\nA: {qa['answer']}",
                "source": filename,
                "chunk_index": i
            }
            for i, qa in enumerate(all_qa_pairs)
        ]
    

