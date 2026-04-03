"""
NeuroSort AI: Robust Document Extraction Layer.
Provides high-fidelity text extraction with noise removal and multi-format support.
"""

import os
import re
import logging
from typing import Optional
from pypdf import PdfReader
from docx import Document

# Setup high-fidelity logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NeuroSortExtractor")

def clean_text(text: str) -> str:
    """
    User-provided cleaning logic to strip metadata and noise.
    """
    if not text: return ""
    text = re.sub(r'Browser:.*', '', text)
    text = re.sub(r'IP:.*', '', text)
    text = re.sub(r'Page \d+', '', text)
    return text.strip()

def clean_extracted_text(text: str) -> str:
    """
    Remove noise, redundant headers/footers, and irrelevant special characters.
    Ensures the text is suitable for transformer-based embedding.
    """
    if not text: return ""
    
    # 1. Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # 2. Remove common noise patterns (Control chars, non-printable)
    text = "".join(char for char in text if char.isprintable())
    
    # 3. Strip leading/trailing noise
    text = text.strip()
    
    return text

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts and cleans text from PDF using pypdf with robust error recovery."""
    try:
        reader = PdfReader(file_path)
        content = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                content.append(page_text)
        
        raw_text = "\n".join(content)
        cleaned_text = clean_extracted_text(clean_text(raw_text))
        logger.info(f"PDF Extraction: {len(cleaned_text)} chars from {file_path}")
        return cleaned_text
    except Exception as e:
        logger.error(f"Failed PDF extraction ({file_path}): {str(e)}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """Extracts and cleans text from DOCX using python-docx."""
    try:
        doc = Document(file_path)
        content = [para.text for para in doc.paragraphs if para.text.strip()]
        raw_text = "\n".join(content)
        cleaned_text = clean_extracted_text(clean_text(raw_text))
        logger.info(f"DOCX Extraction: {len(cleaned_text)} chars from {file_path}")
        return cleaned_text
    except Exception as e:
        logger.error(f"Failed DOCX extraction ({file_path}): {str(e)}")
        return ""

def extract_text_from_txt(file_path: str) -> str:
    """Extracts and cleans text from plain TXT files."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
            cleaned_text = clean_extracted_text(clean_text(raw_text))
            logger.info(f"TXT Extraction: {len(cleaned_text)} chars from {file_path}")
            return cleaned_text
    except Exception as e:
        logger.error(f"Failed TXT extraction ({file_path}): {str(e)}")
        return ""

def parse_document(file_path: str) -> Optional[str]:
    """
    Main entry point for NeuroSort extraction. 
    Routes to specific parsers and validates output.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
        
    _, ext = os.path.splitext(file_path.lower())
    
    if ext == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif ext == '.docx':
        text = extract_text_from_docx(file_path)
    elif ext in ['.txt', '.log', '.md']:
        text = extract_text_from_txt(file_path)
    else:
        logger.warning(f"Unsupported extension: {ext}")
        return None
        
    # Final Validation
    if not text or len(text.strip()) < 20:
        logger.warning(f"Insufficient text extracted from {file_path}")
        return None
        
    return text
