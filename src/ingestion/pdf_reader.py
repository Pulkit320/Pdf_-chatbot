"""
This module handles extracting text from PDF documents.
It serves as Phase 1 of our PDF Chatbot pipeline.
"""

import os
import pdfplumber

from pdf2image import convert_from_path
import pytesseract

def ocr_fallback(pdf_path: str, page_number: int) -> str:
    # Convert only the specific scanned page to an image
    # Note: page_number is 1-based, convert_from_path takes first_page/last_page
    pages = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    if pages:
        # Run OCR on the page image
        ocr_text = pytesseract.image_to_string(pages[0])
        return ocr_text.strip()
    return ""

def extract_text(pdf_path: str) -> list[dict]:
    """
    Extracts text from a PDF file page-by-page.

    Why we keep page numbers:
    When the chatbot retrieves answers from the PDF, it needs to cite its sources.
    Keeping the 1-based page number with each text chunk allows us to tell the user
    exactly where the information came from (e.g., "Found on Page 4").

    Why we return a list of dictionaries:
    Returning a structured list [dict(page_number, text, is_empty_or_scanned)] separates 
    the text content from the metadata (page number) cleanly. This structural isolation 
    makes it easy for subsequent components (e.g. text chunking and embedding) to process 
    both text and metadata in downstream tasks.

    Why we flag empty/scanned pages:
    If a page extracts no text, it is either completely blank or it is a scanned image 
    (which requires OCR like Tesseract to read). Downstream components (like the vector database) 
    should not index empty pages, and frontend/orchestrator layers can warn the user that 
    the PDF might require OCR processing. Flagging instead of crashing ensures the pipeline 
    is robust to mix-and-match PDFs containing both digital and scanned pages.

    Args:
        pdf_path (str): The absolute or relative path to the PDF file.

    Returns:
        list[dict]: A list of dicts where each dict represents a page:
            {
                "page_number": int,
                "text": str,
                "is_empty_or_scanned": bool
            }

    Raises:
        FileNotFoundError: If the specified pdf_path does not exist.
        ValueError: If the file is not a PDF or is corrupted.
    """
    # Why check file existence explicitly:
    # Providing a clear FileNotFoundError up-front avoids generic library exceptions 
    # and simplifies troubleshooting for developers using this module.
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The PDF file at '{pdf_path}' was not found.")

    extracted_pages = []

    # Why use pdfplumber.open as a context manager:
    # Context managers automatically handle closing the file streams and releasing system 
    # memory resources even if an exception occurs during text extraction.
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Why iterate over pdf.pages:
            # pdfplumber parses the PDF into page objects, allowing page-level granularity 
            # and preserving document sequence naturally.
            for page in pdf.pages:
                page_num = page.page_number
                raw_text = page.extract_text()
                
                # Why strip whitespace:
                # Text extracted from PDFs often contains trailing newlines, tabs, and excess spacing 
                # that adds unnecessary token overhead for embeddings and LLM reasoning.
                clean_text = raw_text.strip() if raw_text else ""
                
                # Why we check if clean_text is empty:
                # Digital PDFs have text layers, whereas scanned PDFs or empty pages return empty strings or None.
                # A page is empty or scanned if no text was successfully extracted.
                is_empty_or_scanned = len(clean_text) == 0
                if is_empty_or_scanned:
                    ocr_text = ocr_fallback(pdf_path, page_num)
                    if ocr_text:
                        clean_text = ocr_text
                        is_empty_or_scanned = False
                extracted_pages.append({
                    "page_number": page_num,
                    "text": clean_text,
                    "is_empty_or_scanned": is_empty_or_scanned
                })
                
    except Exception as e:
        # Why catch generic exceptions here:
        # If pdfplumber encounters a corrupted file or an unsupported format, we want to 
        # wrap the error in a readable ValueError to prevent the application from crashing 
        # with obscure internal library tracebacks.
        raise ValueError(f"Failed to parse PDF file '{pdf_path}'. It might be corrupted. Details: {e}")

    return extracted_pages
