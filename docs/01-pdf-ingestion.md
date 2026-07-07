# Phase 1: PDF Text Ingestion

In this guide, we will implement the first step of our PDF Chatbot pipeline: **PDF Text Ingestion**. We will build a module that takes a PDF file path as input, reads the document page-by-page, extracts the raw text content, and prepares structured data containing the extracted text and page numbers.

---

## 1. Why `pdfplumber`?

To parse text from PDFs, Python offers several libraries. For this project, we chose **`pdfplumber`** over other options like `pypdf` or `pdfminer.six` because of its:
1. **Layout Precision:** It is exceptionally good at extracting text in visual reading order, avoiding scrambled sentences when pages have columns or tables.
2. **Metadata Access:** It automatically provides clean, 1-based page numbering (`page.page_number`), making it simple to track the exact page of each text snippet.
3. **Robustness:** It handles a wide variety of PDF encodings and formats without crashing.

---

## 2. Code Implementation

The ingestion logic is located in [pdf_reader.py](file:///home/pulkit/projects/pdf_chatbot/src/ingestion/pdf_reader.py). Below is the complete code showing the integrated OCR fallback mechanism.

```python
"""
This module handles extracting text from PDF documents.
It serves as Phase 1 of our PDF Chatbot pipeline.
"""

import os
import pdfplumber

from pdf2image import convert_from_path
import pytesseract

def ocr_fallback(pdf_path: str, page_number: int) -> str:
    """
    Converts a single PDF page to an image and runs Tesseract OCR on it.
    
    Why we use this helper:
    If a PDF page has no digital text layer (e.g. it is a scanned image or photo), 
    we need to convert the page representation into a graphic image first, then 
    pass it to an OCR tool to detect individual characters and words.
    """
    try:
        # Convert only the specific scanned page to an image.
        # pdf2image page numbering is 1-based for first_page and last_page parameters.
        pages = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
        if pages:
            # Run OCR on the retrieved page image
            ocr_text = pytesseract.image_to_string(pages[0])
            return ocr_text.strip()
    except Exception as e:
        print(f"OCR failed for page {page_number}: {e}")
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

                # Why run OCR Fallback:
                # If the digital layer yielded no characters, the page might be scanned.
                # We attempt to run OCR fallback to recover text.
                if is_empty_or_scanned:
                    ocr_text = ocr_fallback(pdf_path, page_num)
                    if ocr_text:
                        clean_text = ocr_text
                        is_empty_or_scanned = False  # Text has been recovered successfully!

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
```

---

## 3. Block-by-Block Explanation

Let's break down the logic to understand how it functions step-by-step:

### A. Guarding Against Missing Files
```python
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"The PDF file at '{pdf_path}' was not found.")
```
* **What it does:** Verifies that the file path actually points to an existing file before invoking the extraction library.
* **Why it matters:** If we omit this check, `pdfplumber.open` will fail with a library-specific error. Raising `FileNotFoundError` makes it immediately obvious to any caller what went wrong.

### B. Safe Resource Management
```python
with pdfplumber.open(pdf_path) as pdf:
```
* **What it does:** Uses a Python `with` statement (a context manager) to open the PDF.
* **Why it matters:** PDFs are binary files that require operating system file handles. If a crash happens midway through reading, the context manager guarantees the file is properly closed, preventing memory leaks and locked files.

### C. Page Iteration and Text Extraction
```python
for page in pdf.pages:
    page_num = page.page_number
    raw_text = page.extract_text()
```
* **What it does:** Loops through every page of the document sequentially. `page.page_number` gets the page sequence starting at `1`. `page.extract_text()` attempts to parse the digital text layout on that page.
* **Why it matters:** Keeping the exact page number associated with the text is essential for citation provenance. When the user asks a question, we want our chatbot to respond: *"According to Page 12, the formula is..."*

### D. Text Sanitization & Flagging Scanned Pages
```python
clean_text = raw_text.strip() if raw_text else ""
is_empty_or_scanned = len(clean_text) == 0
```
* **What it does:** Safely strips leading and trailing whitespaces from the extracted text. If `extract_text` returned `None` (which happens if a page has no text content), we default to an empty string. We then evaluate if `clean_text` has a length of zero and store that result in `is_empty_or_scanned`.
* **Why it matters:** PDF pages that contain images (scanned document pages) do not have a digital text layer, meaning `extract_text()` returns `None`. By flagging these pages with a boolean `is_empty_or_scanned` instead of throwing an exception or crashing, we handle empty or image-only pages gracefully.

### E. OCR Fallback Trigger
```python
if is_empty_or_scanned:
    ocr_text = ocr_fallback(pdf_path, page_num)
    if ocr_text:
        clean_text = ocr_text
        is_empty_or_scanned = False
```
* **What it does:** If the digital layer returns no text (`is_empty_or_scanned = True`), the engine falls back to calling our `ocr_fallback` helper function. If OCR successfully recovers characters on the page, the text is saved, and we reset the `is_empty_or_scanned` flag to `False` (indicating text has been successfully found).

---

## 4. Handling Scanned PDFs with OCR

What happens if a user uploads a PDF that is a scan of a printed page? 
In this case, the PDF is essentially a collection of images. Our implementation reads it without crashing, notices that the digital extraction yielded empty strings, and activates the **Optical Character Recognition (OCR)** fallback.

### How OCR Works
OCR algorithms analyze the shapes of dark and light areas on an image to recognize letters and synthesize them into computer-readable text. The industry standard open-source OCR engine is **Tesseract** (maintained by Google).

### Setting Up System Dependencies
To enable the OCR system on your local machine, you must install the Tesseract binary and its corresponding libraries:

1. **Install System OCR (Tesseract):**
   * **macOS:** `brew install tesseract`
   * **Linux:** `sudo apt install tesseract-ocr`
   * **Windows:** Download the binary installer.

2. **Install Python Wrappers:**
   Run the following command in your virtual environment:
   ```bash
   pip install pytesseract pdf2image
   ```

### OCR Fallback Function
The `ocr_fallback` function is called directly within the loop when digital parsing fails:

```python
def ocr_fallback(pdf_path: str, page_number: int) -> str:
    # Convert only the specific scanned page to an image
    # Note: page_number is 1-based, convert_from_path takes first_page/last_page
    pages = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    if pages:
        # Run OCR on the page image
        ocr_text = pytesseract.image_to_string(pages[0])
        return ocr_text.strip()
    return ""
```

By integrating this check, we ensure that even scanned documents can be converted to searchable text before moving to the embedding stage.

---

## Learning Outcomes

By reading and building this ingestion phase, you are now able to:
1. **Implement** digital PDF text extraction page-by-page using the `pdfplumber` library.
2. **Defend** the importance of maintaining page numbers throughout the ingestion pipeline for citation provenance.
3. **Debug** and handle empty or scanned PDF pages using clean fallback checks.
4. **Architect** and integrate an OCR (Optical Character Recognition) fallback system using Tesseract and `pdf2image` to process image-only scanned PDFs.
