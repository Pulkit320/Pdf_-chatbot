"""
Pytest unit tests for the PDF reader ingestion module.
To run these tests, execute:
    ./venv/bin/pytest tests/test_pdf_reader.py
"""

import pytest
from unittest.mock import patch, MagicMock
from src.ingestion.pdf_reader import extract_text, ocr_fallback


def test_extract_text_file_not_found():
    # Verify that FileNotFoundError is raised if path does not exist
    with pytest.raises(FileNotFoundError) as exc_info:
        extract_text("non_existent_document.pdf")
    assert "was not found" in str(exc_info.value)


@patch("os.path.exists")
@patch("pdfplumber.open")
def test_extract_text_digital_success(mock_pdfplumber_open, mock_exists):
    # Mock file exists check to pass
    mock_exists.return_value = True

    # Mock pdfplumber context manager and document structure
    mock_pdf = MagicMock()
    
    # Mock page 1
    mock_page1 = MagicMock()
    mock_page1.page_number = 1
    mock_page1.extract_text.return_value = "Digital syllabus page 1."
    
    # Mock page 2 (with leading/trailing spaces)
    mock_page2 = MagicMock()
    mock_page2.page_number = 2
    mock_page2.extract_text.return_value = "   Digital syllabus page 2 contents.   "
    
    mock_pdf.pages = [mock_page1, mock_page2]
    mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf
    
    result = extract_text("dummy_syllabus.pdf")
    
    # Assertions
    assert len(result) == 2
    assert result[0]["page_number"] == 1
    assert result[0]["text"] == "Digital syllabus page 1."
    assert result[0]["is_empty_or_scanned"] is False
    
    assert result[1]["page_number"] == 2
    # Check whitespace stripping
    assert result[1]["text"] == "Digital syllabus page 2 contents."
    assert result[1]["is_empty_or_scanned"] is False


@patch("os.path.exists")
@patch("src.ingestion.pdf_reader.ocr_fallback")
@patch("pdfplumber.open")
def test_extract_text_scanned_page_fallback(mock_pdfplumber_open, mock_ocr_fallback, mock_exists):
    # Mock exists check to pass
    mock_exists.return_value = True

    # Mock a PDF containing a page where pdfplumber returns empty text (scanned image page)
    mock_pdf = MagicMock()
    mock_page = MagicMock()
    mock_page.page_number = 3
    mock_page.extract_text.return_value = "   "  # triggers fallback
    
    mock_pdf.pages = [mock_page]
    mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf
    
    # Mock successful OCR result
    mock_ocr_fallback.return_value = "Extracted OCR text payload."
    
    result = extract_text("scanned_document.pdf")
    
    assert len(result) == 1
    assert result[0]["page_number"] == 3
    assert result[0]["text"] == "Extracted OCR text payload."
    assert result[0]["is_empty_or_scanned"] is False
    
    # Verify fallback was invoked on page 3
    mock_ocr_fallback.assert_called_once_with("scanned_document.pdf", 3)


@patch("os.path.exists")
@patch("pdfplumber.open")
def test_extract_text_corrupted_pdf_handling(mock_pdfplumber_open, mock_exists):
    # Mock exists check to pass
    mock_exists.return_value = True

    # Mock pdfplumber raising open exception
    mock_pdfplumber_open.side_effect = Exception("Format mismatch or corruption")
    
    # Verify it is re-raised as a ValueError
    with pytest.raises(ValueError) as exc_info:
        extract_text("corrupted_notes.pdf")
    assert "Failed to parse PDF file" in str(exc_info.value)

