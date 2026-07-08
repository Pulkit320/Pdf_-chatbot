"""
Pytest unit tests for the text chunking module.
To run these tests, execute:
    ./venv/bin/pytest tests/test_chunker.py
"""

import pytest
from src.embeddings.chunker import chunk_text


def test_chunk_text_basic():
    # Test split offset and overlap logic
    # String is 15 chars: "abcdefghijklmno"
    # Chunk size: 10, Overlap: 5
    # Expect: ["abcdefghij", "fghijklmno"]
    text = "abcdefghijklmno"
    result = chunk_text(text, chunk_size=10, overlap=5)
    assert result == ["abcdefghij", "fghijklmno"]


def test_chunk_text_shorter_than_size():
    # Verifies a short string returns as a single chunk
    text = "hello"
    result = chunk_text(text, chunk_size=20, overlap=5)
    assert result == ["hello"]


def test_chunk_text_empty_inputs():
    # Verifies blank inputs yield empty lists
    assert chunk_text("", chunk_size=10, overlap=2) == []
    assert chunk_text("   ", chunk_size=10, overlap=2) == []
    assert chunk_text(None, chunk_size=10, overlap=2) == []


def test_chunk_text_invalid_parameters():
    # Verifies input boundary checks throw ValueError
    with pytest.raises(ValueError):
        chunk_text("test content", chunk_size=0, overlap=2)
    with pytest.raises(ValueError):
        chunk_text("test content", chunk_size=-5, overlap=2)
    with pytest.raises(ValueError):
        chunk_text("test content", chunk_size=10, overlap=-1)
    with pytest.raises(ValueError):
        chunk_text("test content", chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("test content", chunk_size=10, overlap=12)


def test_chunk_text_whitespace_strip():
    # Verifies trailing white-spaces are correctly cleaned prior to chunking
    text = "   leading and trailing spaces should be stripped   "
    result = chunk_text(text, chunk_size=500, overlap=50)
    assert len(result) == 1
    assert result[0] == "leading and trailing spaces should be stripped"
