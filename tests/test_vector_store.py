"""
Pytest unit tests for the PostgreSQL + pgvector VectorStore component.
To run these tests, execute:
    ./venv/bin/pytest tests/test_vector_store.py
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from src.retrieval.vector_store import VectorStore


@patch("src.retrieval.vector_store.psycopg2.connect")
@patch.dict(os.environ, {"DATABASE_URL": "postgresql://test_user:test_pass@localhost/mock_db"})
def test_vector_store_initialization(mock_connect):
    # Setup mocks for database cursor and connection context managers
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_connect.return_value = mock_conn
    
    # Initialize VectorStore - triggers schema checks
    store = VectorStore()
    
    # Verify psycopg2 connection was made
    mock_connect.assert_called_with("postgresql://test_user:test_pass@localhost/mock_db")
    
    # Verify CREATE EXTENSION and CREATE TABLE queries were executed
    calls = [call[0][0] for call in mock_cur.execute.call_args_list]
    assert any("CREATE EXTENSION" in q for q in calls)
    assert any("CREATE TABLE IF NOT EXISTS chunks" in q for q in calls)
    
    # Verify transaction commit
    mock_conn.commit.assert_called()


@patch("src.retrieval.vector_store.psycopg2.connect")
@patch("src.retrieval.vector_store.execute_values")
@patch.dict(os.environ, {"DATABASE_URL": "postgresql://mock_db"})
def test_store_chunks_bulk_insert(mock_execute_values, mock_connect):
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_connect.return_value = mock_conn
    
    store = VectorStore()
    mock_execute_values.reset_mock()
    mock_conn.commit.reset_mock()
    
    chunks = ["chunk one", "chunk two"]
    embeddings = [[0.1] * 768, [0.2] * 768]
    metadata = {
        "pdf_id": "test_syllabus.pdf",
        "page_numbers": [3, 4]
    }
    
    store.store_chunks(chunks, embeddings, metadata)
    
    # Verify bulk insert executed
    mock_execute_values.assert_called_once()
    args = mock_execute_values.call_args[0]
    
    # Check SQL query and formatted records list
    assert "INSERT INTO chunks" in args[1]
    inserted_rows = args[2]
    assert len(inserted_rows) == 2
    assert inserted_rows[0][0] == "test_syllabus.pdf"
    assert inserted_rows[0][1] == 3
    assert inserted_rows[0][2] == "chunk one"
    assert inserted_rows[0][3].startswith("[0.1,0.1,")
    
    assert inserted_rows[1][0] == "test_syllabus.pdf"
    assert inserted_rows[1][1] == 4
    
    mock_conn.commit.assert_called_once()


@patch("src.retrieval.vector_store.psycopg2.connect")
@patch.dict(os.environ, {"DATABASE_URL": "postgresql://mock_db"})
def test_search_results_mapping(mock_connect):
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_connect.return_value = mock_conn
    
    # Mock cursor fetchall returning a row from cosine similarity operator <=> search
    mock_cur.fetchall.return_value = [
        (10, "syllabus.pdf", 2, "Grounded course data text.", 0.865)
    ]
    
    store = VectorStore()
    results = store.search([0.1] * 768, top_k=1)
    
    # Verify result dictionary structure
    assert len(results) == 1
    assert results[0]["id"] == 10
    assert results[0]["pdf_id"] == "syllabus.pdf"
    assert results[0]["page_number"] == 2
    assert results[0]["content"] == "Grounded course data text."
    assert results[0]["similarity"] == 0.865
