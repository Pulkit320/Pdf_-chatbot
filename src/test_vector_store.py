"""
Unit tests for the VectorStore component.
To run these tests, execute:
    ./venv/bin/python -m unittest src/test_vector_store.py
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from src.retrieval.vector_store import VectorStore

class TestVectorStore(unittest.TestCase):
    
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_database_url_env(self):
        # Verify that if DATABASE_URL is missing, creating a VectorStore raises ValueError
        with self.assertRaises(ValueError) as context:
            VectorStore()
        self.assertIn("DATABASE_URL not found", str(context.exception))

    @patch("psycopg2.connect")
    @patch.dict(os.environ, {"DATABASE_URL": "postgresql://test_user:test_pass@localhost:5432/test_db"})
    def test_initialization_schema_setup(self, mock_connect):
        # Set up mocks for connection and cursor
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn
        
        # Instantiate VectorStore - should trigger _initialize_schema
        store = VectorStore()
        
        # Verify psycopg2.connect was called with our connection string
        mock_connect.assert_called_with("postgresql://test_user:test_pass@localhost:5432/test_db")
        
        # Verify CREATE EXTENSION and CREATE TABLE queries were executed
        calls = [call[0][0] for call in mock_cur.execute.call_args_list]
        self.assertTrue(any("CREATE EXTENSION" in q for q in calls))
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS chunks" in q for q in calls))
        
        # Verify commit was called
        mock_conn.commit.assert_called()

    @patch("psycopg2.connect")
    @patch("src.retrieval.vector_store.execute_values")
    @patch.dict(os.environ, {"DATABASE_URL": "postgresql://mock"})
    def test_store_chunks_list_metadata(self, mock_execute_values, mock_connect):
        # Setup DB mocks
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        store = VectorStore()
        
        # Reset mocks to isolate store_chunks calls
        mock_cur.reset_mock()
        mock_conn.commit.reset_mock()
        mock_execute_values.reset_mock()
        
        chunks = ["chunk one", "chunk two"]
        embeddings = [[0.1] * 768, [0.2] * 768]
        metadata = [
            {"pdf_id": "pdf123", "page_number": 2},
            {"pdf_id": "pdf123", "page_number": 3}
        ]
        
        store.store_chunks(chunks, embeddings, metadata)
        
        # Verify execute_values was called with correct parameters
        mock_execute_values.assert_called_once()
        args = mock_execute_values.call_args[0]
        # args[0] is cur, args[1] is query, args[2] is values
        self.assertIn("INSERT INTO chunks", args[1])
        
        # Verify formatting of parameters: embedding should be serialized as pgvector format '[0.1,0.1,...]'
        inserted_rows = args[2]
        self.assertEqual(len(inserted_rows), 2)
        self.assertEqual(inserted_rows[0][0], "pdf123")
        self.assertEqual(inserted_rows[0][1], 2)
        self.assertEqual(inserted_rows[0][2], "chunk one")
        self.assertTrue(inserted_rows[0][3].startswith("[0.1,0.1,"))
        self.assertTrue(inserted_rows[0][3].endswith("]"))
        
        # Verify transaction commit
        mock_conn.commit.assert_called_once()

    @patch("psycopg2.connect")
    @patch("src.retrieval.vector_store.execute_values")
    @patch.dict(os.environ, {"DATABASE_URL": "postgresql://mock"})
    def test_store_chunks_dict_metadata(self, mock_execute_values, mock_connect):
        # Setup DB mocks
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        store = VectorStore()
        
        chunks = ["chunk one", "chunk two"]
        embeddings = [[0.5] * 768, [0.6] * 768]
        # Provide metadata as a single dict with list of page numbers
        metadata = {
            "pdf_id": "single_pdf",
            "page_numbers": [10, 11]
        }
        
        mock_execute_values.reset_mock()
        store.store_chunks(chunks, embeddings, metadata)
        
        mock_execute_values.assert_called_once()
        args = mock_execute_values.call_args[0]
        inserted_rows = args[2]
        self.assertEqual(inserted_rows[0][0], "single_pdf")
        self.assertEqual(inserted_rows[0][1], 10)
        self.assertEqual(inserted_rows[1][0], "single_pdf")
        self.assertEqual(inserted_rows[1][1], 11)

    @patch("psycopg2.connect")
    @patch.dict(os.environ, {"DATABASE_URL": "postgresql://mock"})
    def test_store_chunks_validation_errors(self, mock_connect):
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        store = VectorStore()
        
        # Mismatched lengths should raise ValueError
        with self.assertRaises(ValueError):
            store.store_chunks(["only one chunk"], [[0.1]*768, [0.2]*768], {"pdf_id": "x"})
            
        with self.assertRaises(ValueError):
            store.store_chunks(["one", "two"], [[0.1]*768, [0.2]*768], [{"pdf_id": "x"}])

    @patch("psycopg2.connect")
    @patch.dict(os.environ, {"DATABASE_URL": "postgresql://mock"})
    def test_search(self, mock_connect):
        # Setup DB mocks
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn
        
        # Mock database cursor fetchall results
        # columns returned: id, pdf_id, page_number, content, similarity
        mock_cur.fetchall.return_value = [
            (1, "pdf1", 3, "Hello retrieval", 0.89),
            (2, "pdf1", 5, "World vector search", 0.72)
        ]

        store = VectorStore()
        mock_cur.execute.reset_mock()
        
        query_emb = [0.15] * 768
        results = store.search(query_emb, top_k=2)
        
        # Verify query matches design requirements (using <=> for cosine distance)
        mock_cur.execute.assert_called_once()
        query_str, params = mock_cur.execute.call_args[0]
        self.assertIn("<=>", query_str)
        self.assertIn("LIMIT %s", query_str)
        
        # Verify format of results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], 1)
        self.assertEqual(results[0]["pdf_id"], "pdf1")
        self.assertEqual(results[0]["page_number"], 3)
        self.assertEqual(results[0]["content"], "Hello retrieval")
        self.assertEqual(results[0]["similarity"], 0.89)

if __name__ == "__main__":
    unittest.main()
