"""
Unit tests for the integrated RAG chatbot component.
To run these tests, execute:
    ./venv/bin/python -m unittest src/test_rag_chat.py
"""

import unittest
from unittest.mock import patch, MagicMock
import src.chatbot.rag_chat as rag_chat


class TestRAGChat(unittest.TestCase):

    @patch("src.chatbot.rag_chat.get_embedding")
    @patch("src.chatbot.rag_chat.VectorStore")
    def test_answer_question_success(self, mock_vector_store_cls, mock_get_embedding):
        # 1. Mock query embedding retrieval
        mock_get_embedding.return_value = [0.1] * 768
        
        # 2. Mock database vector search
        mock_store = MagicMock()
        mock_vector_store_cls.return_value = mock_store
        mock_store.search.return_value = [
            {
                "id": 1,
                "pdf_id": "test_doc.pdf",
                "page_number": 4,
                "content": "This is page 4 content discussing pgvector.",
                "similarity": 0.85
            },
            {
                "id": 2,
                "pdf_id": "test_doc.pdf",
                "page_number": 5,
                "content": "This is page 5 content discussing Neon Postgres.",
                "similarity": 0.75
            }
        ]
        
        # 3. Mock the Gemini generation model response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Based on page 4, pgvector is used. Based on page 5, Neon Postgres is used. [Page 4], [Page 5]"
        mock_client.models.generate_content.return_value = mock_response
        
        # Patch the client instance inside the module
        with patch("src.chatbot.rag_chat.client", mock_client):
            res = rag_chat.answer_question("Tell me about pgvector and Neon.")
            
            # Assertions
            # Verify query embedding retrieval parameters
            mock_get_embedding.assert_called_once_with("Tell me about pgvector and Neon.", task_type="retrieval_query")
            
            # Verify database retrieval parameters
            mock_store.search.assert_called_once_with([0.1] * 768, top_k=3)
            
            # Verify Gemini client API call parameters
            mock_client.models.generate_content.assert_called_once()
            called_args, called_kwargs = mock_client.models.generate_content.call_args
            
            # Check model name is correct
            self.assertEqual(called_kwargs.get("model"), "gemini-2.5-flash")
            self.assertIn("Question: Tell me about pgvector and Neon.", called_kwargs.get("contents"))
            
            # Verify output dictionary format
            self.assertEqual(res["answer"], mock_response.text)
            self.assertEqual(len(res["sources"]), 2)
            self.assertEqual(res["sources"][0]["page_number"], 4)
            self.assertEqual(res["sources"][0]["pdf_id"], "test_doc.pdf")
            self.assertEqual(res["sources"][1]["page_number"], 5)

    @patch("src.chatbot.rag_chat.get_embedding")
    @patch("src.chatbot.rag_chat.VectorStore")
    def test_answer_question_no_results(self, mock_vector_store_cls, mock_get_embedding):
        # Test case where database retrieval yields empty results
        mock_get_embedding.return_value = [0.1] * 768
        mock_store = MagicMock()
        mock_vector_store_cls.return_value = mock_store
        mock_store.search.return_value = []
        
        res = rag_chat.answer_question("Question with no data")
        self.assertIn("no document chunks available", res["answer"])
        self.assertEqual(res["sources"], [])


if __name__ == "__main__":
    unittest.main()
