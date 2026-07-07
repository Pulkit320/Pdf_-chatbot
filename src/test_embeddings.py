"""
Unit tests for the chunker and embedder modules.
To run these tests, execute:
    ./venv/bin/python -m unittest src/test_embeddings.py
"""

import os
import unittest
from unittest.mock import patch, MagicMock

# Import chunker and embedder
from src.embeddings.chunker import chunk_text
from src.embeddings.embedder import get_embedding

class TestChunker(unittest.TestCase):
    def test_basic_chunking(self):
        # A simple string of 15 characters
        text = "abcdefghijklmno"
        # chunk_size = 10, overlap = 5
        # Chunk 1: indices 0-10 -> "abcdefghij"
        # Chunk 2: start at 10 - 5 = 5. indices 5-15 -> "fghijklmno"
        result = chunk_text(text, chunk_size=10, overlap=5)
        self.assertEqual(result, ["abcdefghij", "fghijklmno"])

    def test_text_shorter_than_chunk_size(self):
        text = "short"
        result = chunk_text(text, chunk_size=10, overlap=2)
        self.assertEqual(result, ["short"])

    def test_empty_string(self):
        self.assertEqual(chunk_text("", chunk_size=10, overlap=2), [])
        self.assertEqual(chunk_text("   ", chunk_size=10, overlap=2), [])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            chunk_text("test", chunk_size=0, overlap=5)
        with self.assertRaises(ValueError):
            chunk_text("test", chunk_size=-10, overlap=5)
        with self.assertRaises(ValueError):
            chunk_text("test", chunk_size=10, overlap=-2)
        with self.assertRaises(ValueError):
            chunk_text("test", chunk_size=10, overlap=10)
        with self.assertRaises(ValueError):
            chunk_text("test", chunk_size=10, overlap=12)


class TestEmbedder(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self):
        # With clear=True, os.environ won't have GOOGLE_API_KEY or GEMINI_API_KEY
        with self.assertRaises(ValueError) as context:
            get_embedding("test text")
        self.assertIn("API key not found", str(context.exception))

    @patch("google.generativeai.embed_content")
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_test_key"})
    def test_successful_embedding_mock(self, mock_embed_content):
        # Mock successful response from Gemini API
        fake_vector = [0.1, -0.2, 0.35, -0.8]
        mock_embed_content.return_value = {"embedding": fake_vector}

        result = get_embedding("Hello Gemini")
        self.assertEqual(result, fake_vector)
        
        # Verify it was called with correct model and content
        mock_embed_content.assert_called_once_with(
            model="models/text-embedding-004",
            content="Hello Gemini",
            task_type="retrieval_document"
        )

    @patch("google.generativeai.embed_content")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake_test_key_fallback"})
    def test_fallback_api_key(self, mock_embed_content):
        # Test fallback to GEMINI_API_KEY when GOOGLE_API_KEY is not set
        fake_vector = [0.9, 0.8, 0.7]
        mock_embed_content.return_value = {"embedding": fake_vector}

        # Clear GOOGLE_API_KEY explicitly in the test just in case
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]

        result = get_embedding("Fallback key test")
        self.assertEqual(result, fake_vector)

    @patch("google.generativeai.embed_content")
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_test_key"})
    def test_api_failure_handling(self, mock_embed_content):
        # Mock API raising an exception
        mock_embed_content.side_effect = Exception("Quota exceeded")

        with self.assertRaises(RuntimeError) as context:
            get_embedding("should fail")
        self.assertIn("Gemini Embedding API call failed", str(context.exception))

if __name__ == "__main__":
    unittest.main()
