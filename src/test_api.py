"""
Unit tests for the FastAPI backend API endpoints.
To run these tests, execute:
    ./venv/bin/python -m unittest src/test_api.py
"""

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.api.main import app


class TestAPI(unittest.TestCase):

    def setUp(self):
        # Initialize FastAPI TestClient
        self.client = TestClient(app)

    def test_health_check(self):
        # Verify health check endpoint returns 200 OK
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    @patch("src.api.main.ingest_pdf")
    def test_upload_pdf_success(self, mock_ingest):
        # Create a mock PDF file payload
        file_payload = ("syllabus.pdf", b"%PDF-1.4 mock pdf structure", "application/pdf")
        response = self.client.post(
            "/upload",
            files={"file": file_payload}
        )
        
        # Verify HTTP status code and response payload
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["pdf_id"], "syllabus.pdf")
        self.assertIn("Successfully processed and ingested", response.json()["message"])
        
        # Verify the RAG ingestion pipeline was triggered with correct arguments
        mock_ingest.assert_called_once()
        args, kwargs = mock_ingest.call_args
        # args[0] is the temp path, pdf_id is the filename
        self.assertEqual(kwargs.get("pdf_id"), "syllabus.pdf")

    def test_upload_pdf_invalid_extension(self):
        # Create a mock plain text file payload
        file_payload = ("notes.txt", b"plain text content", "text/plain")
        response = self.client.post(
            "/upload",
            files={"file": file_payload}
        )
        
        # Verify 400 Bad Request is returned
        self.assertEqual(response.status_code, 400)
        self.assertIn("Only PDF files are supported", response.json()["detail"])

    @patch("src.api.main.answer_question")
    def test_ask_question_success(self, mock_answer_question):
        # Mock RAG pipeline return payload
        mock_answer_question.return_value = {
            "answer": "The mid-term exam is on October 12th [Page 2].",
            "sources": [
                {"pdf_id": "syllabus.pdf", "page_number": 2, "similarity": 0.892}
            ]
        }
        
        # Call Q&A endpoint
        response = self.client.post(
            "/ask",
            json={"question": "When is the mid-term exam?"}
        )
        
        # Verify responses
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "The mid-term exam is on October 12th [Page 2].")
        self.assertEqual(len(response.json()["sources"]), 1)
        self.assertEqual(response.json()["sources"][0]["page_number"], 2)
        
        # Verify pipeline was called with exact question
        mock_answer_question.assert_called_once_with("When is the mid-term exam?")

    def test_ask_question_empty_validation_error(self):
        # Post an empty string question to trigger Pydantic validation (min_length=1)
        response = self.client.post(
            "/ask",
            json={"question": ""}
        )
        
        # Verify Pydantic validation returns 422 Unprocessable Entity
        self.assertEqual(response.status_code, 422)

    def test_ask_question_whitespace_validation_error(self):
        # Post a question containing only spaces to trigger custom whitespace validation
        response = self.client.post(
            "/ask",
            json={"question": "     "}
        )
        
        # Verify custom validation returns 400 Bad Request
        self.assertEqual(response.status_code, 400)
        self.assertIn("Question cannot be empty or contain only whitespace", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
