"""
This module implements a FastAPI REST API for the PDF Chatbot backend.
It exposes endpoints to upload PDF documents and ask questions grounded
in the document context.
"""

import os
import sys
import shutil
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel, Field

# Ensure we can import modules from parent directories if executed directly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.chatbot.rag_chat import ingest_pdf, answer_question

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Chatbot API",
    description="A FastAPI backend exposing endpoints for document ingestion and Q&A semantic search.",
    version="1.0.0"
)

# Input/Output validation models
class QuestionRequest(BaseModel):
    question: str = Field(
        ..., 
        min_length=1, 
        description="The question query to ask the chatbot."
    )

class SourceResponse(BaseModel):
    pdf_id: str
    page_number: int
    similarity: float

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "ok", "message": "FastAPI PDF Chatbot Backend is running."}


@app.post("/upload", status_code=status.HTTP_200_OK)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file upload, validates its extension, saves it temporarily, 
    and triggers the RAG ingestion pipeline (extracting, chunking, embedding, 
    and storing vectors in PostgreSQL).
    """
    # Validate file extension
    # We check if filename is empty or not ending with .pdf
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.error(f"Upload rejected: File '{file.filename}' is not a PDF.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported. Please upload a file with a '.pdf' extension."
        )

    logger.info(f"Received PDF upload request for file: {file.filename}")

    # Create temporary directory to safely write file stream
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        # Write UploadFile stream to the temp file
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Trigger RAG ingestion pipeline
        ingest_pdf(temp_path, pdf_id=file.filename)

        logger.info(f"Successfully processed and ingested PDF: {file.filename}")
        return {
            "message": f"Successfully processed and ingested PDF document: {file.filename}",
            "pdf_id": file.filename
        }

    except Exception as e:
        logger.error(f"Ingestion pipeline failed for '{file.filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document. Details: {str(e)}"
        )

    finally:
        # Clean up temporary directories and files to avoid disk leaks
        try:
            shutil.rmtree(temp_dir)
            logger.info("Temporary upload cleanup completed.")
        except Exception as cleanup_err:
            logger.warning(f"Failed to clean up temporary upload directory: {cleanup_err}")


@app.post("/ask", response_model=AskResponse, status_code=status.HTTP_200_OK)
async def ask(request: QuestionRequest):
    """
    Accepts a user question query in JSON, performs a semantic vector search 
    against PostgreSQL chunks, prompts Gemini with context grounding instructions, 
    and returns the grounded answer along with source citations.
    """
    # Custom validation: check if question has content after trimming spaces
    clean_question = request.question.strip()
    if not clean_question:
        logger.error("Rejecting Q&A request: Question contains only whitespace.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or contain only whitespace."
        )

    logger.info(f"Received Q&A query: '{clean_question}'")

    try:
        # Call RAG Q&A pipeline
        result = answer_question(clean_question)
        logger.info("Successfully generated grounded answer.")
        return result

    except Exception as e:
        logger.error(f"Error executing Q&A pipeline for query '{clean_question}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query. Details: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    # Allow running the API server directly with python src/api/main.py
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
