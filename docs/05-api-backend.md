# Phase 6: FastAPI Backend and API Architecture

In this phase, we wrap our Retrieval-Augmented Generation (RAG) chatbot pipeline in a RESTful API backend using **FastAPI**. Exposing the chatbot via HTTP allows frontends (such as web browsers or mobile apps) or command-line clients to easily upload documents and ask questions.

---

## 1. REST Basics

**REST (Representational State Transfer)** is an architectural style for designing networked applications. It relies on a stateless, client-server protocol—specifically **HTTP**—to query and manipulate resources.

We interact with the API using standard HTTP verbs (methods) mapped to endpoints:
* **GET `/`**: Used for safe, idempotent queries. We use this as a simple health check to ensure the backend is running.
* **POST `/upload`**: Used to send data to the server that changes state (writing document chunks to the database). The client uploads a PDF file using `multipart/form-data`.
* **POST `/ask`**: Used to submit a question payload and generate a response. Even though Q&A doesn't modify the database, we use `POST` instead of `GET` because:
  1. The question text can be long, and passing it in the HTTP request body is cleaner and safer than URL query parameters.
  2. Request bodies are standard for `POST` requests and allow us to enforce structural Pydantic validation schemas.

---

## 2. Request and Response JSON Shapes

Data exchanged with the API is formatted as JSON. Below are the schemas and example shapes for each endpoint:

### A. POST `/upload`
* **Request:** `multipart/form-data` containing a `file` field with the PDF stream.
* **Response JSON Shape (200 OK):**
  ```json
  {
    "message": "Successfully processed and ingested PDF document: syllabus.pdf",
    "pdf_id": "syllabus.pdf"
  }
  ```
* **Validation Failure (400 Bad Request):**
  ```json
  {
    "detail": "Only PDF files are supported. Please upload a file with a '.pdf' extension."
  }
  ```

### B. POST `/ask`
* **Request JSON Shape:**
  ```json
  {
    "question": "When is the mid-term exam?"
  }
  ```
* **Response JSON Shape (200 OK):**
  ```json
  {
    "answer": "The mid-term exam is scheduled for October 12th in Room 304 [Page 2].",
    "sources": [
      {
        "pdf_id": "syllabus.pdf",
        "page_number": 2,
        "similarity": 0.8412
      }
    ]
  }
  ```
* **Pydantic Validation Failure (422 Unprocessable Entity - e.g., missing question):**
  ```json
  {
    "detail": [
      {
        "loc": ["body", "question"],
        "msg": "field required",
        "type": "value_error.missing"
      }
    ]
  }
  ```
* **Custom Validation Failure (400 Bad Request - e.g., only spaces):**
  ```json
  {
    "detail": "Question cannot be empty or contain only whitespace."
  }
  ```

---

## 3. Separation of Concerns (SoC)

**Separation of Concerns** is a design principle for separating a computer program into distinct sections, such that each section addresses a separate concern. In our project, we isolate the components into specific architectural layers:

```
+-------------------------------------------------------------+
|              API Layer (FastAPI: main.py)                   |  <-- Handles HTTP routes, file streams, JSON parsing
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|           Business Logic Layer (rag_chat.py)                |  <-- Orchestrates the ingestion and QA pipeline flows
+------------------------------+------------------------------+
                               |
          +--------------------+--------------------+
          |                                         |
          v                                         v
+------------------+                       +------------------+
| Ingestion Layers |                       | Retrieval Layer  |  <-- Database operations (vector search)
| (pdf_reader.py)  |                       | (vector_store.py)|
+------------------+                       +------------------+
```

### Why Separation of Concerns Makes the Code Maintainable

1. **Independent Testability:**
   * We can test the FastAPI routing, HTTP status codes, and Pydantic validation rules in isolation using FastAPI's `TestClient` while mocking the ingestion/QA pipeline. This makes unit testing incredibly fast (under 0.2 seconds) because it doesn't wait for actual database connections or third-party LLM API roundtrips.
   * We can test PDF text extraction or sliding-window chunking algorithms offline without needing a web server running.
2. **Component Swappability:**
   * If we decide to swap our database from PostgreSQL/pgvector to a specialized vector database like Pinecone or Qdrant, we only need to modify `vector_store.py`. The API routing (`main.py`) and the generative QA logic remain completely untouched.
   * If we want to replace FastAPI with another framework (like Flask or Django) or run the chatbot as a Discord bot, we can plug in the same RAG pipeline functions (`ingest_pdf`, `answer_question`) into the new interface file without modifying any core business logic.
3. **Robustness and Error Isolation:**
   * HTTP-specific parsing issues (such as client network drops during uploads or malformed JSON payloads) are caught and handled by FastAPI's middle-ware before they can propagate down to the database or model endpoints, preventing server crashes.

---

## 4. Running and Testing the Server

### A. Starting the FastAPI Server
To launch the backend API using **Uvicorn** (the ASGI web server), execute:
```bash
./venv/bin/uvicorn src.api.main:app --reload --port 8000
```
The `--reload` flag automatically refreshes the server whenever you edit a source code file.

### B. Accessing the Auto-Generated OpenAPI Docs
FastAPI automatically generates interactive REST documentation. Once the server is running, open your browser and navigate to:
* **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### C. Manual Testing with curl

#### 1. Ingestion File Upload
```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/my_syllabus.pdf"
```

#### 2. Asking a Grounded Question
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"question":"When is the capstone review meeting?"}'
```

---

## Learning Outcomes

By completing Phase 6, you can now:
1. **Explain** the principles of RESTful APIs, HTTP verbs (GET, POST), and the role of request bodies.
2. **Implement** web servers in Python using FastAPI and Uvicorn.
3. **Apply** validation schemas using Pydantic models for incoming JSON payloads.
4. **Develop** secure file upload endpoints that handle binary file streams and perform temporary disk cleanup.
5. **Defend** the principle of Separation of Concerns and describe how it simplifies debugging, unit testing, and architectural refactoring in production.
