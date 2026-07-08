# 📚 PDF Chatbot: Grounded Q&A Assistant

A full-stack **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents (such as textbooks, manuals, or documentation) and ask questions grounded strictly in their content. 

This project integrates a **FastAPI backend** and a **Streamlit frontend** with a **PostgreSQL database (using `pgvector`)** for vector search, utilizing **Google Gemini** for embedding generation and factual Q&A.

---

## 🏗️ Architecture & Data Flow

```mermaid
graph TD
    A[Upload PDF via Streamlit] -->|Multipart Form| B[FastAPI Endpoint: /upload]
    B -->|Page-by-Page Extraction| C[pdfplumber / pypdf]
    C -->|Sliding Window Chunking| D[Overlap Chunker]
    D -->|Embed Chunk: text-embedding-004| E[Gemini API]
    E -->|Store Vectors & Metadata| F[PostgreSQL + pgvector]
    
    G[User Asks Question] -->|POST Request| H[FastAPI Endpoint: /ask]
    H -->|Embed Query: text-embedding-004| I[Gemini API]
    I -->|Cosine Similarity Search| F
    F -->|Retrieve Top 3 Matching Chunks| J[RAG Prompt Assembly]
    J -->|Query with Grounded Context| K[Gemini 2.0 Flash]
    K -->|Generate Cited Answer| H
    H -->|JSON Response| L[Streamlit UI Chat Feed]
```

---

## 🚀 Key Features

* **Grounded Q&A (No Hallucinations)**: Prompts Gemini using strict system instructions to answer *only* using the provided text context. If the answer is not present, the bot explicitly says so.
* **Auto-Citations**: Every claim or statement in the chatbot's response is cited with brackets linking back to the exact page number (e.g., `[Page X]`).
* **Interactive Source Audit**: Streamlit UI includes collapsible cards detailing the exact text snippets, source files, page numbers, and cosine similarity matches retrieved from the database.
* **Separation of Concerns**: Modular python architecture decoupling PDF ingestion, embedding generation, vector storage, business logic (RAG pipeline), REST API endpoints, and user interface.

---

## 🛠️ Technology Stack

* **Frontend**: Streamlit
* **Backend API**: FastAPI + Uvicorn
* **Large Language Models**: Google Gemini API (`gemini-2.0-flash` & `text-embedding-004`) via `google-genai` SDK
* **Vector Store**: PostgreSQL with `pgvector` extension
* **Libraries**: `pypdf`, `pdfplumber`, `psycopg2-binary`, `pydantic`, `pytest`

---

## ⚙️ Setup & Installation

### Prerequisite: PostgreSQL with `pgvector`
Ensure you have access to a PostgreSQL database with the `pgvector` extension installed (e.g., local Postgres or hosted services like Neon/Supabase). Run the following SQL query on your database to enable vector features:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 1. Clone the repository and navigate to the directory
```bash
git clone <repository-url>
cd pdf_chatbot
```

### 2. Create and activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql://username:password@host:port/database_name
```

---

## 🖥️ Running the Application

To run the complete full-stack system, launch both the backend API and the frontend application side-by-side in separate terminals:

### Step 1: Start the FastAPI Backend
```bash
./venv/bin/uvicorn src.api.main:app --reload --port 8000
```
* **API Documentation**: Interactive documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Step 2: Start the Streamlit Frontend
```bash
./venv/bin/streamlit run src/ui/app.py --server.port 8501
```
* **Web UI Access**: Navigate to [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Testing and Evaluation

Run the unit tests to verify the chunker, PDF reader, and vector store database connections:
```bash
pytest tests/
```
