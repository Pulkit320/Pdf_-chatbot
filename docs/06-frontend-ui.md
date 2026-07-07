# Phase 7: Streamlit Frontend Web Interface and the Full-Stack Connection

In this final phase of our PDF Chatbot project, we build a clean, web-based frontend using **Streamlit**. This acts as the visual portal for users to interact with the backend API, completing the entire **full-stack** application.

---

## 1. The Full-Stack Request-Response Lifecycle

For a web developer, understanding how data travels between the client (the interface the user sees in the browser) and the server (the API doing the heavy lifting behind the scenes) is a core foundational skill. 

Below is a detailed walkthrough of the complete request-response lifecycle when a user asks a question in our chatbot interface:

```
[ BROWSER CLIENT (Streamlit: app.py) ]
  |
  |  1. User enters text query in chat_input and hits enter
  |  2. UI sets state to "loading" and displays st.spinner
  |  3. UI issues POST request: requests.post("http://127.0.0.1:8000/ask", json={...})
  v
====== Transport layer over TCP/IP Port 8000 ==================================
  |
  |  4. HTTP payload reaches FastAPI backend (main.py)
  v
[ BACKEND API SERVER (FastAPI: main.py) ]
  |
  |  5. FastAPI parses JSON request body and validates it matches QuestionRequest
  |  6. Backend invokes answer_question("my query") inside rag_chat.py
  v
[ BUSINESS & RETRIEVAL LOGIC (rag_chat.py) ]
  |
  |  7. Embedder fetches vector coordinates using models/gemini-embedding-2
  |  8. VectorStore executes Cosine Similarity query (<=>) in PostgreSQL database
  |  9. Database retrieves nearest matching text chunks + page numbers
  | 10. Prompt compiled with retrieved chunks and passed to gemini-2.5-flash
  | 11. LLM returns factual response grounded in context
  v
[ BACKEND API SERVER (FastAPI: main.py) ]
  |
  | 12. FastAPI packs text response and sources list into AskResponse Pydantic model
  | 13. FastAPI serialization generates JSON response body string
  | 14. Server responds with HTTP status 200 OK
  v
====== Transport layer over TCP/IP Port 8000 ==================================
  |
  | 15. Client's HTTP connection resolves successfully
  v
[ BROWSER CLIENT (Streamlit: app.py) ]
  |
  | 16. requests.post() returns Response object; UI parses JSON using .json()
  | 17. st.spinner stops rendering
  | 18. UI appends message to st.session_state and triggers a page re-render
  | 19. Browser updates DOM, displaying the grounded answer and clickable source cards
```

---

## 2. Streamlit State Management: Session State

Streamlit is built on a unique execution model: **on every single user interaction** (e.g., clicking a button, uploading a file, or hitting enter in a chat input box), **Streamlit runs the entire Python script from line 1 to the end.**

This approach simplifies page layouts but introduces a major problem: normal Python variables are completely reset on every re-run. If we stored chat history in a plain list like `chat_history = []`, that list would be re-declared empty every time the user sent a new question, deleting all past messages.

To solve this, Streamlit provides **Session State (`st.session_state`)**.

### How Session State Works

`st.session_state` is a dictionary-like object that persists data between script executions for a specific user session. 
* **Initialization:** We check if the state key exists on boot. If not, we set it:
  ```python
  if "messages" not in st.session_state:
      st.session_state.messages = []
  ```
* **Appending:** When a new user question or assistant answer arrives, we append it directly:
  ```python
  st.session_state.messages.append({"role": "user", "content": user_query})
  ```
* **Rendering:** At the top of the script, we loop through and render all stored messages, ensuring they are redrawn on screen even during re-runs triggered by other widgets (like uploading a file in the sidebar).

---

## 3. UI Design and Usability Choices

A premium user interface requires feedback loops that keep the user informed. We implement several design choices to elevate the UX:

1. **Loading Spinner (`st.spinner`)**:
   Calling the vector search and Gemini generative APIs can take between 1.5 and 3 seconds. Without a loading indicator, the interface would appear frozen, leading the user to believe the website has crashed. The spinner provides immediate visual confirmation that the backend is working.
2. **Alert Banners (`st.success`, `st.error`)**:
   Document ingestion can succeed or fail. Utilizing color-coded notices (Green for success, Red for failure) helps users immediately diagnose problems (such as a disconnected backend server or invalid API keys).
3. **Collapsible Source Cards (`st.expander`)**:
   Displaying the exact matching raw text chunks, document IDs, and similarity scores is essential for grounding transparency. However, dumping pages of raw text directly in the chat feed clutters the UI. Placing citations inside a collapsible expander keeps the layout tidy while keeping citations audit-ready.

---

## 4. How to Run the Full-Stack Application

To test the complete system, we run both backend and frontend applications side-by-side:

### Step 1: Launch the FastAPI Backend
Open a terminal in the project directory, ensure your virtual environment is active, and start Uvicorn:
```bash
./venv/bin/uvicorn src.api.main:app --reload --port 8000
```

### Step 2: Launch the Streamlit Web Server
Open a second terminal, activate the virtual environment, and run:
```bash
./venv/bin/streamlit run src/ui/app.py --server.port 8501
```

Once running, Streamlit will print the local network address (typically `http://localhost:8501`). Open this URL in your web browser to access your interactive PDF chatbot!

---

## Learning Outcomes

By completing Phase 7, you can now:
1. **Explain** the request-response lifecycle of a full-stack web application, detailing how HTTP packets transport structured JSON data between clients and servers.
2. **Implement** dynamic browser interfaces in Python using Streamlit widgets.
3. **Contrast** Streamlit's script-execution model with standard web frameworks and solve memory reset limitations using `st.session_state`.
4. **Integrate** frontend components with external REST endpoints via Python's `requests` client library.
5. **Formulate** UX design concepts (spinners, collapsible cards, alerts) that turn raw JSON responses into a clean, premium interface.
