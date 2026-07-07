"""
Streamlit Web Frontend Application.
Connects to the FastAPI backend API to upload PDF documents and ask Q&A queries.
"""

import os
import sys
import logging
import requests
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FastAPI base endpoint url
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Set Page Configurations
st.set_page_config(
    page_title="PDF Chatbot - Grounded Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sleek CSS for Modern UI/UX and Rich Aesthetics
st.markdown("""
<style>
    /* Styling headers */
    .main-header {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #6B7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Custom containers for sources */
    .source-card {
        background-color: #F9FAFB;
        border-left: 4px solid #06B6D4;
        padding: 12px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .source-title {
        font-weight: 600;
        color: #111827;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }
    
    .source-meta {
        font-size: 0.8rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State Variables to persist chat history across runs
if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar Layout: PDF Ingestion Hub
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/open-book.png", width=80)
    st.title("Document Manager")
    st.write("Upload course syllabi, textbooks, or documentation PDFs to populate the chatbot vector memory database.")
    
    # PDF Ingestion Form Uploader
    uploaded_file = st.file_uploader(
        "Select PDF Document", 
        type=["pdf"], 
        help="Only standard PDF files are accepted."
    )
    
    if uploaded_file is not None:
        # Create a trigger button to run ingestion to avoid accidental double uploads
        upload_button = st.button("🚀 Index Document", use_container_width=True)
        
        if upload_button:
            with st.spinner(f"Ingesting and indexing {uploaded_file.name}..."):
                try:
                    # Prepare file stream dict
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
                    }
                    
                    # POST request to FastAPI endpoint
                    response = requests.post(f"{API_BASE_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        st.success(f"Success! '{uploaded_file.name}' has been split, embedded, and saved to the vector database.")
                        st.balloons()
                    else:
                        error_detail = response.json().get("detail", "Unknown error occurred.")
                        st.error(f"Failed to ingest: {error_detail}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Connection Error: Could not connect to the FastAPI backend. Make sure it is running on port 8000.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during upload: {str(e)}")


# Main Chat Interface Panel
st.markdown("<h1 class='main-header'>📚 Grounded PDF Q&A Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ask questions about your documents. The assistant is anchored to your uploads and cites sources page-by-page.</p>", unsafe_allow_html=True)

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If there are sources attached to the assistant message, render them nicely
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("🔍 Verified Source Citations"):
                for src in message["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">📄 {src['pdf_id']}</div>
                        <div class="source-meta">Page {src['page_number']} | Relevance Similarity: {src['similarity']:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)

# Chat Input Box
user_query = st.chat_input("Ask a question about the ingested PDFs...")

if user_query:
    # 1. Append and display User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Run API generation with spinner
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Searching document coordinates & generating response..."):
            try:
                # Post request payload
                payload = {"question": user_query}
                response = requests.post(f"{API_BASE_URL}/ask", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("answer", "")
                    sources = result.get("sources", [])
                    
                    # Render generated grounded text
                    message_placeholder.markdown(answer)
                    
                    # Render Citations box if sources exist
                    if sources:
                        with st.expander("🔍 Verified Source Citations"):
                            for src in sources:
                                st.markdown(f"""
                                <div class="source-card">
                                    <div class="source-title">📄 {src['pdf_id']}</div>
                                    <div class="source-meta">Page {src['page_number']} | Relevance Similarity: {src['similarity']:.4f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # 3. Store assistant message in history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                else:
                    error_detail = response.json().get("detail", "Error generating response.")
                    error_msg = f"⚠️ Backend returned error: {error_detail}"
                    message_placeholder.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except requests.exceptions.ConnectionError:
                err_msg = "🚨 Connection Error: Unable to reach the FastAPI backend server. Verify the server is running."
                message_placeholder.markdown(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
            except Exception as e:
                err_msg = f"🚨 An unexpected error occurred: {str(e)}"
                message_placeholder.markdown(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
