"""
This module handles generating text embeddings using Google AI Studio's Gemini API.
It serves as Phase 3 of our PDF Chatbot pipeline.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

# Why we call load_dotenv():
# In local development, environment variables are typically stored in a .env file.
# load_dotenv() reads the key-value pairs from .env and adds them to the environment,
# allowing os.getenv() to access them as if they were set globally.
load_dotenv()

def get_embedding(text: str) -> list[float]:
    """
    Generates a mathematical representation (embedding vector) of the input text
    using Google AI Studio's Gemini Embedding API.

    Why we use embeddings:
    Computers cannot easily compare sentences by meaning, but they can compare lists of numbers.
    Embeddings represent text in a high-dimensional vector space (768 dimensions for text-embedding-004) 
    where the distance between vectors correlates with semantic similarity.

    How to get a free Gemini API key from Google AI Studio:
    1. Go to Google AI Studio: https://aistudio.google.com/
    2. Sign in using your Google Account.
    3. Click on the "Get API key" button in the upper-left navigation panel.
    4. Click "Create API key" (you can choose to create it in an existing Google Cloud 
       Project or let the system set up a new one).
    5. Copy your new API key.
    6. Paste it into your project's `.env` file under the name `GOOGLE_API_KEY` (e.g. GOOGLE_API_KEY="AIzaSy...").
       Be sure to keep this file out of version control (it is added to `.gitignore`).

    Args:
        text (str): The raw text segment to be embedded.

    Returns:
        list[float]: A list of float values representing the embedding vector.

    Raises:
        ValueError: If neither GOOGLE_API_KEY nor GEMINI_API_KEY are configured in the environment.
        RuntimeError: If the API call fails or the response does not contain an embedding.
    """
    # Why check both GOOGLE_API_KEY and GEMINI_API_KEY:
    # The prompt explicitly specifies loading GOOGLE_API_KEY from .env, but standard projects
    # and existing templates may have initialized GEMINI_API_KEY. We look up both keys to 
    # maximize backward compatibility and ease of deployment.
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not found. Please add GOOGLE_API_KEY or GEMINI_API_KEY to your .env file."
        )

    # Why configure genai client per-call or lazy load:
    # Since environment variables can be set or changed at runtime (especially during testing),
    # configuring the client inside the function ensures it always uses the most up-to-date key.
    genai.configure(api_key=api_key)

    # Why strip text:
    # Removing leading/trailing whitespace avoids passing empty lines or excess spaces 
    # to the API, minimizing potential token overhead and noise.
    clean_text = text.strip() if text else ""
    if not clean_text:
        return []

    try:
        # Why use text-embedding-004:
        # text-embedding-004 is the state-of-the-art text embedding model from Google,
        # designed specifically for retrieval tasks.
        #
        # Why task_type="retrieval_document":
        # The 'retrieval_document' task type tells the model that this text is part of a 
        # larger document index (which we will search later). If we were embedding a search 
        # query, we would use 'retrieval_query'. Specifying the correct task type yields 
        # significantly higher accuracy for semantic search.
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=clean_text,
            task_type="retrieval_document"
        )
        
        # Why check dictionary keys:
        # Validating the API's return structure prevents AttributeError or KeyError 
        # when accessing the list values if the API payload changes.
        if not response or "embedding" not in response:
            raise KeyError("The API response did not contain the 'embedding' key.")
            
        return response["embedding"]
        
    except Exception as e:
        raise RuntimeError(f"Gemini Embedding API call failed: {e}")
