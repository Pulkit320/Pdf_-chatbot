# Phase 2 & 3: Text Chunking and Embeddings

In this guide, we will implement the next two phases of our PDF Chatbot pipeline: **Text Chunking** (Phase 2) and **Generating Embeddings** (Phase 3). 

Once we extract text from a PDF, we cannot simply send the raw, massive document straight to a Large Language Model (LLM) or database. We need to cut the text down into small, digestible snippets (chunks) and convert those snippets into a format that computers can compare and search mathematically (embeddings).

---

## 1. What are Chunks and Embeddings?

Before looking at the code, let's understand the core concepts using a visual and geographic analogy.

### The GPS Analogy: Coordinates for Meaning

Imagine you have a physical map of the world. Every location on Earth can be pinpointed using a set of numbers: **GPS coordinates** (latitude and longitude). 
* Locations that are geographically close to each other (like your house and your local grocery store) will have very similar GPS coordinates.
* Locations that are far apart (like New York and Tokyo) will have very different coordinates.

In natural language processing, **embeddings are like GPS coordinates for meaning**. 

Instead of representing a physical place on a 2D map, an embedding represents a word, phrase, or paragraph inside a high-dimensional "semantic space" (typically 768 dimensions for Google's `text-embedding-004`). 
* If two sentences mean similar things (e.g., *"My dog loves running in the yard"* and *"A puppy is playing outside in the grass"*), the embedding model will output list of numbers (vectors) that are mathematically very close to each other.
* If two sentences have completely different meanings (e.g., *"The database transaction has completed"* and *"A puppy is playing outside"*), their coordinate lists will be very far apart.

This allows us to perform **semantic search**: we convert a user's question into "GPS coordinates" (an embedding) and find the document chunks whose coordinates are closest to it.

```
                  [Semantic Meaning Space]
             
       "A puppy playing in grass"     "My dog running in the yard"
                     \               /
                      \ (Close coordinates)
                       \           /
                        [   *   *   ]
                            
                            
                            
                        [     *     ]
                              /
                             / (Far coordinates)
                            /
             "Database transaction completed"
```

### Why Do We Chunk Text?

PDF documents can be hundreds of pages long. We split this text into smaller, fixed-size chunks (e.g., 500 characters) for three main reasons:
1. **Model Context Windows:** LLMs can only read a certain number of words at once. Chunking ensures we only send relevant fragments.
2. **Retrieval Precision:** If a 50-page document contains one sentence about database backups, indexing the entire document makes it hard to locate the specific fact. Indexing page-level or paragraph-level chunks makes search incredibly precise.
3. **API Cost Efficiency:** Gemini APIs charge by the number of tokens processed. Sending small, relevant text chunks instead of entire books saves a massive amount of money.

### Why Overlap is Crucial at Chunk Boundaries

When splitting text, we use a technique called **overlapping chunks**. 

If we split a document strictly every 500 characters, we might cut a sentence, a name, or an important number directly in half:
* **Chunk 1:** "... The administrator database password is **ad**"
* **Chunk 2:** "**min123** and must be rotated every 30 days..."

In this scenario, Chunk 1 doesn't contain the password, and Chunk 2 doesn't contain the context of what the password is for. Both chunks become semantically useless to the embedding model and the LLM.

By using an **overlap** (e.g., 50 characters), the end of Chunk 1 is duplicated at the start of Chunk 2:
* **Chunk 1:** "... The administrator database password is **admin123**"
* **Chunk 2:** "... **database password is admin123** and must be rotated every 30 days..."

The overlap acts as a **semantic bridge**, preserving the local context of names, numbers, and pronouns across boundaries.

---

## 2. Google AI Studio Signup Walkthrough

To generate embeddings, we will use Google AI Studio's Generative Language API. Google provides a generous free tier for developers.

Follow these steps to obtain your API key:

1. **Navigate to the Portal:** Open your browser and go to [aistudio.google.com](https://aistudio.google.com/).
2. **Sign In:** Log in using a standard Google Account.
3. **Access Keys:** Click the **"Get API key"** button located on the upper-left navigation panel.
4. **Create a Key:** Click the **"Create API key"** button. You can choose to bind it to an existing Google Cloud project or automatically create a new developer project.
5. **Copy the Key:** A modal will pop up displaying your secret key (starting with `AIzaSy...`). Copy it to your clipboard.
6. **Configure the Project:** In your project directory, open your `.env` file and add the key under the variable `GOOGLE_API_KEY`:
   ```bash
   GOOGLE_API_KEY="AIzaSyYourSecretAPIKeyHere"
   ```
   
> [!WARNING]
> Keep your API key completely secure. Never commit it to GitHub. Ensure your `.env` file is listed inside your [.gitignore](file:///home/pulkit/projects/pdf_chatbot/.gitignore) file.

---

## 3. Code Implementation

We have separated chunking logic and embedding logic into two dedicated files inside the `src/embeddings/` package.

### A. Text Chunking
The chunking logic is located in [chunker.py](file:///home/pulkit/projects/pdf_chatbot/src/embeddings/chunker.py). It uses a character-based sliding window algorithm.

```python
"""
This module handles splitting extracted text into smaller, overlapping chunks.
It serves as Phase 2 of our PDF Chatbot pipeline.
"""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits a single body of text into smaller, overlapping chunks.

    Args:
        text (str): The raw text input to be chunked.
        chunk_size (int): The character length limit of each chunk. Defaults to 500.
        overlap (int): The character overlap size between consecutive chunks. Defaults to 50.

    Returns:
        list[str]: A list of text chunks.

    Raises:
        ValueError: If chunk_size is less than or equal to 0, if overlap is negative, 
                    or if overlap is greater than or equal to chunk_size.
    """
    # Guarding against invalid mathematical boundaries
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer greater than 0.")
    if overlap < 0:
        raise ValueError("overlap must be a non-negative integer (0 or greater).")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly less than chunk_size.")

    # Sanitizing input
    clean_text = text.strip() if text else ""
    if not clean_text:
        return []

    chunks = []
    start = 0
    text_len = len(clean_text)

    # Sliding window algorithm
    while start < text_len:
        end = start + chunk_size
        chunk = clean_text[start:end]
        chunks.append(chunk)
        
        # Stop iteration if we have reached the end of the text
        if end >= text_len:
            break
            
        # Step forward by (chunk_size - overlap) to maintain the requested overlap
        start += (chunk_size - overlap)

    return chunks
```

### B. Embedding Generation
The embedding connector is located in [embedder.py](file:///home/pulkit/projects/pdf_chatbot/src/embeddings/embedder.py). It calls the Gemini `text-embedding-004` model.

```python
"""
This module handles generating text embeddings using Google AI Studio's Gemini API.
It serves as Phase 3 of our PDF Chatbot pipeline.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load key-value pairs from the .env file
load_dotenv()

def get_embedding(text: str) -> list[float]:
    """
    Generates a mathematical representation (embedding vector) of the input text
    using Google AI Studio's Gemini Embedding API.

    Args:
        text (str): The raw text segment to be embedded.

    Returns:
        list[float]: A list of float values representing the embedding vector.

    Raises:
        ValueError: If the API key is missing.
        RuntimeError: If the API call fails.
    """
    # Look up GOOGLE_API_KEY first, fallback to GEMINI_API_KEY
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not found. Please add GOOGLE_API_KEY or GEMINI_API_KEY to your .env file."
        )

    # Configure client authentication
    genai.configure(api_key=api_key)

    # Clean whitespace input
    clean_text = text.strip() if text else ""
    if not clean_text:
        return []

    try:
        # text-embedding-004 is the state-of-the-art retrieval embedding model.
        # Specifying task_type="retrieval_document" optimizes the vectors for indexing.
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=clean_text,
            task_type="retrieval_document"
        )
        
        if not response or "embedding" not in response:
            raise KeyError("The API response did not contain the 'embedding' key.")
            
        return response["embedding"]
        
    except Exception as e:
        raise RuntimeError(f"Gemini Embedding API call failed: {e}")
```

---

## 4. Block-by-Block Explanation

Let's dissect the core operations inside our modules:

### A. Chunker: Parameter Safeguards
```python
if chunk_size <= 0:
    raise ValueError("chunk_size must be a positive integer greater than 0.")
if overlap >= chunk_size:
    raise ValueError("overlap must be strictly less than chunk_size.")
```
* **Why it matters:** If `chunk_size` is negative, slicing operations fail unpredictably. If `overlap` is equal to or greater than `chunk_size`, the sliding window algorithm's step size (`chunk_size - overlap`) becomes `0` or negative. This would lock the program into an **infinite loop**, consuming CPU indefinitely. Raising explicit exceptions avoids this behavior.

### B. Chunker: The Sliding Window Step
```python
start += (chunk_size - overlap)
```
* **Why it matters:** Instead of incrementing `start` by `chunk_size` (which would result in adjacent chunks with zero overlap), we step forward by `chunk_size - overlap`. If `chunk_size` is 500 and `overlap` is 50, we advance by 450 characters. This guarantees that the final 50 characters of our current chunk will match the first 50 characters of our next chunk.

### C. Embedder: Task-Type Specialization
```python
response = genai.embed_content(
    model="models/text-embedding-004",
    content=clean_text,
    task_type="retrieval_document"
)
```
* **Why it matters:** Google's embedding model is multi-purpose. It can embed search queries, classification classes, clustering elements, and documents. By explicitly declaring `task_type="retrieval_document"`, the neural network weights process the text with the context that it represents an indexed database record. When we query the system later, we will use `task_type="retrieval_query"`. Matching the correct task types maximizes retrieval relevance during vector searches.

---

## 5. Verification & Testing

We have built a unit test file in [test_embeddings.py](file:///home/pulkit/projects/pdf_chatbot/src/test_embeddings.py). 

To ensure developer workflows are fast and do not incur API costs or network errors, the test suite uses **mocking** for API calls.

### How to Run the Tests
Ensure your virtual environment is active, then run:
```bash
./venv/bin/python -m unittest src/test_embeddings.py
```

### What is Tested?
1. **Mathematical boundary cases:** Ensuring negative sizes or invalid parameters immediately raise `ValueError`.
2. **Overlap offsets:** Verifying that a test string (`"abcdefghijklmno"`) with size `10` and overlap `5` produces exactly `["abcdefghij", "fghijklmno"]`.
3. **Environment checks:** Verifying the class raises `ValueError` if all API keys are cleared.
4. **Mocked API interaction:** Ensuring the client correctly requests `models/text-embedding-004` and `retrieval_document` without making real network requests.

---

## Learning Outcomes

By building and documenting the chunking and embedding modules, you can now:
1. **Analyze** why text chunking is necessary for retrieval systems and why character-based overlap prevents semantic truncation at chunk boundaries.
2. **Implement** a sliding window text-chunking algorithm in Python with custom chunk size and overlap parameters.
3. **Configure** and authenticate Google AI Studio's Gemini API within a Python script using environment variables.
4. **Generate** semantic vector embeddings for text chunks using the state-of-the-art `text-embedding-004` model.
5. **Design** offline unit tests with mocking to isolate business logic from third-party network APIs.
