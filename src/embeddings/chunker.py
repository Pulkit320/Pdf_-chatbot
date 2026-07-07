"""
This module handles splitting extracted text into smaller, overlapping chunks.
It serves as Phase 2 of our PDF Chatbot pipeline.
"""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits a single body of text into smaller, overlapping chunks.

    Why we use chunking:
    Large Language Models (LLMs) have limited context windows, and processing huge documents
    at once is expensive and slow. Breaking down the text into smaller pieces allows us to 
    only feed relevant sections into the LLM during query time.

    Why overlap prevents losing context at chunk boundaries:
    If we split text strictly at a fixed limit without any overlap, we run a high risk of 
    cutting a key sentence, number, or logical concept right in half at the boundary. 
    For example, if a sentence like "The server password is 'admin123' and should be changed"
    gets split so "The server password is" falls in Chunk A and "'admin123' and should be changed" 
    falls in Chunk B, both chunks become contextually useless.
    By overlapping adjacent chunks (e.g. by 50 characters), the trailing content of Chunk A is 
    duplicated at the beginning of Chunk B. This overlap acts as a bridge that keeps words, 
    names, and structural context intact across boundaries, ensuring that downstream embedding 
    models and the generator LLM can understand the text contextually.

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
    # Why validate inputs explicitly:
    # Ensuring chunk_size and overlap parameters are mathematically valid prevents 
    # infinite loops or empty outputs that would be hard to debug later.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer greater than 0.")
    if overlap < 0:
        raise ValueError("overlap must be a non-negative integer (0 or greater).")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly less than chunk_size.")

    # Why strip text:
    # Extraneous leading/trailing whitespace shouldn't be counted towards the 
    # character count of individual chunks.
    clean_text = text.strip() if text else ""
    if not clean_text:
        return []

    chunks = []
    start = 0
    text_len = len(clean_text)

    # Why use a while loop with variable step size:
    # A sliding window allows us to step forward by exactly `chunk_size - overlap` characters
    # each time, guaranteeing the specified overlap between consecutive chunks.
    while start < text_len:
        end = start + chunk_size
        chunk = clean_text[start:end]
        chunks.append(chunk)
        
        # If we reached the end of the text, we can stop chunking.
        if end >= text_len:
            break
            
        start += (chunk_size - overlap)

    return chunks
