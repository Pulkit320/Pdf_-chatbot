"""
This module handles storing and retrieving document chunk embeddings in a PostgreSQL 
database using the pgvector extension. It serves as Phase 4 of our PDF Chatbot pipeline.
"""

import os
import logging
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Why we call load_dotenv():
# We retrieve database credentials securely from the environment using a .env file.
load_dotenv()

class VectorStore:
    """
    A vector store wrapper for PostgreSQL with pgvector.
    
    Why we use psycopg2 over SQLAlchemy:
    1. psycopg2 is the standard PostgreSQL driver for Python. It is lightweight, fast, 
       and executes raw SQL queries directly.
    2. Using raw SQL makes database operations, schema creation, and pgvector-specific 
       syntax (like the `<=>` distance operator) completely explicit and transparent.
    3. For learning purposes, avoiding ORM abstractions (like SQLAlchemy or pgvector-python's ORM model)
       helps students understand the underlying database interactions and SQL execution models.
    """
    
    def __init__(self, database_url: str = None):
        """
        Initializes the VectorStore connection wrapper and sets up the schema.
        
        Args:
            database_url (str, optional): Connection string. Defaults to DATABASE_URL env var.
        """
        # Load the connection string from environment if not explicitly passed
        self.db_url = database_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError(
                "DATABASE_URL not found. Please provide it or define it in your .env file."
            )
        
        # Initialize schema (tables and extensions)
        self._initialize_schema()
        
    def _get_connection(self):
        """
        Creates and returns a new database connection.
        
        Using a connection creator ensures that if a connection is lost, we can open a new one.
        We use raw psycopg2 connections which can be used as context managers.
        """
        try:
            return psycopg2.connect(self.db_url)
        except Exception as e:
            logger.error(f"Failed to connect to the database: {e}")
            raise ConnectionError(f"Could not connect to PostgreSQL database: {e}")

    def _initialize_schema(self):
        """
        Sets up the database schema, ensuring the pgvector extension is enabled and
        creating the 'chunks' table if it does not exist.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Enable pgvector extension
                # pgvector is a PostgreSQL extension that adds support for vector storage and distance operators.
                logger.info("Ensuring pgvector extension is enabled...")
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                except Exception as e:
                    # In some environments (e.g. strict shared databases), creating extensions might fail.
                    # We roll back the transaction so we can continue in case pgvector was already enabled.
                    conn.rollback()
                    logger.warning(
                        f"Could not run 'CREATE EXTENSION IF NOT EXISTS vector;' - "
                        f"ensuring it's already installed. Detail: {e}"
                    )
                
                # 2. Create chunks table
                # Columns:
                # - id: unique serial identifier.
                # - pdf_id: tracks which PDF document the chunk came from.
                # - page_number: page reference for source attribution.
                # - content: raw text content of the chunk.
                # - embedding: 768-dimensional vector matching Google's text-embedding-004.
                logger.info("Ensuring 'chunks' table exists...")
                create_table_query = """
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    pdf_id VARCHAR(255) NOT NULL,
                    page_number INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(768)
                );
                """
                cur.execute(create_table_query)
                
                # Also create an index to speed up vector searches later if needed
                # For learning purposes, we can use an HNSW index on the embedding column,
                # though pgvector's default sequential scan works perfectly for small student datasets.
                # We won't block on this index creation in case the operator class isn't fully matched.
                try:
                    # HNSW index for cosine distance operator (<=>)
                    cur.execute("""
                    CREATE INDEX IF NOT EXISTS chunks_embedding_cosine_idx 
                    ON chunks USING hnsw (embedding vector_cosine_ops);
                    """)
                except Exception as e:
                    conn.rollback()
                    logger.debug(f"HNSW Index creation skipped/deferred: {e}")
                    
            conn.commit()
            logger.info("Database schema initialized successfully.")

    def store_chunks(self, chunks: list[str], embeddings: list[list[float]], metadata: list[dict] | dict) -> None:
        """
        Inserts document chunks, their embeddings, and associated metadata into the database.
        
        Args:
            chunks (list[str]): The raw text of each chunk.
            embeddings (list[list[float]]): The vector embeddings generated for each chunk.
            metadata (list[dict] or dict): Metadata containing 'pdf_id' and 'page_number'.
                - If list[dict]: Must match the length of chunks. Each dict contains specific metadata.
                - If dict: A single dictionary with 'pdf_id' (str) and 'page_numbers' (list of integers).
                
        Raises:
            ValueError: If inputs are empty or lengths do not match.
        """
        if not chunks or not embeddings:
            logger.warning("Empty chunks or embeddings list provided. Skipping insertion.")
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: Got {len(chunks)} chunks and {len(embeddings)} embeddings.")

        # Normalize metadata format
        normalized_metadata = []
        if isinstance(metadata, dict):
            pdf_id = metadata.get("pdf_id", "unknown_pdf")
            page_numbers = metadata.get("page_numbers", [])
            for i in range(len(chunks)):
                page_num = page_numbers[i] if i < len(page_numbers) else 1
                normalized_metadata.append({"pdf_id": pdf_id, "page_number": page_num})
        elif isinstance(metadata, list):
            if len(metadata) != len(chunks):
                raise ValueError(f"Mismatch: Got {len(chunks)} chunks and {len(metadata)} metadata elements.")
            normalized_metadata = metadata
        else:
            raise TypeError("Metadata must be a dictionary or a list of dictionaries.")

        # Prepare parameters for batch insert
        # We format the embedding float array into pgvector's string representation: '[x,y,z,...]'
        # This string conversion works natively with PostgreSQL's vector type casting without extra libraries.
        insert_data = []
        for chunk, embedding, meta in zip(chunks, embeddings, normalized_metadata):
            pdf_id = meta.get("pdf_id", "unknown")
            page_num = meta.get("page_number", 1)
            
            # Format list of floats as pgvector string representation: [0.1,0.2,...]
            emb_str = "[" + ",".join(map(str, embedding)) + "]"
            insert_data.append((pdf_id, page_num, chunk, emb_str))

        # Perform bulk insertion
        insert_query = """
        INSERT INTO chunks (pdf_id, page_number, content, embedding)
        VALUES %s;
        """
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                logger.info(f"Inserting {len(insert_data)} chunks into the database...")
                execute_values(cur, insert_query, insert_data)
            conn.commit()
            logger.info("Successfully stored all chunks.")

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        """
        Searches the database for the top_k most semantically similar chunks based on cosine distance.
        
        Args:
            query_embedding (list[float]): The embedding vector of the search query.
            top_k (int, optional): The number of chunks to return. Defaults to 3.
            
        Returns:
            list[dict]: A list of retrieved chunks, each containing the id, pdf_id, 
                        page_number, content, and the similarity score.
                        
        Why Cosine Similarity matters:
        1. Cosine similarity measures the angle/direction difference between two vectors in a high-dimensional space.
        2. Unlike Euclidean (L2) distance, cosine similarity is scale-invariant. That is, it measures meaning 
           independent of text lengths or variations in absolute vector magnitudes. This is standard for NLP models.
        3. In pgvector, the cosine distance operator is `<=>`. Since it calculates distance (1 - cosine_similarity),
           lower distance means higher similarity. We sort ascending (lowest distance first) to get the best matches.
           
        Why top_k matters:
        - Too few (e.g. top_k = 1): We may miss critical context. The user's answer might reside across multiple parts
          of a document.
        - Too many (e.g. top_k = 10): We flood the LLM's prompt with irrelevant text, raising costs, slowing response
          times, and causing the LLM to hallucinate or miss the correct answer ("lost in the middle" effect).
        - A default value of 3 balances completeness and cost-effectiveness.
        """
        if not query_embedding:
            return []

        # Convert query embedding float list to a pgvector string representation: '[x,y,z,...]'
        query_emb_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
        # Query matching database rows using cosine distance <=>
        # We calculate the similarity score as: 1 - cosine_distance (since <=> returns cosine distance)
        search_query = """
        SELECT id, pdf_id, page_number, content, 1 - (embedding <=> %s::vector) AS similarity
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """
        
        results = []
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(search_query, (query_emb_str, query_emb_str, top_k))
                rows = cur.fetchall()
                for row in rows:
                    results.append({
                        "id": row[0],
                        "pdf_id": row[1],
                        "page_number": row[2],
                        "content": row[3],
                        "similarity": float(row[4])
                    })
                    
        logger.info(f"Retrieved {len(results)} chunks using semantic search.")
        return results
