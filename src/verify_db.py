"""
Database connection and vector operations integration test.
This script attempts to connect to the live PostgreSQL instance configured
via DATABASE_URL in .env, inserts dummy vectors, tests search, and cleans up.

To run:
    ./venv/bin/python src/verify_db.py
"""

import os
import sys
import random
from dotenv import load_dotenv

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.vector_store import VectorStore

def main():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    
    if not db_url:
        print("[SKIP] DATABASE_URL is not set in the environment or .env file.")
        print("Please configure DATABASE_URL to test database connectivity.")
        sys.exit(0)
        
    print(f"Connecting to database at: {db_url.split('@')[-1] if '@' in db_url else db_url}")
    
    try:
        # Initialize database and tables
        store = VectorStore()
        print("[SUCCESS] Connected to database and schema verified/created.")
        
        # Define mock pdf_id
        test_pdf_id = f"test_pdf_{random.randint(1000, 9999)}"
        
        # 1. Create dummy chunks and 768-dimensional embeddings
        # We will create 3 chunks:
        # - Chunk A: talks about database indexes
        # - Chunk B: talks about baking apple pies
        # - Chunk C: talks about training neural networks
        chunks = [
            "PostgreSQL offers indexes like B-Tree, Hash, and HNSW to optimize queries.",
            "To bake an apple pie, you need apples, flour, sugar, butter, and cinnamon.",
            "Neural networks are trained using backpropagation and gradient descent optimization."
        ]
        
        # We generate random 768-dimensional embeddings.
        # But to simulate similarity, we will make one specific vector stand out.
        # Let's say we want search query to be highly similar to Chunk A.
        # We'll make Chunk A's vector close to our search query vector.
        query_vector = [0.1] * 768
        
        # Chunk A's vector: very close to query_vector (with small perturbation)
        vec_a = [0.1 + random.uniform(-0.01, 0.01) for _ in range(768)]
        # Chunk B's vector: random values
        vec_b = [random.uniform(-0.5, 0.5) for _ in range(768)]
        # Chunk C's vector: random values
        vec_c = [random.uniform(-0.5, 0.5) for _ in range(768)]
        
        embeddings = [vec_a, vec_b, vec_c]
        metadata = {
            "pdf_id": test_pdf_id,
            "page_numbers": [1, 2, 3]
        }
        
        print("Inserting dummy chunks...")
        store.store_chunks(chunks, embeddings, metadata)
        print("[SUCCESS] Dummy chunks stored.")
        
        # 2. Perform search
        print("\nPerforming semantic search for the query vector...")
        # Since Chunk A's vector is extremely close to the query_vector, 
        # it should return Chunk A as the top match with a similarity close to 1.0.
        results = store.search(query_vector, top_k=3)
        
        print("\nSearch Results:")
        for idx, res in enumerate(results):
            print(f"{idx+1}. ID: {res['id']}, Page: {res['page_number']}, Similarity: {res['similarity']:.4f}")
            print(f"   Content: {res['content']}\n")
            
        # Verify that the top result is Chunk A
        if results and "PostgreSQL offers indexes" in results[0]["content"]:
            print("[SUCCESS] Semantic search correctly retrieved the most similar chunk first!")
        else:
            print("[WARNING] Semantic search did not retrieve Chunk A first. (Check vector logic).")

        # 3. Clean up test records
        print("Cleaning up test records from database...")
        connection = store._get_connection()
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM chunks WHERE pdf_id = %s;", (test_pdf_id,))
        print("[SUCCESS] Cleaned up test records.")
        
    except Exception as e:
        print(f"[ERROR] Database integration test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
