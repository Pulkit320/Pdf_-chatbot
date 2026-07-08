"""
Custom RAG retrieval evaluation script.
This script compares expected vs actual retrieved chunks for a set of sample questions.
It computes Hit Rate, Precision@K, and Recall@K, summarizing retrieval performance.

To run:
    ./venv/bin/python tests/evaluate_rag.py
"""

import os
import sys
from dotenv import load_dotenv

# Ensure parent directory is in sys.path so we can import from 'src'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.embeddings.embedder import get_embedding
from src.retrieval.vector_store import VectorStore


def run_evaluation():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("[ERROR] DATABASE_URL not set in .env. Cannot run live RAG evaluation.")
        sys.exit(1)

    print("Connecting to live database for RAG retrieval evaluation...")
    try:
        store = VectorStore()
    except Exception as e:
        print(f"[ERROR] Could not connect to vector database: {e}")
        sys.exit(1)

    # Define our ground-truth evaluation dataset
    # Each test case has:
    # - question: The query string to send to the embedder and Vector Store.
    # - expected_pdf: The exact name/id of the source PDF.
    # - expected_pages: A list of relevant page numbers containing the answer.
    # - expected_keywords: A list of strings that should appear in the retrieved text.
    eval_dataset = [
        {
            "question": "When does the Fall Semester start?",
            "expected_pdf": "Fall Semester 2026-2027 - Academic Timelines.pdf",
            "expected_pages": [1],
            "expected_keywords": ["July 6", "Instructional Day"]
        },
        {
            "question": "When is Capstone Review Meeting II?",
            "expected_pdf": "Fall Semester 2026-2027 - Academic Timelines.pdf",
            "expected_pages": [1],
            "expected_keywords": ["Capstone", "September 23", "Meeting - II"]
        },
        {
            "question": "what is Syntax Directed Definitions",
            "expected_pdf": "SDD.pdf",
            "expected_pages": [1, 2, 3],
            "expected_keywords": ["Syntax", "Directed", "SDD", "translation"]
        },
        {
            "question": "What is SDD",
            "expected_pdf": "SDD.pdf",
            "expected_pages": [1, 2],
            "expected_keywords": ["SDD", "Syntax"]
        }
    ]

    print(f"Loaded {len(eval_dataset)} evaluation test cases.\n")
    print(f"{'Query':<38} | {'Hit?':<5} | {'Prec@3':<7} | {'Recall@3':<8} | {'Citations (PDF, Page)'}")
    print("-" * 95)

    total_hits = 0
    total_precision = 0.0
    total_recall = 0.0
    k = 3

    for case in eval_dataset:
        question = case["question"]
        expected_pdf = case["expected_pdf"]
        expected_pages = case["expected_pages"]
        expected_keywords = case["expected_keywords"]

        # 1. Embed query
        try:
            query_emb = get_embedding(question, task_type="retrieval_query")
        except Exception as e:
            print(f"[ERROR] Failed to embed query '{question}': {e}")
            continue

        # 2. Retrieve chunks (top_k=3)
        retrieved_chunks = store.search(query_emb, top_k=k)

        # 3. Assess relevance of retrieved chunks
        hits_count = 0
        pages_found = set()
        citations = []

        for chunk in retrieved_chunks:
            pdf_match = chunk["pdf_id"] == expected_pdf
            page_match = chunk["page_number"] in expected_pages
            
            # A retrieved chunk is deemed relevant if it matches the expected PDF name
            # and is from the expected page or contains the expected key phrases.
            has_keywords = any(kw.lower() in chunk["content"].lower() for kw in expected_keywords)
            
            is_relevant = pdf_match and (page_match or has_keywords)
            
            if is_relevant:
                hits_count += 1
                pages_found.add(chunk["page_number"])
                
            citations.append(f"{chunk['pdf_id']} P.{chunk['page_number']}")

        # 4. Calculate metrics
        hit = 1 if hits_count > 0 else 0
        precision = hits_count / len(retrieved_chunks) if retrieved_chunks else 0.0
        
        # Recall is the proportion of expected pages that were found among the relevant retrieved pages
        recall = len(pages_found.intersection(expected_pages)) / len(expected_pages) if expected_pages else 0.0

        # Accumulate metrics
        total_hits += hit
        total_precision += precision
        total_recall += recall

        citation_str = ", ".join(citations)
        print(f"{question[:38]:<38} | {hit:<5} | {precision:<7.2f} | {recall:<8.2f} | {citation_str}")

    # Calculate overall dataset averages
    num_cases = len(eval_dataset)
    avg_hit_rate = total_hits / num_cases
    avg_precision = total_precision / num_cases
    avg_recall = total_recall / num_cases

    print("-" * 95)
    print(f"{'AVERAGE':<38} | {avg_hit_rate:<5.2f} | {avg_precision:<7.2f} | {avg_recall:<8.2f} |")
    print(f"\nEvaluation Summary:")
    print(f"  * Hit Rate@3:  {avg_hit_rate * 100:.1f}%  (Percent of queries where at least one correct chunk was found)")
    print(f"  * Precision@3: {avg_precision * 100:.1f}%  (Percent of retrieved chunks that were actually relevant)")
    print(f"  * Recall@3:    {avg_recall * 100:.1f}%  (Percent of expected source pages successfully retrieved)")


if __name__ == "__main__":
    run_evaluation()
