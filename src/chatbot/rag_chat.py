"""
This module implements the final integration layer of the PDF Chatbot: the 
Retrieval-Augmented Generation (RAG) pipeline. It combines Phase 1 text extraction, 
Phase 2 chunking, Phase 3 embedding generation, Phase 4 database storage/retrieval, 
and prompts the Gemini model via the google-genai SDK for grounded answers.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Ensure we can import modules from the parent project directories if executed directly.
# Since our imports are absolute and start with 'src.', the project root folder
# (which is the parent of the 'src' directory) must be appended to sys.path.
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.ingestion.pdf_reader import extract_text
from src.embeddings.chunker import chunk_text
from src.embeddings.embedder import get_embedding
from src.retrieval.vector_store import VectorStore

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables from the .env file
load_dotenv()

# Why the same API Key works for both Embeddings (Phase 3) and Generation (this phase):
# Google AI Studio provides developer access to a unified suite of Generative AI services.
# A single Google AI Studio API key (configured under GOOGLE_API_KEY or GEMINI_API_KEY)
# is authenticated globally against your account/project. This unified auth enables 
# requests to both the representation models (such as `text-embedding-004`) and 
# generative model endpoints (such as `gemini-2.0-flash`) under the same billing 
# and quota allocation.
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "API Key not found. Please define GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
    )

# Initialize the new Google GenAI client
# Why we use the new google-genai SDK:
# Google released the `google-genai` SDK as the unified library to replace the legacy 
# `google-generativeai` library. The new SDK features a cleaner `genai.Client()` 
# instantiation pattern and standardizes APIs across Gemini developer platforms (like AI Studio 
# and Vertex AI).
client = genai.Client(api_key=api_key)


def ingest_pdf(pdf_path: str, pdf_id: str = None) -> None:
    """
    Ingests a PDF document by extracting, chunking, embedding, and storing it
    in the PostgreSQL vector store.
    
    Args:
        pdf_path (str): Path to the source PDF file.
        pdf_id (str, optional): Custom unique ID for the document. Defaults to the filename.
    """
    if not pdf_id:
        pdf_id = os.path.basename(pdf_path)

    logger.info(f"Starting ingestion for: {pdf_path} (ID: {pdf_id})")
    
    # Phase 1: Extract text page-by-page
    pages = extract_text(pdf_path)
    
    chunks_to_store = []
    embeddings_to_store = []
    metadata_to_store = []
    
    # Phase 2: Chunk text page-by-page and compile coordinates
    for page in pages:
        if page["is_empty_or_scanned"]:
            logger.warning(f"Skipping empty or unscannable page {page['page_number']}")
            continue
            
        page_text = page["text"]
        page_num = page["page_number"]
        
        # Split text into overlapping chunks
        page_chunks = chunk_text(page_text)
        
        for chunk in page_chunks:
            # Phase 3: Generate embedding using text-embedding-004
            # We explicitly use task_type="retrieval_document" to tell the embedding model 
            # that this content will reside in a retrieval index.
            embedding = get_embedding(chunk, task_type="retrieval_document")
            
            chunks_to_store.append(chunk)
            embeddings_to_store.append(embedding)
            metadata_to_store.append({
                "pdf_id": pdf_id,
                "page_number": page_num
            })

    if not chunks_to_store:
        logger.warning("No text was extracted from the PDF. Database insert skipped.")
        return

    # Phase 4: Store chunks & embeddings in the database
    store = VectorStore()
    store.store_chunks(chunks_to_store, embeddings_to_store, metadata_to_store)
    logger.info(f"Ingestion complete. Successfully stored {len(chunks_to_store)} chunks.")


def answer_question(question: str) -> dict:
    """
    Performs retrieval-augmented generation to answer a user's question.
    
    Workflow:
    1. Embeds the question using task_type="retrieval_query".
    2. Searches the Vector Store database for the top 3 matches based on cosine similarity.
    3. Formats the matching chunks into a context block.
    4. Prompts the Gemini generation model (gemini-2.0-flash) with strict system instructions.
    
    Args:
        question (str): The question to answer.
        
    Returns:
        dict: A dictionary containing:
            - "answer" (str): The generated response text.
            - "sources" (list[dict]): The source citations (pdf_id, page_number, similarity).
    """
    # Step 1: Embed the search query
    # Why task_type="retrieval_query":
    # Google's text-embedding-004 model is optimized to compare two different types of vectors: 
    # documents (retrieval_document) and queries (retrieval_query). Embedding the question 
    # as a "retrieval_query" projects the search phrase into the same conceptual space 
    # as the indexed documents, yielding substantially more accurate nearest neighbor matches.
    query_embedding = get_embedding(question, task_type="retrieval_query")
    
    # Step 2: Search PostgreSQL vector store for top 3 matching chunks
    store = VectorStore()
    results = store.search(query_embedding, top_k=3)
    
    if not results:
        return {
            "answer": "I'm sorry, but there are no document chunks available in the database to answer your question.",
            "sources": []
        }
        
    # Step 3: Format the context text and extract source metadata
    context_parts = []
    sources = []
    seen_sources = set()
    
    for idx, res in enumerate(results):
        context_parts.append(
            f"--- Context Segment {idx+1} (Source: {res['pdf_id']}, Page {res['page_number']}) ---\n"
            f"{res['content']}"
        )
        
        # deduplicate sources for the final output representation
        source_key = (res["pdf_id"], res["page_number"])
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "pdf_id": res["pdf_id"],
                "page_number": res["page_number"],
                "similarity": res["similarity"]
            })
            
    context_text = "\n\n".join(context_parts)
    
    # Why our prompt structure and system instructions reduce LLM hallucination:
    # 1. Knowledge Anchoring: Large Language Models are prone to "hallucinating" facts by drawing
    #    information from their pre-training weights when they lack specific data. A strict system 
    #    instruction ("answer ONLY from the provided Context") overrides this behavior, locking 
    #    the model's reference frame to the retrieved database rows.
    # 2. Strict Fallback: Explicitly instructing the model to declare "I'm sorry..." if the answer 
    #    cannot be found prevents the model from generating creative, plausible-sounding guesses.
    # 3. Source Auditing: Requiring page citations ([Page X]) ensures that every claim the model 
    #    makes is mapped to a specific document chunk. If a statement cannot be grounded to a citation,
    #    it cannot be legally generated.
    # 4. Temperature Optimization: We set temperature=0.0. Temperature controls creativity/randomness.
    #    Lowering it to 0.0 makes the model's token selection highly deterministic, focusing purely on 
    #    exact contextual mapping and reducing logical drifts.
    system_instruction = (
        "You are a helpful and precise assistant designed to answer questions based ONLY on the provided document context.\n"
        "Strictly adhere to the following rules:\n"
        "1. Answer the question using ONLY the facts explicitly mentioned in the provided Context. Do NOT use any external or background knowledge.\n"
        "2. If the answer cannot be determined or inferred from the provided Context, respond with: "
        "\"I'm sorry, but the provided context does not contain the answer to your question.\"\n"
        "3. For every statement or claim you make in your answer, you MUST cite the source page number(s) in brackets (e.g., [Page X]).\n"
    )
    
    prompt = f"""Context:
{context_text}

Question: {question}

Answer:"""
    
    try:
        # Step 4: Query the generation model via google-genai
        # We use 'gemini-2.5-flash' as it is the current standard production generation model,
        # featuring high speed, low cost, and strong context-adherence capabilities.
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0
            )
        )
        
        return {
            "answer": response.text.strip(),
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"Gemini generation request failed: {e}")
        raise RuntimeError(f"Failed to generate answer from Gemini model: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF Chatbot RAG Pipeline CLI Utility")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Process and store a PDF file")
    ingest_parser.add_argument("pdf_path", type=str, help="Local path to the PDF document")
    ingest_parser.add_argument("--pdf-id", type=str, default=None, help="Custom document ID (defaults to filename)")
    
    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Query the RAG chatbot")
    query_parser.add_argument("question", type=str, help="The query string to search and answer")
    
    # Interactive chat subcommand
    chat_parser = subparsers.add_parser("chat", help="Start an interactive terminal chat session")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        ingest_pdf(args.pdf_path, args.pdf_id)
    elif args.command == "query":
        try:
            res = answer_question(args.question)
            print("\n" + "="*50)
            print(f"QUESTION: {args.question}")
            print("="*50)
            print(f"ANSWER:\n{res['answer']}")
            print("="*50)
            print("CITED SOURCES:")
            for src in res["sources"]:
                print(f" - Document: {src['pdf_id']} | Page: {src['page_number']} | Similarity Score: {src['similarity']:.4f}")
            print("="*50 + "\n")
        except Exception as err:
            print(f"Query Error: {err}")
            sys.exit(1)
    elif args.command == "chat":
        print("\n" + "="*55)
        print("   Interactive PDF Chatbot (RAG Terminal Session)")
        print("   Type 'exit', 'quit', or hit Ctrl+C to terminate.")
        print("="*55 + "\n")
        
        while True:
            try:
                user_query = input("Chat > ")
                if user_query.strip().lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break
                if not user_query.strip():
                    continue
                
                print("Retrieving and generating answer...")
                res = answer_question(user_query)
                
                print(f"\nAnswer:\n{res['answer']}\n")
                print("Sources:")
                for src in res["sources"]:
                    print(f"  * Page {src['page_number']} (Similarity: {src['similarity']:.4f})")
                print("-" * 55 + "\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as err:
                print(f"Error during execution: {err}\n")
    else:
        parser.print_help()
