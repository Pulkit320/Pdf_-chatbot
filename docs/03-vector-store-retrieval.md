# Phase 4: Vector Database and Semantic Retrieval

In this guide, we will implement the fourth phase of our PDF Chatbot pipeline: **Vector Storage and Semantic Retrieval**.

Once text chunks are converted into 768-dimensional mathematical coordinates (embeddings) by Gemini, we need a high-performance database capable of indexing and querying these coordinates. We use **PostgreSQL** with the **pgvector extension** to store document segments and perform **nearest neighbor semantic search**.

---

## 1. Retrieval Pipeline Flow

Below is a word-based diagram illustrating how a user's question goes from a text query to the most semantically relevant document chunks, which are then passed to the LLM.

```
+------------------+
|  User's Question |  e.g., "How do we optimize Postgres index speed?"
+--------+---------+
         |
         v
+--------+---------+
| Gemini Embedder  |  Converts text query to a 768-dimensional vector
|  (retrieval_query) |  using models/text-embedding-004
+--------+---------+
         |
         v
  [ Query Vector ]    [0.153, -0.048, 0.982, ... 768 floats]
         |
         v
+--------+----------------------------+
| PostgreSQL + pgvector Database      |  Performs cosine distance search (<=>):
|                                      |  SELECT content FROM chunks
| SELECT ... ORDER BY embedding <=> %s |  ORDER BY embedding <=> query_vector
| LIMIT 3;                             |  LIMIT 3;
+--------+----------------------------+
         |
         +-----------------+
         |                 |
         v                 v
     [Chunk 1]         [Chunk 2]         [Chunk 3]   (Top 3 nearest segments)
         |                 |                 |
         +-----------------+-----------------+
                           |
                           v
+--------------------------+------------------+
| LLM Prompt Injection (Context)              |  "Use the following facts to answer..."
+--------------------------+------------------+
                           |
                           v
+--------------------------+------------------+
| Gemini LLM (Answer)                         |  "To optimize index speed in Postgres..."
+---------------------------------------------+
```

---

## 2. Setting Up a Free Hosted Postgres DB with pgvector

To run this pipeline, we need a PostgreSQL database that supports the `pgvector` extension. To avoid local database installation issues, we recommend using a free hosted PostgreSQL provider such as **Neon** or **Supabase**.

### Option A: Neon (Recommended)
1. **Sign Up:** Go to [neon.tech](https://neon.tech/) and sign up with your GitHub or Google account.
2. **Create Project:** Set up a new project. Neon will automatically spin up a serverless PostgreSQL database for you.
3. **Get connection string:** Under your project Dashboard, locate the **Connection Details** section. Select the language/driver as `psycopg` or simply copy the **Connection string** (URI format) starting with `postgresql://`.
4. **Configure `.env`:** Paste the string in your [.env](file:///home/pulkit/projects/pdf_chatbot/.env) file:
   ```env
   DATABASE_URL="postgresql://username:password@ep-some-host.us-east-2.aws.neon.tech/neondb?sslmode=require"
   ```

### Option B: Supabase
1. **Sign Up:** Go to [supabase.com](https://supabase.com/) and create a project.
2. **Create Database:** Initialize a new PostgreSQL database.
3. **Get connection URL:** Go to **Project Settings** -> **Database** and copy the **URI connection string** from the Connection Pooler or Direct Connection section.
4. **Configure `.env`:** Paste the string in your [.env](file:///home/pulkit/projects/pdf_chatbot/.env) file.

---

## 3. Enabling pgvector in PostgreSQL

The `pgvector` extension must be registered inside PostgreSQL before you can use the `vector` datatype or vector search operators. 

In our Python class, we automatically run:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```
This command registers the `vector` data type and the operators (`<=>`, `<->`, `<#>`) inside the database schema. If the database is hosted on Supabase or Neon, this extension is pre-installed and only needs to be enabled using this query.

---

## 4. Understanding Search Math: Cosine Distance vs. L2 Distance

When comparing two vectors, there are two primary distance metrics used in search:

### A. Cosine Distance (`<=>`)
* **Mathematical representation:** $1 - \text{cosine similarity}$ (where similarity is the cosine of the angle between two vectors).
* **Why we use it:** Cosine similarity measures the **direction** of the vectors rather than their length (magnitude). In NLP and document retrieval, two text segments might be semantically identical but differ in word count. Cosine similarity is scale-invariant; it evaluates the similarity of the meaning without being skewed by document size.
* **In pgvector:** We order by `<=>` in ascending order (`ORDER BY embedding <=> query_embedding ASC`). Since a smaller distance represents a higher similarity, sorting ascending returns the closest semantic matches.

### B. L2 (Euclidean) Distance (`<->`)
* **Mathematical representation:** The straight-line distance between two points in high-dimensional space.
* **Why it's less ideal here:** L2 distance is heavily affected by vector magnitude (which correlates with document length or token frequency). If one chunk is significantly longer, its embedding might have a larger magnitude, pushing it farther away from the query in Euclidean space, even if its semantic topic is a perfect match.

---

## 5. The Importance of `top_k`

`top_k` controls how many nearest neighbor chunks are retrieved from the database to be sent to the LLM. 

| Setting | Impact on the System |
| :--- | :--- |
| **Too Small ($k=1$)** | **Information Loss:** The complete answer to a user's question might be split across multiple pages. Only pulling one chunk results in incomplete context and incorrect/hallucinated answers from the LLM. |
| **Too Large ($k \ge 10$)** | **Context Flooding:** Sending too many chunks introduces noisy, irrelevant text. This degrades search precision, causes the LLM to get confused ("lost in the middle" effect), increases response latency, and dramatically raises API cost (more tokens). |
| **Optimal Default ($k=3$)** | **Balanced Retrieval:** 3 chunks typically provide enough surrounding context from different pages without overwhelming the LLM's prompt size. |

---

## 6. Code Walkthrough

The vector store database implementation is located in [vector_store.py](file:///home/pulkit/projects/pdf_chatbot/src/retrieval/vector_store.py).

### A. Table Creation and pgvector Registry
```python
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
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
```
* **Why it matters:** The column datatype `vector(768)` specifies that each embedding vector will contain exactly 768 elements. This aligns precisely with Google's `text-embedding-004` output dimension. Attempting to insert a vector of any other dimension will result in a database constraint error, ensuring data integrity.

### B. Bulk Ingestion
```python
insert_data = []
for chunk, embedding, meta in zip(chunks, embeddings, normalized_metadata):
    emb_str = "[" + ",".join(map(str, embedding)) + "]"
    insert_data.append((pdf_id, page_num, chunk, emb_str))

insert_query = "INSERT INTO chunks (pdf_id, page_number, content, embedding) VALUES %s;"
execute_values(cur, insert_query, insert_data)
```
* **Why it matters:** 
  1. We format the python list of floats `[0.1, 0.2, ...]` to pgvector's string representation `"[0.1,0.2,...]"`. PostgreSQL accepts this string format natively and casts it into the `vector` type.
  2. We use `psycopg2.extras.execute_values` to perform a single **bulk insert** query instead of inserting each chunk in a separate loop. This reduces round-trips to the hosted database, improving upload speed by up to 10x.

### C. Semantic Search
```python
search_query = """
SELECT id, pdf_id, page_number, content, 1 - (embedding <=> %s::vector) AS similarity
FROM chunks
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""
```
* **Why it matters:** We order by `<=>` ascending. To return a user-friendly similarity score, we compute `1 - (embedding <=> query)`. This maps the raw distance back to a standard cosine similarity score where `1.0` represents an identical semantic match and `0.0` represents orthogonal (independent) vectors.

---

## 7. Verification and Testing

### Automated Unit Tests
To verify implementation logic offline without setting up database servers, we run unit tests with mock connections:
```bash
./venv/bin/python -m unittest src/test_vector_store.py
```

### Manual Integration Test
If you have configured `DATABASE_URL` in `.env`, verify your connection and vector search correctness using:
```bash
./venv/bin/python src/verify_db.py
```

---

## Learning Outcomes

By completing Phase 4, you can now:
1. **Explain** how the RAG (Retrieval-Augmented Generation) flow routes a user query into nearest neighbor database context for LLMs.
2. **Deploy** and configure a cloud-hosted serverless PostgreSQL database and enable custom SQL extensions (`pgvector`).
3. **Compare** Cosine Distance and L2 (Euclidean) Distance and defend why cosine similarity is the standard metric for text semantic queries.
4. **Perform** batch data uploads in PostgreSQL using `psycopg2`'s fast bulk-insertion features.
5. **Calibrate** retrieval constraints (`top_k`) to optimize information completeness against prompt size and API tokens cost.
