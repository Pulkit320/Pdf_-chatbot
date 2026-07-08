# Phase 8: Testing and RAG Retrieval Evaluation

Building Large Language Model (LLM) applications introduces a unique software engineering challenge: **non-determinism**. Unlike standard applications where the same input always yields the exact same output, LLMs can generate varying responses. 

To build robust, production-grade AI systems, developers must implement a double-layered testing strategy: **traditional unit testing** (for code correctness) and **retrieval/answer evaluation** (for content correctness).

---

## 1. Traditional Code Correctness vs. Generative Content Correctness

| Aspect | Traditional Unit Tests | RAG & LLM Evaluation |
| :--- | :--- | :--- |
| **Objective** | Verify **code correctness** (no syntax errors, proper type mappings, data flows, database cursor commits, logic limits). | Verify **content correctness** (semantic relevance, search accuracy, answer grounding, lack of hallucination). |
| **Method** | Mock network calls, assert exact outputs, test boundary parameters. | Run queries against a live/test database, compare results to expected ground-truth metadata or use LLMs to judge text. |
| **Example** | Test that calling `chunk_text("abc", size=2, overlap=1)` outputs exactly `["ab", "bc"]`. | Test that asking "When does the Fall Semester start?" returns page 1 of the syllabus in the top 3 nearest database chunks. |
| **Why it fails alone** | A script with 100% test coverage can pass perfectly while the bot retrieves completely irrelevant chunks and generates gibberish. | A retrieval evaluation script cannot verify if your backend HTTP router throws a 500 error on file streams. |

---

## 2. RAG Retrieval Metrics Explained

In any Retrieval-Augmented Generation system, the quality of the generated answer is strictly bound by the quality of the retrieved database chunks. We evaluate retrieval using three key mathematical metrics:

### A. Hit Rate (Hit Rate@K)
* **What it measures:** The probability that at least one of the top K retrieved chunks is from the correct expected document page.
* **Why it matters:** Hit rate is a binary indicator of search success. If Hit Rate is 0%, the LLM has zero chance of answering the user's question, since the required fact never made it into the prompt context.
* **Formula:** $\frac{\text{Queries with } \ge 1 \text{ relevant chunk retrieved}}{\text{Total queries evaluated}}$

### B. Precision (Precision@K)
* **What it measures:** The proportion of the top K retrieved chunks that are actually relevant to the query.
* **Why it matters:** In a RAG context, low precision means we are feeding the LLM "noisy" or irrelevant text chunks alongside the useful ones. This degrades performance (the LLM can get confused—known as the "lost in the middle" effect), increases response latency, and raises API costs by bloating the token count.
* **Formula:** $\frac{\text{Count of relevant chunks retrieved in top } K}{K}$

### C. Recall (Recall@K)
* **What it measures:** The proportion of all expected target pages/chunks that were successfully retrieved in the top K.
* **Why it matters:** If a question requires details spread across three different pages (e.g., "What are the rules for homework, exams, and grading?"), retrieving only page 1 gives the LLM incomplete data. Low recall leads to partial answers or logical hallucinations.
* **Formula:** $\frac{\text{Count of expected pages found in top } K}{\text{Total expected target pages}}$

---

## 3. Answer-Quality Evaluation (Generative Layer)

While retrieval metrics evaluate the database lookup, we must also measure the **generative output** (the text Gemini creates). There are three standard criteria for this:

1. **Groundedness / Faithfulness:** Is the answer derived *only* from the provided context? If the LLM makes an assertion that cannot be found in the context chunks, it is flagged as a hallucination.
2. **Answer Relevance:** Does the generated text directly address the user's question? If the bot writes a paragraph of correct facts from the document that doesn't actually answer the prompt, relevance is low.
3. **Context Recall:** Did the generated answer incorporate all the key facts present in the retrieved context chunks?

### Evaluation Methodologies
* **Human-in-the-Loop:** Subject matter experts manually rate answers. High accuracy but slow and expensive.
* **LLM-as-a-Judge:** We prompt a larger, powerful model (like Gemini 1.5 Pro) with the question, context, and generated answer, asking it to rate groundedness and relevance on a scale of 1-5. This is fast, automated, and scales well.
* **Frameworks:** Production apps use automated frameworks like **Ragas** or **TruLens** to calculate these scores programmatically.

---

## 4. Running the Tests and Evaluation

### A. Run Unit Tests (pytest)
To execute all the newly written pytest unit tests for code correctness:
```bash
./venv/bin/pytest tests/
```

### B. Run Retrieval Evaluation Script
To run the semantic retrieval evaluation on our live database chunks:
```bash
./venv/bin/python tests/evaluate_rag.py
```
*Example Output:*
```text
Connecting to live database for RAG retrieval evaluation...
Loaded 4 evaluation test cases.

Query                                  | Hit?  | Prec@3  | Recall@3 | Citations (PDF, Page)
-----------------------------------------------------------------------------------------------
When does the Fall Semester start?     | 1     | 0.67    | 1.00     | Academic Timelines.pdf P.1, Academic Timelines.pdf P.2, SDD.pdf P.2
When is Capstone Review Meeting II?    | 1     | 0.33    | 1.00     | Academic Timelines.pdf P.1, Academic Timelines.pdf P.2, SDD.pdf P.2
what is Syntax Directed Definitions    | 1     | 1.00    | 1.00     | SDD.pdf P.1, SDD.pdf P.2, SDD.pdf P.3
What is SDD                            | 1     | 0.67    | 1.00     | SDD.pdf P.1, SDD.pdf P.2, Academic Timelines.pdf P.2
-----------------------------------------------------------------------------------------------
AVERAGE                                | 1.00  | 0.67    | 1.00     |

Evaluation Summary:
  * Hit Rate@3:  100.0%  (Percent of queries where at least one correct chunk was found)
  * Precision@3: 67.0%   (Percent of retrieved chunks that were actually relevant)
  * Recall@3:    100.0%  (Percent of expected source pages successfully retrieved)
```

---

## Learning Outcomes

By completing Phase 8, you can now:
1. **Contrast** code correctness (verified via unit tests) with content correctness (verified via RAG semantic evaluations).
2. **Defend** why testing generative AI applications requires mock objects to isolate network behaviors from logic tests.
3. **Calculate** search metrics including Hit Rate, Precision@K, and Recall@K.
4. **Calibrate** retrieval parameters based on evaluation outputs to balance model focus (precision) with completeness (recall).
5. **Describe** how the LLM-as-a-judge paradigm automates semantic evaluations in production.
