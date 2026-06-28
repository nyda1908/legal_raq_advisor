# LegalRAG

A retrieval-augmented generation (RAG) system for querying legal contracts in natural language. Upload one or more contract PDFs, ask questions, and get answers grounded in the actual contract text — with source attribution down to the page level.

Built on top of [abdimussa87/legal_contract_advisor_rag](https://github.com/abdimussa87/legal_contract_advisor_rag) with four targeted improvements to retrieval quality, answer faithfulness, and pipeline observability.

---

## Pipeline

```
PDF upload → raw text → token-based chunks → vector store → hybrid retrieval → reranking → classification → LLM answer
```

Each stage is a separate utility module:

| Stage | File | What it does |
|-------|------|-------------|
| Extract | `pdf_utils.py` | PyPDF2 extracts text per page, wraps in Document objects with source + page metadata |
| Chunk | `text_splitter_utils.py` | RecursiveCharacterTextSplitter splits into 512-token chunks with 50-token overlap |
| Embed | `vector_store_utils.py` | OpenAI text-embedding-ada-002 embeds each chunk, persisted to ChromaDB |
| Retrieve | `vector_store_utils.py` | Hybrid BM25 + dense retrieval via EnsembleRetriever, reranked by cross-encoder |
| Classify | `clause_classifier_utils.py` | Zero-shot GPT-3.5 labels each retrieved chunk by clause type |
| Generate | `langchain_utils.py` | GPT-3.5-turbo answers from classified, attributed context via LangChain LCEL |

---

## Modifications

### #0 — Source metadata tracking + token-based chunking
**Problem:** the original codebase merged all uploaded PDFs into one plain string, losing track of which clause came from which contract and which page.

**Fix:** switched from plain strings to LangChain `Document` objects that carry `source` (filename) and `page` metadata through every stage of the pipeline — extraction, chunking, embedding, and retrieval. Also replaced `length_function=len` (character counting) with `tiktoken` token counting, since `text-embedding-ada-002` has a token limit, not a character limit.

**Files:** `pdf_utils.py`, `text_splitter_utils.py`, `vector_store_utils.py`

---

### #1 — Hybrid retrieval (BM25 + dense)
**Problem:** dense retrieval alone fails on exact keyword matches. A query for "Section 12.3" or a specific dollar amount may not surface the right chunk if semantically similar but textually different chunks score higher.

**Fix:** combined BM25 sparse retrieval (keyword-based) with ChromaDB dense retrieval (semantic) using LangChain's `EnsembleRetriever` at 50/50 weighting. BM25 handles exact matches; dense handles synonyms and paraphrases. Together they cover each other's failure modes.

**Files:** `vector_store_utils.py`

---

### #2 — Cross-encoder reranking
**Problem:** cosine similarity is a fast but coarse relevance signal. The top-ranked chunk by embedding similarity is not always the most relevant chunk for a specific question.

**Fix:** after the ensemble retriever fetches 20 candidates, a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescores each chunk by reading the question and chunk together — not separately. Top 7 chunks are passed to the LLM. This reduces hallucination risk and prompt token cost simultaneously.

**Files:** `vector_store_utils.py`

---

### #3 — Zero-shot clause-type classification
**Problem:** retrieved chunks had no semantic label — the LLM received raw text with no indication of whether it was reading a termination clause, a payment clause, or something else.

**Fix:** after reranking, each chunk is classified into one of 9 clause types (termination, indemnity, payment, confidentiality, liability, intellectual property, dispute resolution, governing law, other) using a zero-shot GPT-3.5 prompt. The clause type is stored in chunk metadata and surfaced in the LLM context as a structured label.

**Files:** `clause_classifier_utils.py`, `langchain_utils.py`

Context sent to the LLM now looks like:
```
[TERMINATION CLAUSE - acme_contract.pdf p.3]
The agreement may be terminated by either party with 30 days written notice...

[PAYMENT CLAUSE - acme_contract.pdf p.7]
Invoices shall be settled within 30 days of receipt...
```

---

### #4 — RAGAS evaluation
**Problem:** no objective measure of whether the pipeline modifications actually improved output quality.

**Fix:** integrated RAGAS evaluation across four metrics:
- **Faithfulness** — is the answer grounded in the retrieved context?
- **Answer relevancy** — does the answer address the question asked?
- **Context precision** — of the retrieved chunks, how many were actually useful?
- **Context recall** — did retrieval surface everything needed for a complete answer?

Run `app/evaluate.py` with a test contract and manually written ground truths to get scores. Results are saved to `ragas_results.csv`.

**Files:** `ragas_eval_utils.py`, `evaluate.py`

---

## Tech stack

- **LLM:** GPT-3.5-turbo (OpenAI)
- **Embeddings:** text-embedding-ada-002 (OpenAI)
- **Vector store:** ChromaDB
- **Retrieval:** LangChain EnsembleRetriever (BM25 + dense)
- **Reranking:** cross-encoder/ms-marco-MiniLM-L-6-v2 (HuggingFace)
- **Framework:** LangChain LCEL
- **UI:** Streamlit
- **Evaluation:** RAGAS

---

## Setup

```bash
git clone https://github.com/nyda1908/legal_contract_advisor_rag
cd legal_contract_advisor_rag
pip install -r requirements.txt
```

Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_key_here
```

Run the app:
```bash
cd app
streamlit run main.py
```

Run evaluation:
```bash
cd app
python evaluate.py
```

---

## Project structure

```
app/
├── main.py                          # Streamlit UI and orchestration
└── utils/
    ├── pdf_utils.py                 # PDF extraction with metadata
    ├── text_splitter_utils.py       # Token-based chunking
    ├── vector_store_utils.py        # Hybrid retrieval + reranking
    ├── langchain_utils.py           # LangChain LCEL chain
    ├── clause_classifier_utils.py   # Zero-shot clause classification
    └── ragas_eval_utils.py          # RAGAS evaluation wrapper
evaluate.py                          # Standalone evaluation script
requirements.txt
```

---

## Future work

- CUAD fine-tuned clause classifier — fine-tune a BERT-based model on the CUAD dataset (41 clause categories, 500+ contracts) for higher classification accuracy than zero-shot prompting
- Multi-contract comparison — given a query, surface and compare the relevant clause across multiple uploaded contracts side by side
- BM25 persistence — currently the BM25 index rebuilds on every app restart; persisting it would reduce cold-start latency for large contract libraries
