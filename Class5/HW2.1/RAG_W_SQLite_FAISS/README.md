# Hybrid RAG Pipeline for arXiv cs.CL Papers - SEMANTIC + KEYWORD SEARCH
### This project implements a Retrieval-Augmented Generation (RAG) pipeline over a collection of arXiv papers.
It downloads PDFs, extracts and chunks text, embeds chunks with SentenceTransformers, builds both a FAISS semantic index and a SQLite keyword (FTS5) index, and supports hybrid retrieval (semantic + keyword). At query time, the pipeline retrieves relevant chunks and asks an OpenAI model to generate an answer.

An evaluation script compares vector-only, keyword-only, and hybrid retrieval methods using Recall@k.

## Project workflow
1. Download recent cs.CL PDFs from arXiv.
2. Extract text from each PDF.
3. Chunk the text.
4. Create embeddings for each chunk.
5. Build a FAISS index over all chunk embeddings and save an ID→(pdf, chunk) mapping.
6. Store metadata in SQLite by loading document/chunk metadata into rag.db with an FTS5 full-text index for keyword search.
7. At query time: Embed user question. Run semantic search via FAISS + keyword search via SQLite FTS5. Merge results with hybrid fusion (weighted sum or reciprocal rank fusion).
8. Concatenate the top-N chunks into context and ask the LLM to answer.
9. Return the answer along with the sources (PDF paths/chunk refs).
10. Evaluation (evaluate_retrieval.py): Load test queries from ground_truth.json. Compare vector-only, keyword-only, and hybrid retrieval. Report Recall@k and per-query diagnostics (hits, found, recall).

## 📁 Project folder structure

```
RAG_W_SQLite_FAISS/
├─ master_setup.py         # orchestrates end-to-end pipeline
├─ query.py                # query interface with semantic / keyword / hybrid modes
├─ retrieval.py            # search utilities (FAISS, SQLite FTS5, hybrid fusion)
├─ evaluate_retrieval.py   # evaluation script (Recall@k across methods)
├─ step1_scrape.py         # download PDFs from arXiv
├─ step2_extract.py        # extract text + OCR
├─ step3_chunk.py          # split text into chunks
├─ step4_embed.py          # embed chunks with SentenceTransformers
├─ step5_faiss.py          # build FAISS index + id mapping
├─ rag.db                  # SQLite DB with FTS5 index (created at runtime)
├─ pdfs/                   # downloaded PDFs (created at runtime)
├─ texts.json              # extracted text per PDF (created)
├─ chunks.json             # chunked text per PDF (created)
├─ documents.json          # [{chunk, embedding}] per PDF (created)
├─ id_mapping.json         # FAISS id → (pdf, chunk_idx) mapping (created)
├─ faiss.index             # FAISS vector index (created)
├─ ground_truth.json       # evaluation queries + relevant docs (you create)
└─ .env                    # your OpenAI API key (you create)
```

## 📦 Requirements
Make sure you have Python 3.9+ and virtualenv or venv set up.

```
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
## ➕ Add your OpenAI API key
Create a .env file in the project root:
```
OPENAI_API_KEY=sk-...your-key...
```
## 🚀 Running the Pipeline
The workflow is orchestrated by master_setup.py.

### Full Run (scrape, extract, chunk, embed, index)
```
python3 master_setup.py
```

## ❓ Querying the RAG System
Once setup is complete, you can ask questions with query.py.

```
python query.py "TYPE YOUR QUESTION HERE" --mode hybrid --k 5 --weight_sem 0.5
```

At inference, you can tune the hyperparameters:
```
--mode hybrid (retrieval mode: semantic / keyword / hybrid)
--k 5 (how many chunks is returned)
--weight_sem 0.5 (how much the model's answer tilts towards semantic search)
```

## Evaluating the RAG System
Once you have collected an amount of Q+A pairs, you can evaluate the system to see which method retrieves relevant documents better.
First enter the query and relevant doc paths into the ground_truth.json
This is the datasource used by evaluate_retrieval.py

```
python evaluate_retrieval.py
```

## 📂 Outputs

pdfs/ → Downloaded papers from arXiv

texts.json → Extracted raw text per PDF

chunks.json → Text split into chunks

documents.json → Embedded document chunks

id_mapping.json → Index → (doc, chunk) mapping

faiss.index → FAISS vector index

## 🛠 Notes
Chunk size and overlap can be tuned in step3_chunk.py.

Embedding model can be swapped in step4_embed.py or query.py.

The query_rag() function retrieves top-k chunks and feeds them to GPT.
