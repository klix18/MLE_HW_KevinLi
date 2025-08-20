# RAG Pipeline for arXiv cs.CL Papers
### This project implements a Retrieval-Augmented Generation (RAG) pipeline over a collection of arXiv papers.
It scrapes PDFs, extracts text, chunks content, embeds using SentenceTransformers, builds a FAISS vector index, and finally queries with OpenAI’s GPT models.

## Project workflow
1. Download recent cs.CL PDFs from arXiv.
2. Extract text from each PDF.
3. Chunk the text.
4. Create embeddings for each chunk.
5. Build a FAISS index over all chunk embeddings and save an ID→(pdf, chunk) mapping.
6. At query time: embed the user question, retrieve top-k nearest chunks from FAISS.
7. Concatenate the top-N chunks into context and ask the LLM to answer.
8. Return the answer along with the sources (PDF paths/chunk refs).

## 📁 Project folder structure

```
HW_4_RAG_w_ARXIV/
├─ master_setup.py
├─ query.py
├─ step1_scrape.py
├─ step2_extract.py
├─ step3_chunk.py
├─ step4_embed.py
├─ step5_faiss.py
├─ pdfs/ # downloaded PDFs (created at runtime)
├─ texts.json # extracted text per PDF (created)
├─ chunks.json # chunked text per PDF (created)
├─ documents.json # [{chunk, embedding}] per PDF (created)
├─ id_mapping.json # FAISS id → (pdf, chunk_idx) mapping (created)
├─ faiss.index # FAISS vector index (created)
└─ .env # your OpenAI API key (you create)
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

### Skip Scraping (reuse downloaded PDFs)
```
python3 master_setup.py --skip-scrape
```

### Skip Other Steps
```
python3 master_setup.py --skip-scrape --skip-extract --skip-chunk --skip-embed --skip-faiss
```

## ❓ Querying the RAG System
Once setup is complete, you can ask questions with query.py.

```
python3 query.py
```
Inside query.py, the default example is:
```
if __name__ == "__main__":
    query_rag("What are the latest methods in machine translation?")
```
To ask your own questions, modify the query string.

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
