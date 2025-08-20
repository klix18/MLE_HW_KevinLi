# RAG Pipeline for arXiv cs.CL Papers
### This project implements a Retrieval-Augmented Generation (RAG) pipeline over a collection of arXiv papers.
It scrapes PDFs, extracts text, chunks content, embeds using SentenceTransformers, builds a FAISS vector index, and finally queries with OpenAI’s GPT models.

## 📁 Project structure (key files)

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

