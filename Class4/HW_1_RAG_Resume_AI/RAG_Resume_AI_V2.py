# RAG_Resume_AI.py

import os
import re
import unicodedata
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY from .env if present

# ---- Optional LLM cache (speeds up repeated prompts) ----
try:
    from langchain.globals import set_llm_cache
    from langchain_community.cache import SQLiteCache
    set_llm_cache(SQLiteCache("lc_cache.db"))
except Exception:
    pass  # cache is optional

# ---- Cleaning & Dedup ----
from bs4 import BeautifulSoup
from datasketch import MinHash, MinHashLSH

# ---- LangChain bits ----
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

# Hybrid retrieval (BM25 + dense)
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# =========================
# Config
# =========================
DATA_DIR = "Documents"                 # change to absolute path if you prefer
PERSIST_DIR = "resume_idx"             # FAISS index folder
REBUILD_INDEX = bool(int(os.getenv("REBUILD_INDEX", "0")))  # set 1 to force rebuild

PERSON_NAME = "Kevin Li"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50                     # smaller to avoid near-duplicate top-k
DOC_DEDUP_THRESHOLD = 0.90             # Jaccard (via MinHash) for near-duplicate docs
MIN_DOC_LEN = 1                        # keep short docs (fixes multi-part Qs)
TOP_K = 10                             # larger k to cover multi-doc answers
FETCH_K = 60
MMR_LAMBDA = 0.3
DEBUG = True

# =========================
# Loading
# =========================
def load_dir(path: str):
    all_docs = []
    for p in Path(path).rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        try:
            if ext == ".pdf":
                all_docs += PyPDFLoader(str(p)).load()
            elif ext in {".txt", ".md"}:
                all_docs += TextLoader(str(p), encoding="utf-8").load()
            elif ext == ".docx":
                all_docs += Docx2txtLoader(str(p)).load()
        except Exception as e:
            print(f"[WARN] Skipping {p} due to loader error: {e}")
    return all_docs

# =========================
# Cleaning
# =========================
def clean_text(raw: str) -> str:
    text = BeautifulSoup(raw or "", "html.parser").get_text(separator=" ")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_documents(docs):
    for d in docs:
        d.page_content = clean_text(d.page_content)
    return docs

# =========================
# Dedup (MinHash/LSH)
# =========================
def _minhash_for_text(text: str, num_perm=128) -> MinHash:
    tokens = set(re.findall(r"\w+", text.lower()))
    m = MinHash(num_perm=num_perm)
    for t in tokens:
        m.update(t.encode("utf-8"))
    return m

def deduplicate_documents(docs, threshold=DOC_DEDUP_THRESHOLD, num_perm=128):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique = []
    for i, d in enumerate(docs):
        body = d.page_content or ""
        if len(body) < MIN_DOC_LEN:
            continue  # drop tiny/noisy docs/pages only if literally empty
        m = _minhash_for_text(body, num_perm=num_perm)
        if lsh.query(m):
            continue  # near-duplicate exists
        lsh.insert(f"doc-{i}", m)
        unique.append(d)
    return unique

# =========================
# Build Retriever (Hybrid + MMR + MultiQuery)
# =========================
def build_retriever(vectorstore, chunks):
    # Dense retriever with MMR for diversity
    dense = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": MMR_LAMBDA},
    )
    # BM25 lexical retriever over chunks
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K
    # Hybrid (lexical + dense)
    hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.45, 0.55])
    # Multi-Query expansion (paraphrase the question to improve recall)
    llm_q = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    mq = MultiQueryRetriever.from_llm(
        llm=llm_q,
        retriever=hybrid,
        include_original=True,
    )
    return mq

# =========================
# Debug helpers
# =========================
def debug_dense(vectorstore, query: str, k: int = 6):
    print("\n[DEBUG] Dense-only hits (vectorstore.similarity_search_with_score):")
    hits = vectorstore.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(hits, 1):
        src = doc.metadata.get("source")
        page = doc.metadata.get("page")
        snippet = (doc.page_content or "")[:220]
        print(f"\nHit {i} | score={score:.4f} | source={src} | page={page}\n{snippet}")

def debug_retriever(retriever, query: str):
    print("\n[DEBUG] Retriever hits (after MultiQuery + Hybrid + MMR):")
    docs = retriever.invoke(query)
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source")
        page = d.metadata.get("page")
        snippet = (d.page_content or "")[:220]
        print(f"\n{i}. source={src} | page={page}\n{snippet}")

# =========================
# Main
# =========================
def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY (set in environment or .env)")

    # 1) Load
    docs = load_dir(DATA_DIR)

    # 2) Add a small header so every chunk carries identity/provenance
    for d in docs:
        header = f"Person: {PERSON_NAME}\nSource: {d.metadata.get('source','unknown')}\n"
        d.page_content = header + (d.page_content or "")

    # 3) Clean + dedup BEFORE chunking
    docs = clean_documents(docs)
    docs = deduplicate_documents(docs, threshold=DOC_DEDUP_THRESHOLD)

    # 4) Chunk (smaller chunks help isolate atomic facts like employer names)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # prefer sentence/line breaks
    )
    chunks = splitter.split_documents(docs)

    # 5) Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 6) FAISS: load if exists; else build & persist (prevents re-embedding every run)
    if Path(PERSIST_DIR).exists() and not REBUILD_INDEX:
        vectorstore = FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(PERSIST_DIR)

    # 7) Strong retriever (Hybrid + MMR + MultiQuery)
    retriever = build_retriever(vectorstore, chunks)

    # 8) Prompt with an explicit “don’t know” rule
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Answer using ONLY the context. If the answer is not present, say you don't know.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer concisely."
        ),
    )

    # 9) RetrievalQA
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # with larger k and hybrid, this now works well; swap to "map_reduce" if you prefer
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    # 10) Example query (try multi-part)
    question = "what does kevin do, how old is he, and what is his pet's name?"
    result = agent.invoke({"query": question})
    print("\nANSWER:", result["result"])

    print("\nSOURCES:")
    for i, d in enumerate(result.get("source_documents", []), 1):
        print(f"- {i}. source={d.metadata.get('source')} page={d.metadata.get('page')}")

    if DEBUG:
        debug_dense(vectorstore, question, k=8)
        debug_retriever(retriever, question)

if __name__ == "__main__":
    main()
