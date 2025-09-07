# query.py
import os
import json
import argparse
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
import os
from retrieval import (
    FaissSearcher,
    bm25_search,
    hybrid_search,
    chunk_text_by_path_ord,
    DB_PATH
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------
# OpenAI setup
# ---------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment.")
client = OpenAI(api_key=api_key)

# ---------------------------
# Utilities
# ---------------------------
def build_context(chunks: List[Dict], n: int = 3) -> str:
    """Join top-n chunk texts for the LLM prompt."""
    return "\n\n".join([c["text"] for c in chunks[:n] if c.get("text")])

def print_results(title: str, rows: List[Dict], score_key: str, max_rows: int = 5):
    print(f"\n{title}")
    print("-" * len(title))
    for i, r in enumerate(rows[:max_rows], start=1):
        score = r.get(score_key)
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        src = r.get("doc_path") or r.get("source") or "?"
        snippet = (r.get("text") or "")[:200].replace("\n", " ")
        print(f"[{i}] score={score_str} | {src} | {snippet}...")

def rows_from_semantic(fs: FaissSearcher, query: str, k: int) -> List[Dict]:
    """Vector-only search, materialize text via SQLite by (doc_path, ord)."""
    raw = fs.search(query, k=k)
    rows = []
    for (path, ord_idx, l2) in raw:
        _docid, chunk_id, text = chunk_text_by_path_ord(path, ord_idx)
        rows.append({
            "doc_path": path,
            "ord": ord_idx,
            "chunk_id": chunk_id,
            "text": text,
            "score_sem": -(l2)  # invert so higher is better for display
        })
    # sort by descending score_sem for printing (not required; already by FAISS)
    rows.sort(key=lambda x: x["score_sem"], reverse=True)
    return rows

def rows_from_keyword(query: str, k: int) -> List[Dict]:
    """Keyword-only search via FTS5/BM25 (lower bm25 => better). Convert to similarity-like for display."""
    raw = bm25_search(query, k=k)  # [(doc_path, chunk_id, bm25_score, text)]
    rows = []
    for (path, chunk_id, bm25_score, text) in raw:
        # similarity-like for display (higher better)
        sim = 1.0 / (bm25_score + 1e-9)
        rows.append({
            "doc_path": path,
            "chunk_id": chunk_id,
            "text": text,
            "score_kw": sim
        })
    rows.sort(key=lambda x: x["score_kw"], reverse=True)
    return rows

def rows_from_hybrid(query: str, k: int, weight_sem: float = 0.5) -> List[Dict]:
    """Hybrid (late fusion) from retrieval.hybrid_search."""
    return hybrid_search(query, k=k, weight_semantic=weight_sem)

def call_llm(question: str, context: str, model: str = "gpt-4o-mini") -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer concisely and cite facts to the extent the context allows."},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
        ]
    )
    return resp.choices[0].message.content

# ---------------------------
# Main query entrypoint
# ---------------------------
def query_rag(question: str, k: int = 5, mode: str = "hybrid", weight_sem: float = 0.5, llm_model: str = "gpt-4o-mini"):
    """
    mode: 'semantic' | 'keyword' | 'hybrid'
    """
    mode = mode.lower().strip()
    if mode not in {"semantic", "keyword", "hybrid"}:
        raise ValueError("mode must be one of: semantic | keyword | hybrid")

    # Retrieve
    if mode == "semantic":
        fs = FaissSearcher()
        rows = rows_from_semantic(fs, question, k)
        score_key = "score_sem"
    elif mode == "keyword":
        rows = rows_from_keyword(question, k)
        score_key = "score_kw"
    else:
        rows = rows_from_hybrid(question, k, weight_sem=weight_sem)
        score_key = "score_hybrid"

    if not rows:
        print("No results found.")
        return

    # Build context (top 3 chunk texts)
    context = build_context(rows, n=3)

    # LLM answer
    answer = call_llm(question, context, model=llm_model)

    # Output
    print("\nAI Response:\n")
    print(answer)

    # Diagnostics
    print_results(f"Top {min(3, len(rows))} chunks used in context ({mode})", rows, score_key, max_rows=3)
    print_results(f"Top {min(k, len(rows))} retrieved chunks ({mode})", rows, score_key, max_rows=k)

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query RAG with semantic / keyword / hybrid retrieval.")
    parser.add_argument("question", type=str, help="Your question")
    parser.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["semantic", "keyword", "hybrid"], help="Retrieval mode")
    parser.add_argument("--weight_sem", type=float, default=0.5, help="Hybrid weight for semantic vs keyword (0..1)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI chat model")
    args = parser.parse_args()

    query_rag(
        question=args.question,
        k=args.k,
        mode=args.mode,
        weight_sem=args.weight_sem,
        llm_model=args.model
    )
