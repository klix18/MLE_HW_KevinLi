# evaluate_retrieval.py
import os, json, sqlite3, re, collections
from typing import List, Dict, Set
from retrieval import FaissSearcher, bm25_search, hybrid_search, DB_PATH

GT_FILE = "ground_truth.json"

# -----------------------------
# Ground truth loading / seeding
# -----------------------------
def load_ground_truth() -> List[Dict]:
    """
    Expected format:
    [
      {"query": "...", "relevant_doc_paths": ["pdfs/1234.pdf", ...]},
      ...
    ]
    """
    if os.path.exists(GT_FILE):
        with open(GT_FILE, "r") as f:
            gt = json.load(f)
        return gt
    # Auto-seed 10 term queries from corpus (fallback for smoke tests)
    return auto_seed_ground_truth(n=10)

def auto_seed_ground_truth(n=10) -> List[Dict]:
    """
    Create synthetic queries from frequent nontrivial terms in the corpus.
    Relevance => any doc whose chunks contain the term.
    """
    conn = sqlite3.connect(DB_PATH)
    texts = [t for (t,) in conn.execute("SELECT text FROM chunks").fetchall()]
    conn.close()

    vocab = collections.Counter()
    token_pat = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")
    stop = {
        "the","and","for","that","with","from","this","these","those","into","onto",
        "your","have","has","had","are","was","were","their","there","where",
        "been","also","between","using","used","such"
    }
    for t in texts:
        for w in token_pat.findall(t):
            w_low = w.lower()
            if w_low in stop:
                continue
            vocab[w_low] += 1

    candidates = [w for (w, _c) in vocab.most_common(200)]
    queries = candidates[:n] if len(candidates) >= n else candidates

    gt = []
    conn = sqlite3.connect(DB_PATH)
    for q in queries:
        rows = conn.execute("""
        SELECT DISTINCT d.path
        FROM doc_chunks_fts f
        JOIN documents d ON d.doc_id = f.doc_id
        WHERE f.text LIKE ? LIMIT 50
        """, (f"%{q}%",)).fetchall()
        rel = [r[0] for r in rows]
        if rel:
            gt.append({"query": q, "relevant_doc_paths": rel})
        if len(gt) >= n:
            break
    conn.close()
    if not gt:
        raise RuntimeError("Could not auto-generate ground truth; please create ground_truth.json manually.")
    with open(GT_FILE, "w") as f:
        json.dump(gt, f, indent=2)
    return gt

# -----------------------------
# Metrics helpers (doc-level)
# -----------------------------
def dedupe_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def topk_doc_paths(method_results, k=3):
    """
    Normalize retrieval results to UNIQUE doc paths (strings), preserve order, then take top-k.
    method_results can be:
      - list[str] of doc paths, or
      - list[dict] where each dict has "doc_path".
    """
    if not method_results:
        return []
    if isinstance(method_results[0], dict):
        doc_paths = [r.get("doc_path") for r in method_results if r.get("doc_path")]
    else:
        doc_paths = list(method_results)
    return dedupe_keep_order(doc_paths)[:k]

def hit_at_k(method_results, relevant_set: Set[str], k=3) -> float:
    """Binary hit: 1 if any relevant doc appears in the top-k unique doc paths, else 0."""
    top = topk_doc_paths(method_results, k)
    return 1.0 if any(p in relevant_set for p in top) else 0.0

def found_at_k(method_results, relevant_set: Set[str], k=3) -> int:
    """Count of unique relevant docs found in the top-k unique doc paths."""
    top = topk_doc_paths(method_results, k)
    return sum(1 for p in top if p in relevant_set)

def recall_at_k_strict(method_results, relevant_set: Set[str], k=3) -> float:
    """
    Found@k divided by min(k, #relevant).
    Gives a value in [0,1], measuring how many of the possible relevant docs were found in top-k.
    """
    denom = max(1, min(k, len(relevant_set)))
    return found_at_k(method_results, relevant_set, k) / denom

# -----------------------------
# Main evaluation
# -----------------------------
def evaluate(k: int = 3):
    gt = load_ground_truth()
    fs = FaissSearcher()

    scores_sem, scores_kw, scores_hy = [], [], []

    for case in gt:
        q = case["query"]
        relevant = set(case["relevant_doc_paths"])

        # Vector (FAISS)
        sem = fs.search(q, k=10)  # [(doc_path, ord, dist), ...]
        sem_doc_order = [p for (p, _ord, _d) in sem]

        # Keyword (FTS5/BM25)
        kw = bm25_search(q, k=10)  # [(doc_path, chunk_id, bm25, text), ...]
        kw_doc_order = [p for (p, _chunk, _s, _txt) in kw]

        # Hybrid (fusion)
        hy = hybrid_search(q, k=10)  # [{"doc_path": ..., ...}, ...]
        hy_doc_order = hy  # normalized later by helpers

        # Compute metrics
        hit_sem = hit_at_k(sem_doc_order, relevant, k=k)
        hit_kw  = hit_at_k(kw_doc_order,  relevant, k=k)
        hit_hy  = hit_at_k(hy_doc_order,  relevant, k=k)

        scores_sem.append(hit_sem)
        scores_kw.append(hit_kw)
        scores_hy.append(hit_hy)

        c_sem = found_at_k(sem_doc_order, relevant, k=k)
        c_kw  = found_at_k(kw_doc_order,  relevant, k=k)
        c_hy  = found_at_k(hy_doc_order,  relevant, k=k)

        r_sem = recall_at_k_strict(sem_doc_order, relevant, k=k)
        r_kw  = recall_at_k_strict(kw_doc_order,  relevant, k=k)
        r_hy  = recall_at_k_strict(hy_doc_order,  relevant, k=k)

        # Per-query report
        print(f"Q: {q}")
        print(f"  Relevant docs: {len(relevant)}")
        print(f"  Vector@{k} hit:  {hit_sem:.0f} | found={c_sem}/{len(relevant)} | recall={r_sem:.2f}")
        print(f"  Keyword@{k} hit: {hit_kw:.0f} | found={c_kw}/{len(relevant)} | recall={r_kw:.2f}")
        print(f"  Hybrid@{k} hit:  {hit_hy:.0f} | found={c_hy}/{len(relevant)} | recall={r_hy:.2f}")
        print("-" * 40)

    def avg(xs): return sum(xs) / len(xs) if xs else 0.0
    print("=== Summary (Hit@{0}) ===".format(k))
    print(f"Vector-only   : {avg(scores_sem):.3f}")
    print(f"Keyword-only  : {avg(scores_kw):.3f}")
    print(f"Hybrid (late) : {avg(scores_hy):.3f}")

if __name__ == "__main__":
    evaluate(k=3)
