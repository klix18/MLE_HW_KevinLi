# retrieval.py
import json
import numpy as np
import faiss
import sqlite3
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer

DB_PATH = "rag.db"
INDEX_FILE = "faiss.index"
ID_MAP_JSON = "id_mapping.json"
DOCS_JSON = "documents.json"

# ------------------------------
# FAISS helpers
# ------------------------------
class FaissSearcher:
    def __init__(self, index_file=INDEX_FILE, id_map_json=ID_MAP_JSON, model_name="all-MiniLM-L6-v2"):
        self.index = faiss.read_index(index_file)
        with open(id_map_json, "r") as f:
            # [(doc_path, chunk_ord), ...] in FAISS row order
            self.id_map: List[Tuple[str, int]] = json.load(f)
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, int, float]]:
        q_emb = self.model.encode([query]).astype("float32")
        D, I = self.index.search(q_emb, k)
        # Return [(doc_path, chunk_ord, l2_distance)]
        out = []
        for idx, dist in zip(I[0], D[0]):
            doc_path, chunk_ord = self.id_map[idx]
            out.append((doc_path, chunk_ord, float(dist)))
        return out

# ------------------------------
# SQLite/BM25 helpers
# ------------------------------
def bm25_search(query: str, k: int = 10, db_path: str = DB_PATH):
    import re, sqlite3

    def to_fts5_query(q: str) -> str:
        # keep alnum tokens, lowercase, drop very short terms
        toks = re.findall(r"[A-Za-z0-9]+", q.lower())
        toks = [t for t in toks if len(t) >= 2]
        if not toks:
            return ""  # will return no rows
        # Quote each token to avoid operator parsing; AND semantics via spaces
        return " ".join(f'"{t}"' for t in toks)

    fts_q = to_fts5_query(query)
    if not fts_q:
        return []

    conn = sqlite3.connect(db_path)

    q = """
    SELECT f.doc_id,
           f.chunk_id,
           bm25(doc_chunks_fts) AS score,   -- must pass real table name
           f.text,
           d.path
    FROM doc_chunks_fts AS f
    JOIN documents d ON d.doc_id = f.doc_id
    WHERE doc_chunks_fts MATCH ?
    ORDER BY score ASC
    LIMIT ?;
    """

    try:
        rows = conn.execute(q, (fts_q, k)).fetchall()
    except sqlite3.OperationalError as e:
        # Fallback if bm25() not available in your SQLite build
        if "no such function: bm25" in str(e):
            q_fallback = """
            SELECT f.doc_id,
                   f.chunk_id,
                   0.0 AS score,
                   f.text,
                   d.path
            FROM doc_chunks_fts AS f
            JOIN documents d ON d.doc_id = f.doc_id
            WHERE doc_chunks_fts MATCH ?
            LIMIT ?;
            """
            rows = conn.execute(q_fallback, (fts_q, k)).fetchall()
        else:
            conn.close()
            raise
    conn.close()
    # Return [(doc_path, chunk_id, bm25_score, text)]
    return [(r[4], r[1], float(r[2]), r[3]) for r in rows]


# ------------------------------
# Utilities to fetch chunk text by (doc_path, ord)
# ------------------------------
def _doc_id_for_path(conn: sqlite3.Connection, path: str) -> int:
    r = conn.execute("SELECT doc_id FROM documents WHERE path = ?", (path,)).fetchone()
    return r[0] if r else None

def chunk_text_by_path_ord(path: str, ord_idx: int, db_path: str = DB_PATH) -> Tuple[int, int, str]:
    conn = sqlite3.connect(db_path)
    doc_id = _doc_id_for_path(conn, path)
    row = conn.execute(
        "SELECT chunk_id, text FROM chunks WHERE doc_id = ? AND ord = ?",
        (doc_id, ord_idx)
    ).fetchone()
    conn.close()
    if not row:
        return None, None, ""
    return doc_id, row[0], row[1]

# ------------------------------
# Merging: normalize + weighted sum OR RRF
# ------------------------------
def minmax_norm(vals: List[float]) -> List[float]:
    if not vals:
        return vals
    vmin, vmax = min(vals), max(vals)
    if vmax - vmin < 1e-12:
        return [0.5] * len(vals)
    return [(v - vmin) / (vmax - vmin) for v in vals]

def reciprocal_rank_fusion(rank_lists: Dict[str, List[Tuple[str,int,float,str]]], k: int = 60, c: int = 60):
    """
    rank_lists: {"semantic": [(doc_path, chunk_id/ord, score, text?)...], "keyword": [...]}.
    We treat second element as either chunk_id or ord; it won't affect dedupe key.
    """
    # Build unique keys by (doc_path, ord OR chunk_id). Prefer (doc_path, ord) if available.
    ranks: Dict[Tuple[str,int], float] = {}
    for _name, results in rank_lists.items():
        for rank, item in enumerate(results, start=1):
            doc_path, second, _score, _text = item
            key = (doc_path, second)
            ranks[key] = ranks.get(key, 0.0) + 1.0 / (c + rank)
    # Sort descending
    merged = sorted(ranks.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return merged  # [((doc_path, second), fused_score), ...]

def hybrid_search(query: str, k: int = 10, weight_semantic: float = 0.5):
    """
    Returns final ranked list of dicts:
    [{"doc_path", "chunk_id", "ord", "text", "score_sem", "score_kw", "score_hybrid"}]
    """
    # Semantic
    fs = FaissSearcher()
    sem_raw = fs.search(query, k=k*2)  # overfetch for better merge
    sem_paths, sem_ords, sem_dists = zip(*sem_raw) if sem_raw else ([], [], [])
    # Convert L2 distance to similarity (lower dist -> higher sim)
    if sem_dists:
        sem_sims = [1.0 - x for x in minmax_norm(list(sem_dists))]
    else:
        sem_sims = []

    # Convert to unified tuples: (doc_path, ord, sem_similarity)
    sem_tuples = []
    for (p, o, _), s in zip(sem_raw, sem_sims):
        # fetch chunk_id and text from DB
        _docid, chunk_id, text = chunk_text_by_path_ord(p, o)
        sem_tuples.append((p, o, s, text, chunk_id))

    # Keyword/BM25
    kw_raw = bm25_search(query, k=k*2)
    # bm25: smaller is better, invert then normalize to [0,1]
    kw_scores = [1.0 / (r[2] + 1e-9) for r in kw_raw] if kw_raw else []
    kw_sims = minmax_norm(kw_scores) if kw_scores else []
    # (doc_path, chunk_id, kw_sim, text)
    kw_tuples = []
    for (p, chunk_id, _bm, text), s in zip(kw_raw, kw_sims):
        # need ord for key, fetch by chunk_id
        kw_tuples.append((p, chunk_id, s, text))

    # Build dicts keyed by (doc_path, ord/ chunk_id)
    sem_by_key: Dict[Tuple[str,int], Dict] = {}
    for p, ord_idx, s, text, chunk_id in sem_tuples:
        sem_by_key[(p, ord_idx)] = {
            "doc_path": p, "ord": ord_idx, "chunk_id": chunk_id,
            "text": text, "score_sem": s, "score_kw": 0.0
        }

    # Attach keyword scores by chunk_id; try to recover ord via DB if needed
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    for p, chunk_id, s, text in kw_tuples:
        row = conn.execute("SELECT ord FROM chunks c JOIN documents d ON d.doc_id=c.doc_id WHERE d.path=? AND c.chunk_id=?",(p, chunk_id)).fetchone()
        if not row:
            continue
        ord_idx = int(row[0])
        key = (p, ord_idx)
        if key in sem_by_key:
            sem_by_key[key]["score_kw"] = max(sem_by_key[key]["score_kw"], s)
            # Prefer richer text if missing
            if not sem_by_key[key]["text"]:
                sem_by_key[key]["text"] = text
        else:
            sem_by_key[key] = {
                "doc_path": p, "ord": ord_idx, "chunk_id": chunk_id,
                "text": text, "score_sem": 0.0, "score_kw": s
            }
    conn.close()

    # Weighted late fusion
    out = []
    for v in sem_by_key.values():
        v["score_hybrid"] = weight_semantic * v["score_sem"] + (1.0 - weight_semantic) * v["score_kw"]
        out.append(v)

    out.sort(key=lambda x: x["score_hybrid"], reverse=True)
    return out[:k]
