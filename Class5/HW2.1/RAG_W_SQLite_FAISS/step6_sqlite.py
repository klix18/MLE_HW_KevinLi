# step6_sqlite.py
import os
import json
import sqlite3
from typing import Dict, List, Tuple

DB_PATH = "rag.db"
DOCS_JSON = "documents.json"      # produced by step4_embed.py
ID_MAP_JSON = "id_mapping.json"   # produced by step5_faiss.py

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA temp_store = MEMORY;

CREATE TABLE IF NOT EXISTS documents (
  doc_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  path       TEXT UNIQUE,        -- pdf path as unique handle
  title      TEXT,
  author     TEXT,
  year       INTEGER,
  keywords   TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id     INTEGER NOT NULL,
  ord        INTEGER NOT NULL,   -- position within doc
  text       TEXT NOT NULL,s
  FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

-- Contentless FTS5 table for keyword/BM25 search.
-- Keep TINY row: just the text and a couple of unindexed columns we join on.
CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_fts USING fts5(
  text,
  doc_id UNINDEXED,
  chunk_id UNINDEXED,
  tokenize = 'unicode61'
);
"""

def _ensure_db(conn: sqlite3.Connection):
    conn.executescript(SCHEMA)
    conn.commit()

def _get_or_create_doc_id(conn: sqlite3.Connection, path: str, title=None, author=None, year=None, keywords=None) -> int:
    cur = conn.execute("SELECT doc_id FROM documents WHERE path = ?", (path,))
    row = cur.fetchone()
    if row:
        return row[0]
    conn.execute(
        "INSERT INTO documents (path, title, author, year, keywords) VALUES (?,?,?,?,?)",
        (path, title, author, year, keywords)
    )
    conn.commit()
    return conn.execute("SELECT doc_id FROM documents WHERE path = ?", (path,)).fetchone()[0]

def load_into_sqlite(
    db_path: str = DB_PATH,
    docs_json: str = DOCS_JSON
):
    """
    Load chunks & (basic) metadata into SQLite + FTS5.
    Uses file path as the unique document key; you can enrich metadata later.
    """
    with open(docs_json, "r") as f:
        docs: Dict[str, List[Dict]] = json.load(f)  # {pdf_path: [{chunk, embedding}, ...]}

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    _ensure_db(conn)

    # Speed up bulk inserts
    conn.execute("BEGIN IMMEDIATE")
    try:
        for path, items in docs.items():
            doc_id = _get_or_create_doc_id(conn, path=path)
            # Insert chunks & mirror into FTS
            for i, item in enumerate(items):
                text = item["chunk"]
                cur = conn.execute(
                    "INSERT INTO chunks (doc_id, ord, text) VALUES (?, ?, ?)",
                    (doc_id, i, text)
                )
                chunk_id = cur.lastrowid
                conn.execute(
                    "INSERT INTO doc_chunks_fts (text, doc_id, chunk_id) VALUES (?, ?, ?)",
                    (text, doc_id, chunk_id)
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def keyword_search(
    query: str,
    k: int = 10,
    db_path: str = DB_PATH
) -> List[Tuple[int, int, float, str]]:
    """
    FTS5 keyword/BM25 search over chunk text.
    Returns list of (doc_id, chunk_id, bm25_score, text).
    Lower bm25(...) is more relevant; we invert it later for merging.
    """
    conn = sqlite3.connect(db_path)
    # Register bm25 rank function for ORDER BY if needed (SQLite exposes bm25() natively for FTS5)
    q = """
    SELECT doc_id, chunk_id, bm25(doc_chunks_fts) AS score, text
    FROM doc_chunks_fts
    WHERE doc_chunks_fts MATCH ?
    ORDER BY score ASC
    LIMIT ?;
    """
    rows = conn.execute(q, (query, k)).fetchall()
    conn.close()
    return rows
