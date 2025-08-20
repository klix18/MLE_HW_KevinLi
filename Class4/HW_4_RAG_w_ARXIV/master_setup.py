import os
import json
from step1_scrape import scrape_pdfs
from step2_extract import extract_text_from_pdf
from step3_chunk import chunk_text
from step4_embed import build_embeddings
from step5_faiss import build_faiss_index
import glob


    pdfs = scrape_pdfs()
# File names
TEXTS_JSON = "texts.json"
CHUNKS_JSON = "chunks.json"
DOCS_JSON = "documents.json"

def setup_rag(skip_scrape=False, skip_extract=False, skip_chunk=False, skip_embed=False, skip_faiss=False):
    # ----------------
    # Step 1: Scrape
    # ----------------
    if skip_scrape:
        # Just grab whatever is already in the pdfs folder
        pdfs = glob.glob("pdfs/*.pdf")
        print(f"Skipping scrape. Found {len(pdfs)} PDFs in local folder.")
    else:

    # ----------------
    # Step 2: Extract
    # ----------------
    if not skip_extract or not os.path.exists(TEXTS_JSON):
        texts = {pdf: extract_text_from_pdf(pdf) for pdf in pdfs}
        with open(TEXTS_JSON, "w") as f:
            json.dump(texts, f, indent=2)
    else:
        with open(TEXTS_JSON, "r") as f:
            texts = json.load(f)

    # ----------------
    # Step 3: Chunk
    # ----------------
    if not skip_chunk or not os.path.exists(CHUNKS_JSON):
        chunks = {pdf: chunk_text(texts[pdf]) for pdf in pdfs}
        with open(CHUNKS_JSON, "w") as f:
            json.dump(chunks, f, indent=2)
    else:
        with open(CHUNKS_JSON, "r") as f:
            chunks = json.load(f)

    # ----------------
    # Step 4: Embed
    # ----------------
    if not skip_embed or not os.path.exists(DOCS_JSON):
        build_embeddings(CHUNKS_JSON, DOCS_JSON)
    else:
        print("Skipping embeddings, using cached documents.json")

    # ----------------
    # Step 5: FAISS
    # ----------------
    if not skip_faiss:
        build_faiss_index(input_json=DOCS_JSON)
    else:
        print("Skipping FAISS index build")

    print("✅ Setup complete!")

if __name__ == "__main__":
    setup_rag()
