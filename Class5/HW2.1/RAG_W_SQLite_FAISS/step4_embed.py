from sentence_transformers import SentenceTransformer
import json

def embed_chunks(chunks: list, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    results = []
    for chunk, emb in zip(chunks, embeddings):
        results.append({"chunk": chunk, "embedding": emb.tolist()})
    return results

# ----------------------------
# Higher-level pipeline
# ----------------------------
def build_embeddings(input_json="chunks.json", output_json="documents.json", model_name="all-MiniLM-L6-v2"):
    """
    Load chunked documents from JSON, embed them, and save into a new JSON.
    """
    model = SentenceTransformer(model_name)

    with open(input_json, "r") as f:
        data = json.load(f)

    documents_with_embeddings = {}
    for pdf_path, chunks in data.items():
        embeddings = model.encode(chunks, show_progress_bar=True).tolist()
        documents_with_embeddings[pdf_path] = [
            {"chunk": chunk, "embedding": emb}
            for chunk, emb in zip(chunks, embeddings)
        ]

    with open(output_json, "w") as f:
        json.dump(documents_with_embeddings, f, indent=2)

    print(f"Embeddings saved to {output_json}")


if __name__ == "__main__":
    build_embeddings()
