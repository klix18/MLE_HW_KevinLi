import faiss
import numpy as np
import json

INDEX_FILE = "faiss.index"
DATA_JSON = "documents.json"

# ----------------------------
# Build FAISS index
# ----------------------------
def build_faiss_index(dim=384, input_json=DATA_JSON, index_file=INDEX_FILE):
    """
    Build and save a FAISS index from embedded chunks stored in documents.json.
    """
    with open(input_json, "r") as f:
        documents = json.load(f)

    # Flatten all embeddings
    embeddings = []
    id_mapping = []
    for doc_id, chunks in documents.items():
        for i, item in enumerate(chunks):
            embeddings.append(item["embedding"])
            id_mapping.append((doc_id, i))

    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, index_file)

    # Save mapping
    with open("id_mapping.json", "w") as f:
        json.dump(id_mapping, f, indent=2)

    print(f"FAISS index saved to {index_file}")
    return index


# ----------------------------
# Load FAISS index
# ----------------------------
def load_faiss_index(index_file=INDEX_FILE):
    return faiss.read_index(index_file)


# ----------------------------
# Save explicitly if needed
# ----------------------------
def save_faiss_index(index, index_file=INDEX_FILE):
    faiss.write_index(index, index_file)
    print(f"FAISS index saved to {index_file}")

