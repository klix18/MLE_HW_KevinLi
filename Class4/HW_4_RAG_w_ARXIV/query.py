import numpy as np
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from step5_faiss import load_faiss_index

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

DATA_JSON = "documents.json"
MAPPING_JSON = "id_mapping.json"

def query_rag(question: str, k: int = 5):
    # Load index + documents + mapping
    index = load_faiss_index()
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        documents = json.load(f)
    with open(MAPPING_JSON, "r", encoding="utf-8") as f:
        id_mapping = json.load(f)

    # Embed query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_embedding, k)

    all_chunks = []
    for i, (d, idx) in enumerate(zip(distances[0], indices[0])):
        doc_id, chunk_idx = id_mapping[idx]  # map back to (pdf, chunk number)
        chunk_entry = documents[doc_id][chunk_idx]  # retrieve chunk
        all_chunks.append({
            "rank": i+1,
            "distance": float(d),
            "chunk": chunk_entry["chunk"],
            "source": doc_id
        })

    # Top-3 context
    context = "\n".join([c["chunk"] for c in all_chunks[:3]])

    # LLM response
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer."},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
        ]
    )
    answer = response.choices[0].message.content

    # Final format
    print("AI Response:\n", answer)
    print("\nTop 3 chunks used:\n")
    for c in all_chunks[:3]:
        print(f"[Rank {c['rank']}] (from {c['source']}) {c['chunk'][:200]}...\n")
    print("\nTop 5 chunks (distance + snippet):\n")
    for c in all_chunks:
        print(f"[Rank {c['rank']}] Score: {c['distance']:.4f} | (from {c['source']}) {c['chunk'][:100]}...")


if __name__ == "__main__":
    query_rag("What are the latest methods in machine translation?")
