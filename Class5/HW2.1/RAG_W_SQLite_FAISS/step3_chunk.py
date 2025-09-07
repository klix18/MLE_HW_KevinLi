from typing import List

def chunk_text(text: str, max_tokens: int = 450, overlap: int = 50) -> List[str]:
    words = text.split()
    step = max_tokens - overlap
    chunks = []

    for i in range(0, len(words), step):
        chunk = words[i:i + max_tokens]
        chunks.append(" ".join(chunk))

    return chunks

if __name__ == "__main__":
    text = "This is a long document example ..." * 100
    print(chunk_text(text)[:2])
