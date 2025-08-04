import os
import json
import re
from langdetect import detect
from datasketch import MinHash, MinHashLSH
from bs4 import BeautifulSoup
from collections import Counter

# 📁 Input directories
LOC1 = "/Users/kevinli_home/Desktop/MLE_in_Gen_AI-Course/class2/03_Demo/HW_Bonus1/arxiv_clean.json"
LOC2 = "/Users/kevinli_home/Desktop/MLE_in_Gen_AI-Course/class2/03_Demo/HW_Bonus2/pdf_ocr"
LOC3 = "/Users/kevinli_home/Desktop/MLE_in_Gen_AI-Course/class2/03_Demo/HW_Bonus3/transripts/v2-language-set-to-en"

# 📁 Output directory
OUTPUT_DIR = "/Users/kevinli_home/Desktop/MLE_in_Gen_AI-Course/class2/03_Demo/HW_Bonus4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 📊 Stats tracker
stats = {
    "total_docs_loaded": 0,
    "non_english_docs": 0,
    "duplicates_removed": 0,
    "pii_removed": 0,
    "html_stripped": 0,
    "ngram_reductions": 0
}

def is_english(text):
    try:
        return detect(text.strip()) == "en"
    except:
        return False

def clean_html(text):
    if "<" in text and ">" in text:
        soup = BeautifulSoup(text, "html.parser")
        stripped = soup.get_text(separator=" ", strip=True)
        stats["html_stripped"] += 1
        return stripped
    return text

def remove_pii(text):
    original = text
    text = re.sub(r'\b[\w\.-]+?@\w+?\.\w+?\b', '[EMAIL]', text)
    text = re.sub(r'\b(\+?\d{1,2}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[CREDITCARD]', text)
    if text != original:
        stats["pii_removed"] += 1
    return text

def remove_repetitive_ngrams(text, n=3):
    tokens = text.split()
    new_tokens = []
    i = 0
    while i < len(tokens):
        ngram = tokens[i:i+n]
        next_ngram = tokens[i+n:i+2*n]
        if ngram == next_ngram:
            stats["ngram_reductions"] += 1
            i += n
        else:
            new_tokens.append(tokens[i])
            i += 1
    return " ".join(new_tokens)

def get_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in set(text.split()):
        m.update(word.encode("utf8"))
    return m

def process_documents():
    source_docs = []
    source_counts = {"arxiv": 0, "ocr": 0, "whisper": 0}

    # -- Load from arxiv_clean.json
    if os.path.exists(LOC1):
        with open(LOC1, "r") as f:
            json_data = json.load(f)
            for entry in json_data:
                content = entry.get("content", "").strip()
                if content:
                    source_docs.append(("arxiv", content))
                    source_counts["arxiv"] += 1

    # -- Load from OCR and Whisper folders
    for folder, label in [(LOC2, "ocr"), (LOC3, "whisper")]:
        for file in os.listdir(folder):
            if file.endswith(".txt") or file.endswith(".jsonl"):
                path = os.path.join(folder, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        if text:
                            source_docs.append((label, text))
                            source_counts[label] += 1
                except Exception:
                    continue

    stats["total_docs_loaded"] = sum(source_counts.values())

    cleaned = []
    sources = []

    for source, doc in source_docs:
        if not is_english(doc):
            stats["non_english_docs"] += 1
            continue
        doc = clean_html(doc)
        doc = remove_pii(doc)
        doc = remove_repetitive_ngrams(doc)
        cleaned.append(doc)
        sources.append(source)

    # Deduplication with tracking
    lsh = MinHashLSH(threshold=0.7, num_perm=128)
    unique_texts = []
    unique_sources = []
    dropped_by_source = Counter()

    for i, text in enumerate(cleaned):
        m = get_minhash(text)
        if lsh.query(m):
            dropped_by_source[sources[i]] += 1
            stats["duplicates_removed"] += 1
            continue
        lsh.insert(f"doc{i}", m)
        unique_texts.append(text)
        unique_sources.append(sources[i])

    # Save per-source deduplication stats
    for src in ["arxiv", "ocr", "whisper"]:
        stats[f"{src}_retained"] = unique_sources.count(src)
        stats[f"{src}_deduplicated"] = dropped_by_source[src]

    return unique_texts

def write_outputs(cleaned_texts):
    # Save clean corpus
    corpus_path = os.path.join(OUTPUT_DIR, "clean_corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in cleaned_texts:
            f.write(doc.strip() + "\n\n---\n\n")

    # Save stats
    stats_path = os.path.join(OUTPUT_DIR, "stats.md")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("# Cleaning Statistics\n\n")
        for k, v in stats.items():
            label = k.replace("_", " ").capitalize()
            f.write(f"- **{label}**: {v}\n")

def main():
    cleaned = process_documents()
    write_outputs(cleaned)

if __name__ == "__main__":
    main()

