# Week 2

- Performed OCR on images using **Tesseract**
  - **Tesseract** – open‑source OCR engine for extracting English and Chinese text
    
- Scraped latest **arXiv** cs.AI papers and extracted abstracts with **Trafilatura**
  - **Trafilatura** – library for converting and cleaning HTML into plain text/JSON
- Converted arXiv **PDFs** to text via OCR
  - **pdf2image** – converts PDFs to high‑resolution images for OCR
  - **pytesseract** – Python wrapper for Tesseract
- Transcribed **YouTube audio** into JSONL transcripts using **Whisper**
  - **Whisper** – OpenAI speech recognition model
  - **yt-dlp** – utility for downloading YouTube audio
- Cleaned and deduplicated the collected texts
  - **langdetect** – filtered non‑English documents
  - **datasketch** – MinHash/LSH deduplication; removed PII, HTML, and repetitive n‑grams
