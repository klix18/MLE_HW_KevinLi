import os
import json
import requests
import arxiv
import pytesseract
from pdf2image import convert_from_path
from datetime import datetime
from tqdm import tqdm

# Settings
CATEGORY = 'cs.AI'
MAX_RESULTS = 20
PDF_DIR = '/Users/kevinli_home/Desktop/MLE_in_Gen_AI-Course/class2/03_Demo/HW_Bonus2/pdfs'
TXT_DIR = '/Users/kevinli_home/Desktop/MLE_in_Gen_AI-Course/class2/03_Demo/HW_Bonus2/pdf_ocr'
OUTPUT_JSON = '/Users/kevinli_home/Desktop/MLE_in_Gen_AI-Course/class2/03_Demo/HW_Bonus2/arxiv_pdf_ocr.json'

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)

# Search arXiv
search = arxiv.Search(
    query=f"cat:{CATEGORY}",
    max_results=MAX_RESULTS,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

data = []

for result in tqdm(list(search.results()), desc="Processing papers"):
    pdf_url = result.pdf_url
    title = result.title.strip()
    date = result.published.strftime('%Y-%m-%d')
    authors = [a.name for a in result.authors]
    arxiv_id = result.get_short_id()

    pdf_path = os.path.join(PDF_DIR, f'{arxiv_id}.pdf')
    txt_path = os.path.join(TXT_DIR, f'{arxiv_id}.txt')

    # Download PDF
    try:
        if not os.path.exists(pdf_path):
            response = requests.get(pdf_url, timeout=20)
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
    except Exception as e:
        print(f"Failed to download {pdf_url}: {e}")
        continue

    # Convert PDF to images
    try:
        images = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        print(f"Failed to convert PDF {arxiv_id}: {e}")
        continue

    # OCR each page
    ocr_text = ""
    for img in images:
        ocr_text += pytesseract.image_to_string(img, lang='eng', config='--psm 3') + "\n\n"

    # Save individual .txt
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(ocr_text.strip())

    data.append({
        'url': result.entry_id,
        'title': title,
        'authors': authors,
        'date': date,
        'txt_file': txt_path
    })

# Save metadata JSON
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"OCR completed for {len(data)} papers.")
print(f"Text files saved to: {TXT_DIR}")
print(f"Metadata JSON saved to: {OUTPUT_JSON}")
