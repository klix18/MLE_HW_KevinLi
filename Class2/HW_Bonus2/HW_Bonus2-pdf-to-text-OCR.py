import os
import json
import requests  # <-- ADD THIS LINE
import arxiv
import pytesseract
from pdf2image import convert_from_path
from datetime import datetime
from tqdm import tqdm


# Settings
CATEGORY = 'cs.AI'
MAX_RESULTS = 200
OUTPUT_DIR = '/Users/kevinli_home/Desktop/MLE_in_Gen_AI-Course/class2/03_Demo/HW_Bonus2/pdfs'
OUTPUT_JSON = '/Users/kevinli_home/Desktop/MLE_in_Gen_AI-Course/class2/03_Demo/HW_Bonus2/arxiv_pdf_ocr.json'

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    
    pdf_path = os.path.join(OUTPUT_DIR, f'{arxiv_id}.pdf')
    
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

    # OCR on each page
    ocr_text = ""
    for img in images:
        ocr_text += pytesseract.image_to_string(img, lang='eng', config='--psm 3')  # psm 3 = fully automatic layout

    data.append({
        'url': result.entry_id,
        'title': title,
        'abstract_ocr': ocr_text.strip(),
        'authors': authors,
        'date': date
    })

# Save as JSON
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"OCR completed for {len(data)} papers. Results saved to: {OUTPUT_JSON}")
