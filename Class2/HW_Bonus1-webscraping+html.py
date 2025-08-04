import requests
import json
import trafilatura
from datetime import datetime

CATEGORY = 'cs.AI'
MAX_RESULTS = 200
OUTPUT_JSON = '/Users/kevinli_home/Desktop/MLE_in_Gen_AI-Course/class2/03_Demo/HW_Bonus1/arxiv_clean.json'

# 1. Query arXiv API for latest papers in cs.AI
base_url = 'http://export.arxiv.org/api/query'
params = {
    'search_query': f'cat:{CATEGORY}',
    'start': 0,
    'max_results': MAX_RESULTS,
    'sortBy': 'submittedDate',
    'sortOrder': 'descending'
}
response = requests.get(base_url, params=params)
response.raise_for_status()

import xml.etree.ElementTree as ET
root = ET.fromstring(response.text)

data = []

# 2. Loop through each paper entry
for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
    url = entry.find('{http://www.w3.org/2005/Atom}id').text
    title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
    published = entry.find('{http://www.w3.org/2005/Atom}published').text
    date = datetime.fromisoformat(published).strftime('%Y-%m-%d')
    authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
    
    abs_url = url.replace('pdf', 'abs')
    html = requests.get(abs_url).text

    # 3. Use Trafilatura to extract and clean abstract
    abstract = trafilatura.extract(html, include_comments=False, include_tables=False)
    
    # Sometimes Trafilatura may return None, fallback to empty string
    if abstract is None:
        abstract = ""

    data.append({
        'url': abs_url,
        'title': title,
        'abstract': abstract.strip(),
        'authors': authors,
        'date': date
    })

# 4. Save to JSON file
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(data)} papers to {OUTPUT_JSON}")
