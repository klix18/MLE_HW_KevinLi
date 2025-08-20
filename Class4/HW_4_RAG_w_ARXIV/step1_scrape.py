import arxiv
import os
import time
import requests
from tqdm import tqdm

SAVE_DIR = "pdfs"
os.makedirs(SAVE_DIR, exist_ok=True)

def safe_download(url, filename, retries=3, delay=5):
    """
    Download a file from `url` to `filename` with retries.
    """
    for i in range(retries):
        try:
            r = requests.get(url, stream=True, timeout=30)
            if r.status_code == 200:
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
                return True
            else:
                print(f"Failed {url} with status {r.status_code}, retry {i+1}/{retries}")
        except Exception as e:
            print(f"Error downloading {url}: {e}, retry {i+1}/{retries}")
        time.sleep(delay)
    return False

def scrape_pdfs(max_results=50, category="cs.CL"):
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    pdf_paths = []
    for result in tqdm(search.results(), desc="Downloading PDFs"):
        pdf_path = os.path.join(SAVE_DIR, f"{result.get_short_id()}.pdf")
        if not os.path.exists(pdf_path):  # skip if already downloaded
            ok = safe_download(result.pdf_url, pdf_path)
            if not ok:
                print(f"❌ Skipping {result.get_short_id()} due to repeated failures")
                continue
        pdf_paths.append(pdf_path)
        time.sleep(1)  # throttle to avoid hammering arXiv
    return pdf_paths

if __name__ == "__main__":
    scrape_pdfs()
