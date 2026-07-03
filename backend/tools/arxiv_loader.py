import requests
import os

DOWNLOAD_DIR = "papers"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def load_arxiv_paper(arxiv_url):

    paper_id = arxiv_url.rstrip("/").split("/")[-1]

    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

    response = requests.get(pdf_url, timeout=60)
    response.raise_for_status()

    path = os.path.join(DOWNLOAD_DIR, f"{paper_id}.pdf")

    with open(path, "wb") as f:
        f.write(response.content)

    return path
