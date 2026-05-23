import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

from tools.arxiv_loader import load_arxiv_paper
from tools.text_extractor import extract_text_from_pdf

from rag.chunking import chunk_text
from rag.embeddings import create_embeddings
from rag.vector_store import VectorStore

from retrieval.retiever import Retriever
from agent.orchestrator import analyze_paper


app = FastAPI()


class PaperRequest(BaseModel):
    paper_url: str


def _run_analysis(pdf_path: str):
    # Step 1 - extract text
    text = extract_text_from_pdf(pdf_path)

    # Step 2 - chunk text
    chunks = chunk_text(text)

    # Step 3 - create embeddings
    embeddings = create_embeddings(chunks)

    # Step 4 - build vector database
    vector_db = VectorStore(embeddings, chunks)

    # Step 5 - create retriever
    retriever = Retriever(vector_db)

    # Step 6 - run full agent analysis
    return analyze_paper(retriever)


@app.post("/analyze-paper")
def analyze(request: PaperRequest):
    # Download paper from URL
    pdf_path = load_arxiv_paper(request.paper_url)
    return _run_analysis(pdf_path)


@app.post("/analyze-upload")
async def analyze_upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        return _run_analysis(temp_path)
    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
