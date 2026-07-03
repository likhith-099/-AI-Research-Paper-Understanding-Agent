import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

from tools.arxiv_loader import load_arxiv_paper
from tools.equation_extractor import extract_equations
from tools.section_detector import detect_sections_verbose
from tools.text_extractor import extract_text_from_pdf

from rag.chunking import chunk_text
from rag.embeddings import create_embeddings
from rag.vector_store import VectorStore
from rag.validation import (
    validate_chunk_metadata,
    validate_parent_child_mapping,
)

from retrieval.retiever import Retriever
from agent.orchestrator import analyze_paper


app = FastAPI()


class PaperRequest(BaseModel):
    paper_url: str


def _is_debug_requested(debug: bool) -> bool:
    # Allow both an explicit query parameter and an environment flag so developers can enable debug mode without changing clients.
    env_debug = os.getenv("DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    app_debug = os.getenv("APP_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    return bool(debug or env_debug or app_debug)


def _run_analysis(pdf_path: str, *, debug: bool = False):
    # Step 1 - extract text
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in the PDF.")

    section_debug = detect_sections_verbose(text)
    equations = extract_equations(text)

    # Step 2 - chunk text
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No analyzable chunks could be created from the PDF.")

    chunk_metadata_debug = validate_chunk_metadata(chunks)
    parent_child_debug = validate_parent_child_mapping(chunks)

    # Step 3 - create embeddings
    embeddings = create_embeddings(chunks)

    # Step 4 - build vector database
    vector_db = VectorStore(embeddings, chunks)

    # Step 5 - create retriever
    retriever = Retriever(
        vector_db,
        paper_text=text,
        equations=equations,
        section_debug={
            **section_debug,
            "chunk_metadata_debug": chunk_metadata_debug,
            "parent_child_debug": parent_child_debug,
        },
    )

    # Step 6 - run the full analysis, then return either the compact public payload or the full debug payload.
    return analyze_paper(retriever, include_debug=debug)


@app.post("/analyze-paper")
def analyze(request: PaperRequest, debug: bool = Query(default=False, description="Return internal debug metadata when true.")):
    # Download paper from URL
    pdf_path = load_arxiv_paper(request.paper_url)
    return _run_analysis(pdf_path, debug=_is_debug_requested(debug))


@app.post("/analyze-upload")
async def analyze_upload(
    file: UploadFile = File(...),
    debug: bool = Query(default=False, description="Return internal debug metadata when true."),
):
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

        return _run_analysis(temp_path, debug=_is_debug_requested(debug))
    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
