import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

from tools.arxiv_loader import load_arxiv_paper
from tools.equation_extractor import extract_equations
from tools.paper_metadata import annotate_chunk_pages, extract_pdf_metadata, extract_pdf_pages
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
from analysis.chat_service import answer_question


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PaperRequest(BaseModel):
    paper_url: str


class ChatRequest(BaseModel):
    question: str
    paper: dict


class Citation(BaseModel):
    section: str
    page: str
    chunk_id: int | None = None
    parent_id: int | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    routed_sections: list[str] = Field(default_factory=list)


@app.get("/health")
def health():
    return {"status": "ok"}


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

    paper_metadata = extract_pdf_metadata(pdf_path)
    page_texts = extract_pdf_pages(pdf_path)

    section_debug = detect_sections_verbose(text)
    equations = extract_equations(text)

    # Step 2 - chunk text
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No analyzable chunks could be created from the PDF.")

    annotate_chunk_pages(chunks, page_texts)
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
        paper_metadata=paper_metadata,
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

        return await run_in_threadpool(_run_analysis, temp_path, debug=_is_debug_requested(debug))
    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/chat-paper", response_model=ChatResponse)
def chat_paper(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")
    if not isinstance(request.paper, dict):
        raise HTTPException(status_code=400, detail="Paper payload is required.")

    response = answer_question(request.question, request.paper)
    return ChatResponse(**response)
