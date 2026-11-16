import os
import re
from pathlib import Path
from typing import Optional

import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from ollama import Client
from subjects import SUBJECTS, subject_from_filename

ROOT = Path(__file__).parent
PDF_DIR = ROOT / "data" / "pdfs"
DB_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "lpu_academic_documents"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.25"))

_embedding_model = None
_chroma_client = None
_collection = None
_ollama_client = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = Client(host=OLLAMA_HOST)
    return _ollama_client


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_unit(text: str) -> str:
    """Best-effort unit heading detection; source text remains authoritative."""
    patterns = [
        r"\bUNIT\s*[-:]?\s*([IVX]+|\d+)\b",
        r"\bUNIT\s+([A-Z])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return "Not detected"


def split_into_chunks(text: str, chunk_size: int = 1100, overlap: int = 180):
    """Character-based chunking with overlap; works without extra tokenizers."""
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Prefer ending at a paragraph or sentence boundary.
            boundary = max(text.rfind("\n", start, end),
                           text.rfind(". ", start, end))
            if boundary > start + int(chunk_size * 0.55):
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_pdf_pages(pdf_path: Path):
    """Extract page text. OCR is not silently attempted; scanned pages are reported."""
    pages = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append((page_index + 1, text))
            else:
                print(
                    f"WARNING: {pdf_path.name}, page {page_index + 1} "
                    "has no selectable text. It may be scanned and needs OCR."
                )
    return pages


def ingest_pdfs(replace_existing: bool = True):
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found. Add your syllabus PDFs to: {PDF_DIR}"
        )

    collection = get_collection()
    model = get_embedding_model()

    # Delete previously indexed records for each document being replaced.
    for pdf_path in pdf_files:
        code, _ = subject_from_filename(pdf_path.name)
        if replace_existing:
            collection.delete(where={"source_file": pdf_path.name})

        pages = extract_pdf_pages(pdf_path)
        if not pages:
            print(f"WARNING: No text extracted from {pdf_path.name}; skipped.")
            continue

        ids, docs, metas = [], [], []
        for page_number, page_text in pages:
            chunks = split_into_chunks(page_text)
            for chunk_index, chunk in enumerate(chunks):
                chunk_id = f"{pdf_path.stem}_p{page_number}_c{chunk_index}"
                unit = detect_unit(chunk)
                metadata = {
                    "document_id": pdf_path.stem,
                    "document_name": pdf_path.name,
                    "document_type": "syllabus",
                    "subject_code": code,
                    "subject_name": SUBJECTS.get(code, "Unclassified document"),
                    "unit_number": unit,
                    "section": "Extracted page text",
                    "page_number": int(page_number),
                    "source_file": pdf_path.name,
                }
                ids.append(chunk_id)
                docs.append(chunk)
                metas.append(metadata)

        if docs:
            embeddings = model.encode(
                docs, normalize_embeddings=True, show_progress_bar=False
            ).tolist()
            # Add in batches to avoid oversized requests.
            batch_size = 64
            for i in range(0, len(docs), batch_size):
                collection.add(
                    ids=ids[i:i + batch_size],
                    documents=docs[i:i + batch_size],
                    metadatas=metas[i:i + batch_size],
                    embeddings=embeddings[i:i + batch_size],
                )
            print(f"Indexed {pdf_path.name}: {len(docs)} chunks ({len(pages)} pages).")
        else:
            print(f"No usable text chunks in {pdf_path.name}.")

    return collection.count()


def retrieve(question: str, subject_code: Optional[str] = None, top_k: int = TOP_K):
    collection = get_collection()
    model = get_embedding_model()

    if collection.count() == 0:
        raise RuntimeError("Knowledge base is empty. Run: python ingest.py")

    query_embedding = model.encode(
        [question], normalize_embeddings=True
    ).tolist()[0]

    where = None
    if subject_code and subject_code != "ALL":
        where = {"subject_code": subject_code}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(docs, metas, distances):
        # Chroma cosine distance: lower is more similar. Convert to similarity.
        similarity = 1.0 - float(distance)
        hits.append({
            "text": document,
            "metadata": metadata,
            "similarity": similarity,
        })
    return hits


def answer_question(question: str, subject_code: Optional[str] = None):
    hits = retrieve(question, subject_code=subject_code)
    useful_hits = [h for h in hits if h["similarity"] >= MIN_SIMILARITY]

    if not useful_hits:
        return {
            "answer": (
                "I couldn't find sufficient supporting information in the "
                "currently indexed university documents. Please check the "
                "relevant university document or ask an authorised department."
            ),
            "sources": [],
            "retrieved": hits,
        }

    context_parts = []
    for index, hit in enumerate(useful_hits, start=1):
        m = hit["metadata"]
        context_parts.append(
            f"[SOURCE {index}]\n"
            f"Subject: {m.get('subject_code')} - {m.get('subject_name')}\n"
            f"Document: {m.get('document_name')}\n"
            f"Page: {m.get('page_number')}\n"
            f"Unit detected: {m.get('unit_number')}\n"
            f"Text:\n{hit['text']}"
        )
    context = "\n\n".join(context_parts)

    system_prompt = """You are LPU Smart AI Assistant, an academic assistant.
Answer using ONLY the supplied source excerpts.
Do not use outside knowledge to fill gaps in university-specific information.
If the excerpts do not support an answer, say that you could not find sufficient
information in the available documents.
Do not invent subject details, unit numbers, page numbers, policies, marks,
attendance rules, dates, or textbooks.
If sources disagree, clearly describe the disagreement and cite both.
Explain concepts clearly and at a student-friendly level. The student may use
English or Hinglish; respond in the language used by the student.
Cite claims inline like [CSE202.pdf, page 2].
Distinguish what the document explicitly says from your explanation.
"""

    user_prompt = f"""STUDENT QUESTION:
{question}

RETRIEVED UNIVERSITY DOCUMENT EXCERPTS:
{context}

Write a clear answer. Include the subject code/name where supported and cite
the document filename and page for factual syllabus claims."""

    response = get_ollama_client().chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.1},
    )

    sources = []
    seen = set()
    for hit in useful_hits:
        m = hit["metadata"]
        key = (m.get("document_name"), m.get("page_number"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "document": m.get("document_name"),
                "subject_code": m.get("subject_code"),
                "subject_name": m.get("subject_name"),
                "page": m.get("page_number"),
                "unit": m.get("unit_number"),
                "similarity": round(hit["similarity"], 3),
            })

    return {
        "answer": response["message"]["content"],
        "sources": sources,
        "retrieved": useful_hits,
    }


def get_index_status():
    collection = get_collection()
    return collection.count()
