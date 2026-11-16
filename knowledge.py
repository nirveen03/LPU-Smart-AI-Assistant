import os
import re
from pathlib import Path
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from subjects import SUBJECTS

ROOT = Path(__file__).parent
PDF_DIR = ROOT / "data" / "pdfs"
DB_DIR = ROOT / "chroma_db"
COLLECTION = "lpu_knowledge"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_embedder = None
_client = None
_collection = None

def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def collection():
    global _client, _collection
    if _collection is None:
        DB_DIR.mkdir(exist_ok=True)
        _client = chromadb.PersistentClient(path=str(DB_DIR))
        _collection = _client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )
    return _collection

def subject_for(filename):
    upper = filename.upper()
    for code, name in SUBJECTS.items():
        if code in upper:
            return code, name
    return "UNKNOWN", "Unclassified document"

def clean(text):
    text = text.replace("\\x00", " ")
    text = re.sub(r"[ \\t]+", " ", text)
    text = re.sub(r"\\n{3,}", "\\n\\n", text)
    return text.strip()

def chunks(text, size=1100, overlap=160):
    text = clean(text)
    output, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(text.rfind("\\n", start, end), text.rfind(". ", start, end))
            if boundary > start + size * 0.55:
                end = boundary + 1
        part = text[start:end].strip()
        if part:
            output.append(part)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return output

def index_pdfs():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = list(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {PDF_DIR}. You can still use general chat without PDFs.")
    col = collection()
    model = embedder()
    total = 0
    for pdf in pdfs:
        code, name = subject_for(pdf.name)
        col.delete(where={"source_file": pdf.name})
        records = []
        with fitz.open(pdf) as doc:
            for pno, page in enumerate(doc, start=1):
                text = clean(page.get_text("text"))
                if not text:
                    print(f"Warning: {pdf.name} page {pno} has no selectable text (possibly scanned).")
                    continue
                for ci, part in enumerate(chunks(text)):
                    records.append((f"{pdf.stem}_p{pno}_c{ci}", part, {
                        "source_file": pdf.name,
                        "subject_code": code,
                        "subject_name": name,
                        "page": int(pno),
                        "document_type": "syllabus"
                    }))
        for start in range(0, len(records), 64):
            batch = records[start:start+64]
            texts = [x[1] for x in batch]
            vectors = model.encode(texts, normalize_embeddings=True).tolist()
            col.add(
                ids=[x[0] for x in batch],
                documents=texts,
                metadatas=[x[2] for x in batch],
                embeddings=vectors
            )
        total += len(records)
        print(f"Indexed {pdf.name}: {len(records)} chunks")
    return total

def retrieve(question, subject_code="ALL", top_k=5):
    col = collection()
    if col.count() == 0:
        return []
    vector = embedder().encode([question], normalize_embeddings=True).tolist()[0]
    kwargs = dict(query_embeddings=[vector], n_results=min(top_k, col.count()),
                  include=["documents", "metadatas", "distances"])
    if subject_code != "ALL":
        kwargs["where"] = {"subject_code": subject_code}
    result = col.query(**kwargs)
    hits = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({"text": doc, "meta": meta, "similarity": 1-float(dist)})
    return hits

def indexed_count():
    return collection().count()
