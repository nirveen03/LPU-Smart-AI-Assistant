# LPU Smart AI Assistant — RAG Chatbot

A local, source-grounded academic chatbot for seven LPU subject syllabi.

## 1. Requirements

- Windows 10/11
- Python 3.10 or 3.11 recommended
- VS Code
- Ollama installed and running
- The seven original syllabus PDFs

This starter project uses:
- PyMuPDF for PDF text extraction
- Sentence-Transformers for embeddings
- ChromaDB as persistent vector store
- Ollama for local answer generation
- Streamlit for the chat UI

## 2. Add your real PDFs

Copy the original PDFs into `data/pdfs/`. The filenames must contain their subject code:

- `PEL132.pdf`
- `MTH401.pdf`
- `INT335.pdf`
- `CSE306.pdf`
- `CSE276.pdf`
- `CSE205.pdf`
- `CSE202.pdf`

If your filenames differ, that is fine as long as the relevant code (for example `CSE202`) appears in the filename.

**Important:** The PDFs are not included in this starter ZIP. Use the actual university files. Do not substitute generic sample syllabi.

## 3. Create and activate a virtual environment

Open this folder in VS Code. Choose Terminal → New Terminal.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt terminal instead:

```cmd
venv\Scripts\activate.bat
```

## 4. Install Python packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first run of Sentence-Transformers downloads the embedding model. Internet is needed for this initial download.

## 5. Install and start Ollama

Install Ollama from https://ollama.com/

Then open a terminal:

```bash
ollama pull llama3.2
```

Check that it is available:

```bash
ollama list
```

Keep Ollama running while using the chatbot. If you use another model, update `OLLAMA_MODEL` in your environment.

## 6. Index the PDFs

From the project root:

```bash
python ingest.py
```

The script extracts text page by page, splits it into overlapping chunks, creates embeddings, and stores the chunks with subject and page metadata in `chroma_db/`.

If a PDF page has no selectable text, the script prints a warning. This often means the page is scanned. OCR is not automatically performed by this starter version; run OCR on that PDF or add a Tesseract-based OCR step before indexing.

Re-run `python ingest.py` after adding or replacing PDFs. It replaces indexed chunks for each matching filename.

## 7. Launch the chatbot

```bash
streamlit run app.py
```

Streamlit will show a local URL, normally `http://localhost:8501`.

## 8. How to use

1. Select a subject from the sidebar, or choose All subjects.
2. Ask a question.
3. Read the answer and expand "View retrieved sources" to inspect the source documents and pages.

Subject selection filters retrieval. The answer model is instructed to use only retrieved source excerpts and to say when evidence is insufficient.

## 9. Important accuracy notes

- Only the PDFs you place in `data/pdfs/` are indexed.
- University policies such as attendance and exam eligibility are not in the syllabus corpus unless you add official policy documents.
- Unit detection is best-effort metadata. Always verify exact unit headings in the original PDF.
- Similarity thresholds need testing against your own labelled questions. The default is a starting point, not a validated cutoff.
- A language model can still make mistakes. Check citations and compare answers with the original PDF before presenting them as verified.
- The sample evaluation CSV is a blank template. Populate expected answers and page numbers from the actual PDFs, then calculate metrics from real runs.

## 10. Troubleshooting

### `No PDFs found`
Put your original PDFs in `data/pdfs/`.

### `model not found`
Run `ollama pull llama3.2`, or change `OLLAMA_MODEL` to a model shown by `ollama list`.

### Cannot connect to Ollama
Make sure the Ollama application/service is running. Default address is `http://localhost:11434`.

### No selectable text on a PDF page
The page may be scanned. OCR it before indexing. You can install Tesseract OCR and add a PyMuPDF + pytesseract OCR fallback.

### Out of memory / slow embedding
Close other applications, process fewer PDFs at a time, or use a machine with more RAM. The first embedding model download and indexing can take time.

## 11. Suggested next improvements

- Add OCR fallback for scanned PDFs.
- Extract course outcomes, credits, practicals, and textbooks into structured tables.
- Add administrator authentication and document upload.
- Add a FastAPI backend if you need a separate frontend.
- Implement a labelled evaluation runner for Recall@k, MRR, F1, faithfulness, hallucination rate, and refusal accuracy.
- Add feedback collection and logging with privacy safeguards.
