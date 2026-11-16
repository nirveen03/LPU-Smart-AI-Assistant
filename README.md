# LPU Smart AI Assistant — Hybrid Chatbot

This is an interactive chatbot, not just a PDF reader. It supports:
- General AI conversation (learning, coding, explanations, MCQs, study help)
- Optional RAG retrieval from the seven LPU syllabus PDFs
- Subject filtering
- Source file and page references when retrieved syllabus passages support an answer
- English and Hinglish responses
- Streamlit chat interface and conversation history during the current session

## Stack
Python, Streamlit, Ollama, Sentence-Transformers, ChromaDB, PyMuPDF.

## A. Install prerequisites
1. Install Python 3.10 or 3.11.
2. Install VS Code.
3. Install Ollama from https://ollama.com/
4. Open a terminal and download a model:
   ```bash
   ollama pull llama3.2
   ```

## B. Open project in VS Code
Open this folder in VS Code, then open Terminal → New Terminal.

Windows PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
If PowerShell activation is blocked, use Command Prompt:
```cmd
venv\Scripts\activate.bat
```

Install packages:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## C. Run general chatbot first
You can use the chatbot even before adding PDFs:
```bash
streamlit run app.py
```
Make sure Ollama is running and `llama3.2` appears in `ollama list`.

The sidebar toggle "Use university knowledge (RAG)" lets you switch between general AI chat and hybrid chat.

## D. Add the seven real syllabus PDFs
Copy original PDFs into `data/pdfs/`. Filenames must include their subject code:
- PEL132.pdf
- MTH401.pdf
- INT335.pdf
- CSE306.pdf
- CSE276.pdf
- CSE205.pdf
- CSE202.pdf

The ZIP does not contain university PDFs. Use your own original documents. Do not substitute generic content.

Index the PDFs:
```bash
python ingest.py
```
Then restart Streamlit or refresh the app. The sidebar should show indexed chunks.

If a PDF page has no selectable text, it may be scanned. This starter version reports the page but does not perform OCR. OCR can be added as a later improvement.

## E. Run the app
```bash
streamlit run app.py
```
Usually opens at http://localhost:8501

## How hybrid mode works
- General questions: the assistant answers using the local Ollama model.
- University/syllabus questions: the app retrieves relevant text chunks from ChromaDB and passes them to the model.
- If no sufficiently relevant excerpt is found, the model is told not to invent LPU-specific syllabus or policy facts.
- The source expander shows retrieved source file, subject and page. Similarity is a retrieval score, not proof that an answer is correct.

## Important limitations
This is a working starter chatbot, not a production-validated university system. Review answers against original documents. Attendance rules, examination policies and current notices must be added as official documents before the bot can answer them reliably. No fake university policies are included. Add authentication, OCR, automated evaluation, data retention controls and deployment hardening before production use.

## Troubleshooting
- Model missing: `ollama pull llama3.2`
- Ollama connection error: open Ollama and check `http://localhost:11434`
- No PDF chunks: check `data/pdfs/`, then run `python ingest.py`
- Slow first run: embedding model downloads once and may take a few minutes.
