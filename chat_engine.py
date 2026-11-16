import os
from ollama import Client
from knowledge import retrieve

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
client = Client(host=OLLAMA_HOST)

GENERAL_SYSTEM = """You are LPU Smart AI Assistant, a friendly and capable student assistant.
Help with learning, programming, study planning, explanations, examples, and general questions.
Support English and Hinglish. Be clear and student-friendly.
Do not claim that an answer comes from LPU documents unless retrieved source excerpts are provided.
For university-specific rules, dates, attendance, marks, or official policies, explain that the student
needs an official source if no verified source is available."""

RAG_SYSTEM = """You are LPU Smart AI Assistant. Answer the student's university syllabus question
using ONLY the supplied source excerpts for claims about the syllabus.
Do not invent course outcomes, units, topics, practicals, books, page numbers, attendance rules,
exam policies, dates, or marks. If the excerpts do not support the answer, say you could not find
sufficient information in the indexed documents. Explain concepts clearly, in the student's language.
Cite factual syllabus claims inline using [filename, page N]. Clearly distinguish source facts from
additional explanatory examples. Do not present model knowledge as if it were in the PDF."""

def generate(messages, use_knowledge=True, subject_code="ALL"):
    last_question = messages[-1]["content"]
    hits = retrieve(last_question, subject_code=subject_code) if use_knowledge else []
    # Avoid treating weak semantic matches as reliable syllabus evidence.
    supported = [h for h in hits if h["similarity"] >= 0.28]

    if supported:
        context = "\n\n".join(
            f"[SOURCE]\nSubject: {h['meta']['subject_code']} - {h['meta']['subject_name']}\n"
            f"Document: {h['meta']['source_file']}\nPage: {h['meta']['page']}\n"
            f"Excerpt:\n{h['text']}"
            for h in supported
        )
        system = RAG_SYSTEM + "\n\nRETRIEVED EXCERPTS:\n" + context
        answer_mode = "University sources used"
    elif use_knowledge:
        system = GENERAL_SYSTEM + (
            "\n\nNo sufficiently relevant university document excerpt was retrieved for this question. "
            "You may answer general learning questions using your general knowledge, but for LPU-specific "
            "syllabus or policy facts, clearly say the indexed documents did not verify the answer."
        )
        answer_mode = "General AI response (no supporting syllabus passage found)"
    else:
        system = GENERAL_SYSTEM
        answer_mode = "General AI chat"

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "system", "content": system}] + messages,
        options={"temperature": 0.25}
    )
    sources = []
    seen = set()
    for h in supported:
        m = h["meta"]
        key = (m["source_file"], m["page"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": m["source_file"], "page": m["page"],
                "subject": f'{m["subject_code"]} — {m["subject_name"]}',
                "similarity": round(h["similarity"], 3)
            })
    return response["message"]["content"], sources, answer_mode
