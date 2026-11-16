import streamlit as st
from ollama import Client
from chat_engine import generate, OLLAMA_HOST, OLLAMA_MODEL
from knowledge import indexed_count
from subjects import SUBJECTS

st.set_page_config(page_title="LPU Smart AI Assistant", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1050px; padding-top: 2rem;}
.hero {padding: 1.3rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#13243a,#24527a); color:white; margin-bottom:1rem;}
.hero h1 {color:white; margin:0;}
.hero p {color:#dce9f5; margin:.35rem 0 0 0;}
.small-muted {color: #8b95a5; font-size: .85rem;}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("🎓 LPU Assistant")
    st.caption("Hybrid AI + University Knowledge")
    st.divider()
    subject = st.selectbox(
        "Subject filter",
        ["ALL"] + list(SUBJECTS.keys()),
        format_func=lambda x: "All subjects" if x == "ALL" else f"{x} — {SUBJECTS[x]}"
    )
    use_knowledge = st.toggle("Use university knowledge (RAG)", value=True)
    st.caption("Turn off for general AI conversation.")
    st.divider()
    st.subheader("Knowledge base")
    try:
        count = indexed_count()
        st.metric("Indexed text chunks", count)
        if count == 0:
            st.info("No PDFs indexed yet. General chat still works.")
    except Exception:
        st.info("Knowledge base not initialized. General chat still works.")
    st.caption(f"AI model: {OLLAMA_MODEL}")
    if st.button("＋ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Tip: Add original syllabus PDFs to data/pdfs, then run python ingest.py.")

st.markdown("""
<div class="hero">
  <h1>✨ LPU Smart AI Assistant</h1>
  <p>Your personal AI study partner — ask, learn, practise and explore.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([3, 1])
with left:
    st.markdown("Ask about your subjects, programming, AI concepts, exam preparation or everyday learning.")
with right:
    st.caption("English • Hinglish")

if not st.session_state.messages:
    st.markdown("#### Try asking")
    suggestions = [
        "Explain the OSI model in simple Hinglish",
        "Create 10 MCQs on Python OOP",
        "Explain AGI and sustainable AI",
        "What topics are listed in CSE202 Unit IV?",
    ]
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 2].button(suggestion, use_container_width=True):
            st.session_state.pending_prompt = suggestion
            st.rerun()

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("sources"):
            with st.expander("📚 Sources used"):
                for source in item["sources"]:
                    st.write(f"**{source['subject']}** · {source['file']} · Page {source['page']} · similarity {source['similarity']}")
        if item.get("mode"):
            st.caption(item["mode"])

prompt = st.chat_input("Message LPU Smart AI Assistant...")
if not prompt and "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Keep conversation context bounded for local models.
                history = st.session_state.messages[-12:]
                answer, sources, mode = generate(
                    history, use_knowledge=use_knowledge, subject_code=subject
                )
                st.markdown(answer)
                if sources:
                    with st.expander("📚 Sources used"):
                        for source in sources:
                            st.write(f"**{source['subject']}** · {source['file']} · Page {source['page']} · similarity {source['similarity']}")
                st.caption(mode)
                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "sources": sources, "mode": mode
                })
            except Exception as exc:
                message = (
                    f"Could not reach the local AI model: `{exc}`\n\n"
                    "Check that Ollama is installed and running, and that the model is downloaded."
                )
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message})
