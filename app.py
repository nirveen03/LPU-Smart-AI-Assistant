import streamlit as st
from rag import answer_question, get_index_status, OLLAMA_MODEL
from subjects import SUBJECTS

st.set_page_config(
    page_title="LPU Smart AI Assistant",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 LPU Smart AI Assistant")
st.caption("Your academic assistant for answers grounded in uploaded university documents.")

with st.sidebar:
    st.header("Chat settings")
    selected_subject = st.selectbox(
        "Choose a subject",
        options=["ALL"] + list(SUBJECTS.keys()),
        format_func=lambda code: (
            "All subjects" if code == "ALL"
            else f"{code} — {SUBJECTS[code]}"
        ),
    )
    st.divider()
    st.write("**Knowledge base**")
    try:
        chunk_count = get_index_status()
        st.metric("Indexed text chunks", chunk_count)
    except Exception as exc:
        st.warning("Knowledge base is not ready yet.")
        st.caption(str(exc))
    st.caption(f"LLM: {OLLAMA_MODEL}")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.expander("Example questions"):
    st.markdown(
        "- What are the units in CSE306?\n"
        "- Explain Unit IV of CSE202.\n"
        "- What are the course outcomes of CSE276?\n"
        "- Which subjects mention graph theory?\n"
        "- What practicals are listed in the syllabus?"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View retrieved sources"):
                for source in message["sources"]:
                    st.write(
                        f"- **{source['subject_code']} — {source['subject_name']}** | "
                        f"{source['document']}, page {source['page']} "
                        f"| similarity {source['similarity']}"
                    )

question = st.chat_input("Ask a question about your LPU subjects...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching university documents and preparing an answer..."):
            try:
                result = answer_question(
                    question,
                    subject_code=selected_subject,
                )
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("View retrieved sources"):
                        for source in result["sources"]:
                            st.write(
                                f"- **{source['subject_code']} — {source['subject_name']}** | "
                                f"{source['document']}, page {source['page']} "
                                f"| similarity {source['similarity']}"
                            )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except Exception as exc:
                error_message = (
                    f"Something went wrong: {exc}\n\n"
                    "Check that the PDFs are indexed and Ollama is running "
                    "with the selected model installed."
                )
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                })
