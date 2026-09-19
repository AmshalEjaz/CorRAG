from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from crag_engine import (
    BASE_DIR,
    ENV_PATH,
    answer_with_validation,
    clear_vector_db,
    collection_count,
    get_groq_client,
    process_uploaded_file,
)

# Optional web fallback package. Prefer the current `ddgs` package, but keep
# compatibility with older environments that still have `duckduckgo_search`.
try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover
    try:
        from duckduckgo_search import DDGS
    except ImportError:  # pragma: no cover
        DDGS = None

load_dotenv(dotenv_path=ENV_PATH, override=True)

st.set_page_config(
    page_title="CorRAG - Advanced Document QA",
    page_icon="🛡️",
    layout="wide",
)


def web_search_fallback(query: str) -> str:
    if DDGS is None:
        return "Web fallback is unavailable because the DDGS package is not installed."

    try:
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return "No web results were found."

        snippets = []
        for item in results:
            title = item.get("title", "")
            body = item.get("body", "")
            href = item.get("href", "")
            snippets.append(f"Title: {title}\nSnippet: {body}\nURL: {href}")

        search_context = "\n\n".join(snippets)
        prompt = f"""Answer the question using ONLY the web snippets below.
If the snippets are insufficient, say so. Be concise.

WEB SNIPPETS:
{search_context}

QUESTION:
{query}"""

        client = get_groq_client()
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            reasoning_effort="low",
            include_reasoning=False,
            max_completion_tokens=768,
            stream=False,
        )
        content = (response.choices[0].message.content or "").strip()
        return "🌐 **Web Search Result**\n\n" + (content or "No answer was produced.")
    except Exception as exc:
        return f"Web search failed: {exc}"


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("🛡️ CorRAG: Self-Correcting Document QA")
st.caption("Local RAG with relevance validation, citations, memory, and optional web fallback")

# Do not crash the whole Streamlit page if the key is missing.
api_key = (os.getenv("GROQ_API_KEY") or "").strip()
key_ready = bool(api_key and not api_key.upper().startswith("PASTE_"))

if not key_ready:
    st.error(
        f"Groq API key is missing. Add GROQ_API_KEY to `{ENV_PATH.name}` in the project folder, then restart Streamlit."
    )

with st.sidebar:
    st.header("📄 Document Ingestion")
    st.caption(f"Indexed chunks: **{collection_count()}**")

    uploaded_file = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
    )

    if uploaded_file is not None and st.button("Process & Index File", use_container_width=True):
        try:
            with st.spinner("Reading, chunking, embedding, and indexing..."):
                chunk_count = process_uploaded_file(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )
            st.success(f"Indexed '{uploaded_file.name}' into {chunk_count} chunk(s).")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.header("⚙️ Database Controls")

    if st.button("Clear Vector DB & History", use_container_width=True):
        try:
            clear_vector_db()
            st.session_state.messages = []
            st.success("Vector database and chat history cleared.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not clear the database: {exc}")

    use_web_fallback = st.toggle(
        "Use web fallback when local docs are irrelevant",
        value=False,
        help="Off by default so document QA remains grounded in uploaded files.",
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input(
    "Ask a question about your documents...",
    disabled=not key_ready,
):
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and validating document context..."):
            answer, eval_logs, latency, sources = answer_with_validation(
                user_query,
                st.session_state.messages,
            )

            no_local_match = bool(eval_logs) and not any(
                log.get("status") == "YES" for log in eval_logs
            )

            if no_local_match and use_web_fallback:
                answer = (
                    "⚠️ **Local document fallback triggered.** No retrieved local chunk passed "
                    "the relevance check.\n\n"
                    + web_search_fallback(user_query)
                )

            st.markdown(answer)
            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"⚡ **Response Time:** `{latency}s`")
            with col2:
                if sources:
                    st.caption(f"📌 **Sources:** {', '.join(sources)}")

            if eval_logs:
                with st.expander("🔍 Guardrail / Evaluation Logs"):
                    for idx, log in enumerate(eval_logs, start=1):
                        status = log.get("status", "NO")
                        color = "green" if status == "YES" else "red"
                        source = log.get("source", "Indexed File")
                        chunk_id = log.get("chunk_id", idx - 1)
                        distance = log.get("distance")
                        verdict = log.get("verdict", "")
                        document = log.get("document", "")

                        distance_text = (
                            f"{distance:.4f}" if isinstance(distance, (int, float)) else "n/a"
                        )
                        st.markdown(
                            f"**Chunk {idx}** — `{source}` / chunk `{chunk_id}` — "
                            f":{color}[{status}] — distance `{distance_text}`"
                        )
                        st.caption(f"Evaluator: {verdict}")
                        st.code(document[:800] + ("..." if len(document) > 800 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})
