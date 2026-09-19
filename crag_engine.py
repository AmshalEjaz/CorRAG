from __future__ import annotations

import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import chromadb
import ollama
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

# -----------------------------------------------------------------------------
# Paths / configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "local_docs_crag"
EMBED_MODEL = "nomic-embed-text"
GROQ_MODEL = "openai/gpt-oss-20b"

load_dotenv(dotenv_path=ENV_PATH, override=True)


def _read_api_key() -> str:
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not key or key.upper().startswith("PASTE_"):
        raise RuntimeError(
            f"GROQ_API_KEY is missing. Put a valid Groq key in {ENV_PATH}."
        )
    return key


def get_groq_client() -> Groq:
    return Groq(api_key=_read_api_key())


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def collection_count() -> int:
    return get_collection().count()


# -----------------------------------------------------------------------------
# Embeddings / ingestion
# -----------------------------------------------------------------------------
def create_embedding(text: str) -> list[float]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Cannot create an embedding for empty text.")

    try:
        response = ollama.embed(model=EMBED_MODEL, input=text)
        embeddings = response.get("embeddings")
        if not embeddings:
            raise ValueError("Ollama returned no embeddings.")
        return embeddings[0]
    except Exception as exc:
        raise RuntimeError(
            "Could not create embeddings. Make sure Ollama is running and "
            f"'{EMBED_MODEL}' is installed. Original error: {exc}"
        ) from exc


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    words = (text or "").split()
    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), step)
        if words[i : i + chunk_size]
    ]


def extract_pdf_text(file_or_bytes: Any) -> str:
    if isinstance(file_or_bytes, (bytes, bytearray)):
        source = BytesIO(file_or_bytes)
    else:
        source = file_or_bytes

    reader = PdfReader(source)
    pages: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        if extracted.strip():
            pages.append(extracted)
    return "\n".join(pages)


def _replace_source_chunks(source_name: str, chunks: list[str]) -> int:
    if not chunks:
        raise ValueError(
            "No readable text was found in this file. If it is a scanned PDF, "
            "OCR is required before indexing."
        )

    collection = get_collection()

    # Remove old chunks from the same file so re-indexing never leaves stale data.
    try:
        old = collection.get(where={"source": source_name})
        old_ids = old.get("ids") or []
        if old_ids:
            collection.delete(ids=old_ids)
    except Exception:
        # Old databases may contain chunks without metadata. Upsert still works.
        pass

    embeddings: list[list[float]] = []
    ids: list[str] = []
    metadatas: list[dict[str, Any]] = []

    safe_source = source_name.replace("/", "_").replace("\\", "_")
    for idx, chunk in enumerate(chunks):
        embeddings.append(create_embedding(chunk))
        ids.append(f"{safe_source}_chunk_{idx}")
        metadatas.append({"source": source_name, "chunk_id": idx})

    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    return len(chunks)


def process_uploaded_file(filename: str, data: bytes) -> int:
    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        full_text = extract_pdf_text(data)
    elif lower_name.endswith(".txt"):
        full_text = data.decode("utf-8", errors="replace")
    else:
        raise ValueError("Only PDF and TXT files are supported.")

    return _replace_source_chunks(filename, chunk_text(full_text))


def process_file_path(file_path: str | Path) -> int:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = path.read_bytes()
    return process_uploaded_file(path.name, data)


def clear_vector_db() -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    client.get_or_create_collection(name=COLLECTION_NAME)


# -----------------------------------------------------------------------------
# Groq helpers
# -----------------------------------------------------------------------------
def _groq_text(
    messages: list[dict[str, str]],
    *,
    max_completion_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Return non-empty Groq text, with one retry for GPT-OSS reasoning output."""
    client = get_groq_client()

    attempts = [
        {
            "reasoning_effort": "low",
            "include_reasoning": False,
            "max_completion_tokens": max_completion_tokens,
        },
        {
            "include_reasoning": False,
            "max_completion_tokens": max(max_completion_tokens, 256),
        },
    ]

    last_error: Exception | None = None
    for extra in attempts:
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                stream=False,
                **extra,
            )
            content = response.choices[0].message.content or ""
            content = content.strip()
            if content:
                return content
        except Exception as exc:
            last_error = exc

    if last_error:
        raise RuntimeError(f"Groq request failed: {last_error}") from last_error
    raise RuntimeError("Groq returned an empty response after retrying.")


def check_relevance(user_query: str, document: str) -> tuple[bool, str]:
    prompt = f"""You are a strict document relevance classifier.

QUESTION:
{user_query}

DOCUMENT CHUNK:
{document}

Does the document chunk contain information that could help answer the question?
Reply with exactly one word: YES or NO."""

    try:
        raw = _groq_text(
            [
                {
                    "role": "system",
                    "content": "Return only YES or NO. No explanation.",
                },
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=128,
            temperature=0.0,
        )
        first = raw.strip().upper().split()[0].strip(".,:;!?") if raw.strip() else ""
        return first == "YES", raw
    except Exception as exc:
        # A failed evaluator must never silently approve unrelated context.
        return False, f"ERROR: {exc}"


def generate_answer(
    user_query: str,
    valid_contexts: list[str],
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    history_lines: list[str] = []
    for msg in (chat_history or [])[-4:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = (msg.get("content") or "").strip()
        if content:
            history_lines.append(f"{role}: {content}")

    history_block = "\n".join(history_lines) or "(none)"
    context_block = "\n\n--- VERIFIED CHUNK ---\n\n".join(valid_contexts)

    prompt = f"""You are a grounded document QA assistant.

Rules:
1. Answer using ONLY the verified document context below.
2. Conversation history may help resolve references, but it is not a factual source.
3. Do not invent facts or use outside knowledge.
4. If the answer is not contained in the verified context, say exactly:
   The available documents do not contain enough information to answer that question.
5. Be clear and concise.

RECENT CONVERSATION:
{history_block}

VERIFIED DOCUMENT CONTEXT:
{context_block}

CURRENT QUESTION:
{user_query}

ANSWER:"""

    return _groq_text(
        [{"role": "user", "content": prompt}],
        max_completion_tokens=768,
        temperature=0.0,
    )


# -----------------------------------------------------------------------------
# Main CRAG flow
# -----------------------------------------------------------------------------
def answer_with_validation(
    user_query: str,
    chat_history: list[dict[str, str]] | None = None,
) -> tuple[str, list[dict[str, Any]], float, list[str]]:
    start_time = time.time()
    query = (user_query or "").strip()

    if not query:
        return "Please enter a question.", [], 0.0, []

    collection = get_collection()
    total_chunks = collection.count()
    if total_chunks == 0:
        return (
            "I couldn't find any indexed documents. Please upload and index a PDF or TXT file first.",
            [],
            round(time.time() - start_time, 2),
            [],
        )

    try:
        query_embedding = create_embedding(query)
        n_results = min(5, total_chunks)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        return (
            f"Local retrieval failed: {exc}",
            [],
            round(time.time() - start_time, 2),
            [],
        )

    docs = (results.get("documents") or [[]])[0] or []
    metas = (results.get("metadatas") or [[]])[0] or []
    distances = (results.get("distances") or [[]])[0] or []

    if not docs:
        return (
            "I couldn't retrieve any document chunks from local storage.",
            [],
            round(time.time() - start_time, 2),
            [],
        )

    valid_contexts: list[str] = []
    eval_logs: list[dict[str, Any]] = []
    sources_used: list[str] = []

    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        distance = distances[i] if i < len(distances) else None
        is_relevant, raw_verdict = check_relevance(query, doc)

        source = meta.get("source", "Indexed File")
        chunk_id = meta.get("chunk_id", i)

        eval_logs.append(
            {
                "document": doc,
                "status": "YES" if is_relevant else "NO",
                "source": source,
                "chunk_id": chunk_id,
                "distance": distance,
                "verdict": raw_verdict,
            }
        )

        if is_relevant:
            valid_contexts.append(doc)
            sources_used.append(f"`{source}` (Chunk #{chunk_id})")

    elapsed = round(time.time() - start_time, 2)

    if not valid_contexts:
        return (
            "I found indexed documents, but none of the retrieved chunks were relevant enough to answer this question.",
            eval_logs,
            elapsed,
            [],
        )

    try:
        answer = generate_answer(query, valid_contexts, chat_history)
    except Exception as exc:
        return (
            f"The answer generation step failed: {exc}",
            eval_logs,
            round(time.time() - start_time, 2),
            sorted(set(sources_used)),
        )

    return (
        answer,
        eval_logs,
        round(time.time() - start_time, 2),
        sorted(set(sources_used)),
    )


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("CRAG ENGINE HEALTH CHECK")
    print("=" * 60)
    print("Project folder:", BASE_DIR)
    print(".env path:", ENV_PATH)
    print("ChromaDB path:", CHROMA_PATH)
    print("Indexed chunks:", collection_count())

    if len(sys.argv) > 1:
        cli_query = " ".join(sys.argv[1:]).strip()
        answer, logs, latency, sources = answer_with_validation(cli_query, [])
        print("\nQuestion:", cli_query)
        for idx, log in enumerate(logs, start=1):
            print(
                f"Chunk {idx}: status={log['status']} source={log['source']} "
                f"distance={log['distance']} verdict={log['verdict']!r}"
            )
        print("\nAnswer:", answer)
        print("Sources:", sources)
        print("Latency:", latency, "seconds")
    else:
        print("\nHealth check complete.")
        print('To test a real question, run: py crag_engine.py "your question"')
