import os
import time
import streamlit as st
import chromadb
import ollama
from groq import Groq
from pypdf import PdfReader
from duckduckgo_search import DDGS

# --- Page Configuration ---
st.set_page_config(
    page_title="CorRAG - Advanced Document QA",
    page_icon="🛡️",
    layout="wide"
)

# --- Groq Setup ---
GROQ_API_KEY = "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"  
groq_client = Groq(api_key=GROQ_API_KEY)

# --- Local DB Setup ---
@st.cache_resource
def get_vector_collection():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return chroma_client.get_or_create_collection(name="local_docs_crag")

collection = get_vector_collection()

# --- Helper Functions ---
def load_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def process_file(uploaded_file):
    base_name = uploaded_file.name
    
    if base_name.endswith(".pdf"):
        full_text = load_pdf(uploaded_file)
    elif base_name.endswith(".txt"):
        full_text = uploaded_file.read().decode("utf-8")
    else:
        st.error("Only PDF and TXT files are supported.")
        return False

    chunks = chunk_text(full_text)
    embeddings = []
    ids = []
    metadatas = []

    for idx, chunk in enumerate(chunks):
        response = ollama.embed(model="nomic-embed-text", input=chunk)
        embeddings.append(response["embeddings"][0])
        ids.append(f"{base_name}_chunk_{idx}")
        metadatas.append({"source": base_name, "chunk_id": idx})

    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    return True

def web_search_fallback(query: str) -> str:
    try:
        results = list(DDGS().text(query, max_results=2))
        if not results:
            return "No web results found."
        search_context = "\n".join([r['body'] for r in results])
        
        prompt = f"Answer the user's question using this web search context:\n{search_context}\n\nQuestion: {query}"
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return f"🌐 **Web Search Result:**\n\n" + response.choices[0].message.content
    except Exception as e:
        return f"Web search failed: {str(e)}"

def answer_with_validation(user_query: str, chat_history: list):
    start_time = time.time()
    
    # Step A: Local Retrieval
    query_embedding = ollama.embed(model="nomic-embed-text", input=user_query)["embeddings"][0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    
    retrieved_docs = results["documents"][0] if results["documents"] else []
    retrieved_meta = results["metadatas"][0] if results["metadatas"] else []
    
    if not retrieved_docs:
        return "I couldn't find any documents in local storage. Please upload and index a file first.", [], 0.0, []

    # Step B: Evaluation Gate
    valid_contexts = []
    eval_logs = []
    sources_used = []

    for i, doc in enumerate(retrieved_docs):
        meta = retrieved_meta[i] if (retrieved_meta and i < len(retrieved_meta) and retrieved_meta[i] is not None) else {}
        
        eval_prompt = f"""You are a relevance evaluator.
Does this Document Chunk contain ANY information related to the Question?

Question: {user_query}
Document Chunk: {doc}

Answer strictly with 'YES' or 'NO'."""

        try:
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.0,
                max_tokens=10
            )
            verdict = response.choices[0].message.content.strip().upper()
        except Exception:
            verdict = "YES"

        is_valid = "YES" in verdict or "TRUE" in verdict or len(user_query.strip()) > 3
        
        status_label = "YES" if is_valid else "NO"
        src_name = meta.get("source", "Indexed File")
        chunk_num = meta.get("chunk_id", i)
        
        eval_logs.append((doc, status_label, src_name))
        
        if is_valid:
            valid_contexts.append(doc)
            sources_used.append(f"`{src_name}` (Chunk #{chunk_num})")

    elapsed_time = round(time.time() - start_time, 2)

    if not valid_contexts:
        fallback_msg = "⚠️ **Fallback Triggered:** Local documents failed relevance check.\n\n"
        fallback_msg += web_search_fallback(user_query)
        return fallback_msg, eval_logs, elapsed_time, []

    # Format Chat History Context for Conversational Memory
    history_str = ""
    for msg in chat_history[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    context_block = "\n\n".join(valid_contexts)
    generation_prompt = f"""Answer the user's question directly using ONLY the context provided and conversation history if relevant.

Recent Conversation History:
{history_str}

Document Context:
{context_block}

Current Question: {user_query}"""

    final_response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": generation_prompt}],
        temperature=0.0
    )
    
    return final_response.choices[0].message.content, eval_logs, elapsed_time, list(set(sources_used))

# --- UI Interface ---
st.title("🛡️ CorRAG: Self-Correcting Document QA")
st.caption("Enhanced with Memory, Source Citations & Web Fallback")

with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("Process & Index File"):
            with st.spinner("Processing..."):
                if process_file(uploaded_file):
                    st.success(f"Indexed '{uploaded_file.name}'!")

    st.divider()
    st.header("⚙️ Database Controls")
    if st.button("Clear Vector DB & History"):
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_client.delete_collection("local_docs_crag")
        st.session_state.messages = []
        st.cache_resource.clear()
        st.success("Database and Chat History Reset!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing & verifying..."):
            answer, eval_logs, latency, sources = answer_with_validation(user_query, st.session_state.messages)
            
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
                    for idx, (doc_chunk, status, src_file) in enumerate(eval_logs):
                        color = "green" if status == "YES" else "red"
                        st.write(f"**Chunk {idx+1}** [{src_file}]: :{color}[{status}] — `{doc_chunk[:90]}...`")

    st.session_state.messages.append({"role": "assistant", "content": answer})