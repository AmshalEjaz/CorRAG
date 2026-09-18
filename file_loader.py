import os
from pypdf import PdfReader
import ollama
import chromadb

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="local_docs_crag")

def load_pdf(file_path: str) -> str:
    """PDF file se text extract karta hai."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Bade text ko chote chunks mein divide karta hai."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def process_and_store_file(file_path: str):
    """File ko read, chunk aur ChromaDB mein store karta hai."""
    if not os.path.exists(file_path):
        print(f"❌ File nahi mili: {file_path}")
        return

    print(f"📄 Processing file: {file_path}...")
    
    # Check file type
    if file_path.endswith(".pdf"):
        full_text = load_pdf(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    else:
        print("❌ Unsupported file format. Only .pdf and .txt allowed.")
        return

    # Text Chunking
    chunks = chunk_text(full_text)
    print(f"🧩 Created {len(chunks)} chunks from document.")

    # Embed & Store (Now using the dedicated embedding model)
    embeddings = []
    ids = []
    base_name = os.path.basename(file_path)
    
    for idx, chunk in enumerate(chunks):
        response = ollama.embed(model="nomic-embed-text", input=chunk)
        embeddings.append(response["embeddings"][0])
        ids.append(f"{base_name}_chunk_{idx}")

    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )
    print(f"✅ Successfully ingested '{base_name}' into database!")

if __name__ == "__main__":
    test_file = "sample.txt" 
    
    if not os.path.exists(test_file):
        with open(test_file, "w") as f:
            f.write("Project Beta uses PostgreSQL database and React frontend framework.")
            
    process_and_store_file(test_file)