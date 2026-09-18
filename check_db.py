import chromadb

# Connect to the local database folder
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="local_docs_crag")

# Retrieve all stored documents and IDs
data = collection.get()

print("--- Stored Documents in chroma_db ---")
for doc_id, doc_text in zip(data["ids"], data["documents"]):
    print(f"ID: {doc_id} | Text: {doc_text}")