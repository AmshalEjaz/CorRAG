from crag_engine import CHROMA_PATH, COLLECTION_NAME, get_collection

collection = get_collection()
data = collection.get(include=["documents", "metadatas"])

print(f"ChromaDB: {CHROMA_PATH}")
print(f"Collection: {COLLECTION_NAME}")
print(f"Total chunks: {collection.count()}")
print("\n--- Stored Documents ---")

for idx, doc_id in enumerate(data.get("ids") or []):
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    doc = docs[idx] if idx < len(docs) else ""
    meta = metas[idx] if idx < len(metas) else {}
    print(f"\nID: {doc_id}")
    print(f"Metadata: {meta}")
    print(f"Text: {doc}")
