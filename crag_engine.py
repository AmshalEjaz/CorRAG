import os
import chromadb
import ollama
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found! Please check your .env file.")

groq_client = Groq(api_key=GROQ_API_KEY)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="local_docs_crag")

def answer_with_validation(user_query: str):
    print(f"\n--- Processing Query: '{user_query}' ---")
    
    # Step A: Local Retrieval using nomic-embed-text
    query_embedding = ollama.embed(model="nomic-embed-text", input=user_query)["embeddings"][0]
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    
    retrieved_docs = results["documents"][0] if results["documents"] else []
    
    if not retrieved_docs:
        print("❌ Retrieval Error: No documents found in database.")
        return "I couldn't find any relevant documents in your local database."

    print(f"Retrieved {len(retrieved_docs)} raw chunks from local storage.")

    valid_contexts = []
    
    for i, doc in enumerate(retrieved_docs):
        eval_prompt = f"""You are a strict relevance grader.
Determine if the provided Document Chunk contains ANY information or numbers that can help answer the Question.

Question: {user_query}
Document Chunk: {doc}

Does the document contain useful or related details? Answer strictly with ONLY 'YES' or 'NO'."""
        
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.0,
            max_tokens=10
        )
        
        verdict = response.choices[0].message.content.strip().lower()
        print(f"Chunk {i+1} relevance check: {verdict.upper()}")
        
        if "yes" in verdict or "true" in verdict:
            valid_contexts.append(doc)

    if not valid_contexts:
        print("⚠️ Warning: Retrieved documents failed the relevance gate. Triggering fallback response.")
        return "I found some documents in storage, but none of them appeared relevant enough to accurately answer your question."

    print("✅ Context verified. Generating final response...")
    context_block = "\n\n".join(valid_contexts)
    
    generation_prompt = f"""Answer the user's question concisely using ONLY the provided verified context below.

Context:
{context_block}

Question: {user_query}"""
    
    final_response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": generation_prompt}],
        temperature=0.0
    )
    
    return final_response.choices[0].message.content


if __name__ == "__main__":
    test_query = "What database does Project Beta use?"
    print(answer_with_validation(test_query))