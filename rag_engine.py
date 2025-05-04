import requests

def generate_answer_ollama(question, collection, embedding_model, top_k=5, model="llama3"):
    q_embedding = embedding_model.encode(question)
    results = collection.query(query_embeddings=[q_embedding], n_results=top_k)
    
    context = "\n\n".join(results['documents'][0])
    prompt = f"""Answer the following question using *only* the context below.

Context:
{context}

Question:
{question}

Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )

    if response.status_code == 200:
        return response.json()["response"].strip()
    else:
        return f"Error: {response.text}"
