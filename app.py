import gradio as gr
from loader import load_documents, chunk_documents
from embedder import init_chroma, index_documents, embedding_model
from rag_engine import generate_answer_ollama

# Step 1: Load & Embed
raw_docs = load_documents()
chunks = chunk_documents(raw_docs)
collection = init_chroma()
index_documents(chunks, collection)

# Step 2: Gradio Interface
def rag_interface(question):
    return generate_answer_ollama(question, collection, embedding_model)

gr.Interface(
    fn=rag_interface,
    inputs="text",
    outputs="text",
    title="Local RAG with LLaMA 3 + Ollama",
    description="Ask questions based on your internal documents."
).launch()
