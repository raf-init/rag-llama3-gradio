import chromadb
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def init_chroma():
    client = chromadb.Client()
    collection = client.get_or_create_collection("docs")
    return collection

def index_documents(docs, collection):
    for i, doc in enumerate(docs):
        embedding = embedding_model.encode(doc.page_content)
        collection.add(documents=[doc.page_content], embeddings=[embedding], ids=[f"doc-{i}"])
