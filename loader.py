from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

def load_documents(folder="data"):
    docs = []
    for file in Path(folder).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs

def chunk_documents(docs, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents(docs)
