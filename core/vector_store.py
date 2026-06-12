import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcripts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

def build_vector_store(transcript: str) -> Chroma:
    """
    Build a Chroma vector store from the transcript.
    """
    print("Building vector store...")

    # Split the transcript into smaller pieces for better embedding.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(
            page_content=chunk,
            metadata={'chunk_index' : i}
        ) for i, chunk in enumerate(chunks)     
    ]
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )
    return vector_store

def load_vector_store() -> Chroma:
    """
    Load the Chroma vector store from disk.
    """
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vector_store

def get_retriever(vector_store: Chroma, k: int = 4):
    """
    Get a retriever from the vector store.
    """
    return vector_store.as_retriever(search_kwargs={"k": k},search_type="similarity")
