from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GOOGLE_API_KEY
from document_loader import load_documents
import os
import logging
import sys

import chromadb.api

chromadb.api.client.SharedSystemClient.clear_system_cache()
headers = {
    "User-Agent": os.environ.get("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# chromadb.api.client.SharedSystemClient.clear_system_cache()

# Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

docs_list = load_documents()
logging.info("Documents loaded successfully")

def vectordb():
    # Check if the collection already exists
    client = chromadb.Client()
    collection_name = "icl-docs"
    persist_directory = "./chroma_db"
    
    if collection_name in chromadb.PersistentClient(path=persist_directory).list_collections():
        logger.info(f"Loading existing collection: {collection_name}")
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        logger.info(f"Creating new collection: {collection_name}")
        vectorstore = Chroma.from_documents(
            documents=docs_list,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory  # Ensure Chroma is stored locally
        )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
    )
    return retriever

# Example usage
retriever = vectordb()
