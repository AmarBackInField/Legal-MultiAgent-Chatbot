import os
import logging
import sys
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GOOGLE_API_KEY
from document_loader import load_documents

# Logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('vector_db.log')
    ]
)
logger = logging.getLogger(__name__)

# Set default headers for external requests
headers = {
    "User-Agent": os.environ.get(
        "USER_AGENT", 
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

def initialize_embeddings():
    """
    Initialize Google Generative AI Embeddings with error handling.
    
    Returns:
        GoogleGenerativeAIEmbeddings: Configured embedding model
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        raise

def vectordb(docs_list=None):
    """
    Create or load a vector database using Chroma.
    
    Args:
        docs_list (list, optional): List of documents to index. Defaults to None.
    
    Returns:
        Chroma retriever object
    """
    # Ensure persistent directory exists
    persist_directory = "./chroma_db"
    os.makedirs(persist_directory, exist_ok=True)
    
    # Initialize embeddings
    try:
        embeddings = initialize_embeddings()
    except Exception as e:
        logger.error("Embedding initialization failed")
        raise

    # Prepare documents if not provided
    if docs_list is None:
        try:
            docs_list = load_documents()
            logger.info("Documents loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise

    # Collection Configuration
    collection_name = "icl-docs"
    
    try:
        # Use PersistentClient for reliable storage
        client = chromadb.PersistentClient(path=persist_directory)
        
        # Check if collection exists
        try:
            client.get_collection(name=collection_name)
            logger.info(f"Collection {collection_name} already exists")
            
            # Load existing vectorstore
            vectorstore = Chroma(
                collection_name=collection_name,
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
        except Exception:
            # Create new collection if it doesn't exist
            logger.info(f"Creating new collection: {collection_name}")
            vectorstore = Chroma.from_documents(
                documents=docs_list,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=persist_directory
            )

        # Configure retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
        )
        return retriever

    except Exception as e:
        logger.error(f"Vector database initialization failed: {e}")
        raise
