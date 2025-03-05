from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents():
    """Load and split legal documents"""
    loaders = [
        PyPDFLoader("Data/Guide-to-Litigation-in-India.pdf"),
        PyPDFLoader("Data/PDFFile5b28c9ce64e524.54675199.pdf")
    ]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    # Text Splitting
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200,
        chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs)
    return doc_splits
