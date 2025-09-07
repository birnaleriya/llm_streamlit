import os
from langchain_community.document_loaders import (
    PyPDFLoader, 
    DirectoryLoader, 
    TextLoader, 
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader  # <-- ADDED THIS LINE
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
DATA_PATH = "documents/"
DB_FAISS_PATH = "vectorstore/"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def create_vector_db():
    """
    Reads documents (including .pptx), processes them, and creates a vector store.
    """
    print("--- Starting Document Ingestion Process ---")
    
    # Set up loaders for all supported file types
    txt_loader = DirectoryLoader(DATA_PATH, glob='**/*.txt', loader_cls=TextLoader, show_progress=True)
    pdf_loader = DirectoryLoader(DATA_PATH, glob='**/*.pdf', loader_cls=PyPDFLoader, show_progress=True)
    docx_loader = DirectoryLoader(DATA_PATH, glob='**/*.docx', loader_cls=UnstructuredWordDocumentLoader, show_progress=True)
    pptx_loader = DirectoryLoader(DATA_PATH, glob='**/*.pptx', loader_cls=UnstructuredPowerPointLoader, show_progress=True) # <-- ADDED THIS LINE
    
    # Load all documents from all loaders
    documents = txt_loader.load() + pdf_loader.load() + docx_loader.load() + pptx_loader.load() # <-- ADDED PPTX LOADER HERE
    
    if not documents:
        print(f"No documents found in '{DATA_PATH}'. Please add your files to this directory.")
        return
        
    print(f"Successfully loaded {len(documents)} documents.")

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split the documents into {len(texts)} chunks.")

    # Initialize the embedding model
    print(f"Initializing embedding model: '{EMBEDDING_MODEL_NAME}'")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    # Create and save the FAISS vector store
    print("Creating and building the FAISS vector store...")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    
    print(f"--- Vector database created and saved successfully at '{DB_FAISS_PATH}' ---")

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created '{DATA_PATH}' directory. Please add your documents there before running again.")
        
    create_vector_db()

