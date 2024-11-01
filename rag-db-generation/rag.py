import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate  # Added this import


from dotenv import load_dotenv



## Load environment variables (for OpenAI API key)
load_dotenv()

def process_pdfs(pdf_directory):
    print("Processing PDFs...")
    """Process all PDFs in the specified directory and create a vector store."""
    documents = []
    
    # Load all PDFs from the directory
    for file in os.listdir(pdf_directory):
        if file.endswith('.pdf'):
            print(f"Processing {file}...")
            pdf_path = os.path.join(pdf_directory, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Create and persist vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vectorstore

