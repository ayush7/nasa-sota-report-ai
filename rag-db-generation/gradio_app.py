import gradio as gr

from retrive import create_qa_chain_openai
from rag import process_pdfs
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import key
OPENAI_API_KEY = key.api_key
# from dotenv import load_dotenv


# Initialize embeddings and load the existing vectorstore
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Initialize the QA chain
qa_chain = create_qa_chain_openai(vectorstore, OPENAI_API_KEY)

def process_question(question):
    """Process the user's question and return the answer"""
    result = qa_chain({"query": question})
    
    # Extract answer and sources
    answer = result['result']
    sources = [ f"- {doc.metadata['source']}, Page {doc.metadata['page']}"+ "..." for doc in result['source_documents']]
    
    return answer, "\n\nSources:\n" + "\n\n".join(sources)



# f"- {doc.metadata['source']}, Page {doc.metadata['page']}"