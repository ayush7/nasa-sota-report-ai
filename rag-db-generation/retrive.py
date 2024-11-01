from rag import *


def create_qa_chain_openai(vectorstore, key):
    """Create a question-answering chain using the vector store."""
    # Initialize language model
    llm = ChatOpenAI(api_key=key,temperature=0)
    print("Querying the vector store...")
    
    # Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        return_source_documents=True
    )
    
    return qa_chain