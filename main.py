import streamlit as st
import os
import pickle
import tempfile
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'full_text' not in st.session_state:
    st.session_state.full_text = {}

# Streamlit app
st.title("Scheme Research Tool")

# Sidebar for URL input
st.sidebar.header("Input URLs")
urls = st.sidebar.text_area("Enter URLs (one per line)", height=150)
process_button = st.sidebar.button("Process URLs")

@st.cache_resource
def load_llm():
    # Load a larger model for more descriptive answers
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Increase max_length for longer outputs
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=500,  # Increased from default
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    
    return HuggingFacePipeline(pipeline=pipe)

# Main content area
if process_button:
    if urls:
        urls = urls.split('\n')
        # Load data from URLs
        with st.spinner("Loading and processing URLs..."):
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()
        
        # Store full text content
        for doc in data:
            st.session_state.full_text[doc.metadata['source']] = doc.page_content
        
        # Split text into chunks
        with st.spinner("Splitting text into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            docs = text_splitter.split_documents(data)
        
        # Generate embeddings and create FAISS index
        with st.spinner("Generating embeddings and creating FAISS index..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)
        
        # Save FAISS index
        with open("faiss_store.pkl", "wb") as f:
            pickle.dump(vectorstore, f)
        
        st.session_state.processed = True
        st.success("URLs processed and indexed successfully!")
    else:
        st.warning("Please enter at least one URL.")

# Display full text content of articles
if st.session_state.processed:
    st.header("Article Contents")
    for url, content in st.session_state.full_text.items():
        with st.expander(f"Article: {url}"):
            st.write(content)

# Load the saved FAISS index if it exists
if os.path.exists("faiss_store.pkl"):
    with open("faiss_store.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    
    # Create a retrieval chain
    llm = load_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # Retrieve more documents for context
    )
    
    # User query input
    query = st.text_input("Ask a question about the scheme:")
    if query:
        with st.spinner("Generating detailed answer..."):
            # Formulate a more descriptive prompt
            detailed_query = f"""Provide a detailed and comprehensive answer to the following question about the scheme. 
            Include relevant facts, explanations, and examples if applicable. 
            If there are multiple aspects to the answer, break it down into clear points.
            
            Question: {query}
            
            Detailed answer:"""
            
            result = qa_chain({"query": detailed_query})
        
        # Display answer
        st.header("Detailed Answer")
        st.write(result["result"])
        
        # Generate and display summary
        with st.spinner("Generating comprehensive summary..."):
            summary_prompt = f"""Based on the articles, provide a detailed summary covering the following key criteria:
            1. Scheme Benefits: Explain the main advantages and positive outcomes for participants.
            2. Scheme Application Process: Describe the step-by-step procedure for applying, including any deadlines or special requirements.
            3. Eligibility: Clearly outline who can apply, including any age, income, or other restrictions.
            4. Documents Required: List and explain all necessary documentation for the application.

            Please provide a comprehensive explanation for each point, using information from all relevant articles.

            Detailed summary:"""
            
            summary = qa_chain({"query": summary_prompt})
        
        st.header("Comprehensive Summary")
        st.write(summary["result"])
else:
    st.info("Please process URLs to start asking questions.")