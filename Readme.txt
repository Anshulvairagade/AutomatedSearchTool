Scheme Research Tool
This is a Streamlit-based app that allows users to input URLs of articles related to schemes, process them, and ask questions to retrieve relevant information from the indexed content.

Features
Input multiple URLs to process and extract content.
The content is split into chunks and indexed using FAISS with sentence-transformer embeddings.
Ask questions based on the scheme using a Hugging Face LLM (FLAN-T5-large).
View the full content of the processed URLs.
Historical Q&A view for all queries.

Setup Instructions
1. Clone the repository:
git clone https://github.com/Anshulvairagade/AutomatedSearchTool.git

2. Install the required packages:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py

Dependencies
Streamlit
FAISS
Hugging Face Transformers
Sentence-Transformers
LangChai

Usage
Add URLs in the sidebar to process.
Click on Process URLs to load and index the content.
Ask questions related to the content in the query box and get answers.
View article content and Q&A history.