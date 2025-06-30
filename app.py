import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
import os

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄 PDF RAG Chatbot")

# Hardcoded OpenAI API key
openai_api_key = "Enter my openai api key here"

# Sidebar for PDF upload only

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
pdf_process_button = st.sidebar.button("Process PDF")

# Session state to store vectorstore and texts
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'texts' not in st.session_state:
    st.session_state.texts = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

# PDF processing
if uploaded_file and pdf_process_button:
    def extract_text_from_pdf(pdf_file):
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    with st.spinner("Reading PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    with st.spinner("Embedding and indexing PDF..."):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts, embeddings)

    st.session_state.vectorstore = vectorstore
    st.session_state.texts = texts
    st.session_state.pdf_processed = True
    st.success("PDF processed successfully! Now you can ask questions below.")

# Chat interface
st.header("Ask questions about your PDF!")
user_query = st.text_input("Your question:")
submit_question = st.button("Submit Question")

if submit_question:
    if not st.session_state.pdf_processed or st.session_state.vectorstore is None:
        st.warning("Please upload and process a PDF first.")
    elif not user_query.strip():
        st.warning("Please enter a question.")
    else:
        # Get relevant documents with scores
        docs_and_scores = st.session_state.vectorstore.similarity_search_with_score(user_query, k=3)
        
        # Create QA chain
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(),
            return_source_documents=True,
        )
        
        with st.spinner("Generating answer..."):
            result = qa_chain({"query": user_query})
            st.markdown(f"**Answer:** {result['result']}")
            
            with st.expander("Show source chunks with relevance scores"):
                for i, (doc, score) in enumerate(docs_and_scores):
                    # Convert similarity score to percentage (higher is better)
                    relevance_percentage = (1 - score) * 100
                    st.markdown(f"**Chunk {i+1}** (Relevance: {relevance_percentage:.2f}%):\n{doc.page_content}\n---")
                    
        
        