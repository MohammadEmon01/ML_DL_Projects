#Load Important Libraries 
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import torch
import base64 
import tempfile
import os

# Import only what we need
from transformers import pipeline

# Use a smaller model or different approach
@st.cache_resource
def load_summarization_model():
    """Load the model with caching to avoid reloading"""
    return pipeline(
        "summarization",
        model="MBZUAI/LaMini-Flan-T5-248M",
        max_length=150,
        min_length=30,
        device=-1  # Force CPU usage
    )

def file_preprocessing(uploaded_file):
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load PDF and extract text more efficiently
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        
        # Extract only first few pages for large documents
        max_pages = 5  # Limit to first 5 pages
        full_text = ""
        for i, page in enumerate(pages[:max_pages]):
            full_text += page.page_content + "\n"
        
        return full_text
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def llm_pipeline(uploaded_file):
    # Load model
    summarizer = load_summarization_model()
    
    # Get text from PDF
    input_text = file_preprocessing(uploaded_file)
    
    # Limit text size more aggressively
    if len(input_text) > 2000:
        st.warning("Document is large. Processing first 2000 characters...")
        input_text = input_text[:2000]
    
    try:
        # Summarize in one go with smaller input
        summary = summarizer(input_text, max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        # Fallback: return first part of text
        st.error(f"Summarization failed: {str(e)}")
        return input_text[:500] + "..." if len(input_text) > 500 else input_text

@st.cache_data
def display_pdf(uploaded_file):
    base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit App
st.set_page_config(page_title="Document Summarizer", layout="wide")

def main():
    st.title("📄 Document Summarizer")
    st.write("Upload a PDF document to generate its summary.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📖 Uploaded Document")
            display_pdf(uploaded_file)
        
        with col2:
            st.subheader("📝 Document Summary")
            if st.button("🚀 Generate Summary", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        uploaded_file.seek(0)
                        summary = llm_pipeline(uploaded_file)
                        st.success("Summary generated!")
                        st.write(summary)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

if __name__ == "__main__":  
    main()