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
        max_length=300,  # Increased for better summaries
        min_length=100,
        device=-1,  # Force CPU usage
        torch_dtype=torch.float32  # Use float32 for stability
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
        
        # Extract text from more pages (increased from 5 to 10)
        max_pages = 10  # Increased page limit
        full_text = ""
        for i, page in enumerate(pages[:max_pages]):
            full_text += page.page_content + "\n"
        
        return full_text
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def chunk_text_for_summarization(text, chunk_size=800):
    """Split text into chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100,
        length_function=len
    )
    
    # Split the text
    chunks = text_splitter.split_text(text)
    return chunks

def llm_pipeline(uploaded_file):
    # Load model
    summarizer = load_summarization_model()
    
    # Get text from PDF
    input_text = file_preprocessing(uploaded_file)
    
    # Show text statistics
    st.info(f"📊 Document Statistics: {len(input_text)} characters, approx {len(input_text.split())} words")
    
    # Increased character limit significantly
    if len(input_text) > 8000:
        st.warning("Document is large. Processing first 8000 characters for optimal performance...")
        input_text = input_text[:8000]
    elif len(input_text) > 4000:
        st.info("Processing document with 4000-8000 characters...")
    else:
        st.success(f"Processing complete document ({len(input_text)} characters)")
    
    try:
        # For longer texts, use chunking strategy
        if len(input_text) > 2000:
            chunks = chunk_text_for_summarization(input_text)
            st.write(f"📑 Processing document in {len(chunks)} chunks...")
            
            summaries = []
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 100:  # Only process substantial chunks
                    with st.spinner(f"Processing chunk {i+1}/{len(chunks)}..."):
                        chunk_summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                        summaries.append(chunk_summary[0]['summary_text'])
            
            # Combine summaries
            if summaries:
                combined_summary = " ".join(summaries)
                # Final summary of combined chunks
                if len(combined_summary) > 1000:
                    final_summary = summarizer(combined_summary[:1000], max_length=250, min_length=100, do_sample=False)
                    return final_summary[0]['summary_text']
                return combined_summary
            else:
                return "No substantial content found to summarize."
        else:
            # Direct summarization for shorter texts
            summary = summarizer(input_text, max_length=250, min_length=100, do_sample=False)
            return summary[0]['summary_text']
            
    except Exception as e:
        # Fallback: return meaningful excerpt
        st.error(f"Summarization failed: {str(e)}")
        # Return a better excerpt
        sentences = input_text.split('.')
        meaningful_excerpt = '. '.join(sentences[:5]) + '.' if len(sentences) > 5 else input_text
        return f"Text extraction (AI summarization failed): {meaningful_excerpt}"

@st.cache_data
def display_pdf(uploaded_file):
    base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
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
                with st.spinner("Processing document... This may take a minute for larger documents."):
                    try:
                        uploaded_file.seek(0)
                        summary = llm_pipeline(uploaded_file)
                        st.success("✅ Summary generated successfully!")
                        
                        # Display summary with better formatting
                        st.markdown("### 📋 Summary")
                        st.write(summary)
                        
                        # Show summary statistics
                        st.metric("Summary Length", f"{len(summary)} characters")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("💡 Tip: Try with a smaller document or check the console for detailed error messages.")

if __name__ == "__main__":  
    main()