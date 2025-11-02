#Load Important Libraries 
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer , T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64 
# !pip install accelerate
# !pip install hf_xet
# Model and Tokenizer
checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float16)

def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    text = text_splitter.split_documents(pages)
    final_text = ""
    for doc in text:
        print(doc)
        final_text += doc.page_content + "\n"
    return final_text

#define llm pipeline
def llm_pipeline(filepath):

    pipeline_summarization = pipeline(
        "summarization", 
        model=model, 
        tokenizer=tokenizer, 
        max_length=500,  # Fixed typo
        min_length=50
    ) 
    
    input_text = file_preprocessing(filepath)
    summary = pipeline_summarization(input_text)
    summary = summary[0]['summary_text']  # Fixed: should be summary[0], not input_text[0]
    return summary

# Now you can call the function with an actual filepath
# Example usage:
# result = llm_pipeline("your_file.pdf")
# print(result)
@st.cache_data
def display_pdf(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)
    #Streamlit App Code
st.set_page_config(page_title="Document Summarizer")
def main():
    st.title("Document Summarizer using LLM")
    st.write("Upload a PDF document to generate its summary.")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Generate Summary"):
            col1, col2 = st.columns(2)
            with col1:
                st.header("Uploaded Document")
                display_pdf(uploaded_file)
            with col2:
                st.header("Document Summary")
                summary = llm_pipeline(uploaded_file)
                st.write(summary)
    
if __name__ == "__main__":  
    main()
    
            
            

