import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "true"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# Page config
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = None
if 'qa_model' not in st.session_state:
    st.session_state.qa_model = None

# to extract text from pdf file using pdfReader from pypdf2
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# splitting Extracted text into small small chunks with operlaping text
def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# feed chunks to model to encode and return embeddings 
def create_embeddings(chunks, model):
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

# Index embeddings into FAISS local index
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

# Search for similar chunks using FAISS
def search_similar_chunks(query, model, index, chunks, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), k)
    return [chunks[i] for i in indices[0]]

# Generate answer using Open Source Model google/flan-t5-base
def generate_answer(question, context, qa_model):
    
    max_context_length = 2000 # Combine context (limit to avoid token limits)
    if len(context) > max_context_length:
        context = context[:max_context_length]
    
    input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    # Generate answer
    result = qa_model(input_text, max_length=200, min_length=20, do_sample=False)
    answer = result[0]['generated_text']
    
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    
    return answer

# Main User Interface webpage
st.title("📚 PDF-Based RAG Chatbot")
st.markdown("Upload two PDF documents and ask questions about their content!")
st.markdown("**100% Free** - Uses open-source models from Hugging Face")

with st.sidebar:
    st.header("📄 Upload PDFs")
    pdf1 = st.file_uploader("Upload PDF 1", type=['pdf'], key="pdf1")
    pdf2 = st.file_uploader("Upload PDF 2", type=['pdf'], key="pdf2")
    
    st.markdown("---")
    
    if st.button("🔄 Process PDFs", type="primary"):
        if not pdf1 or not pdf2:
            st.error("Please upload both PDF files!")
        else:
            with st.spinner("Processing PDFs... This may take a minute on first run."):
                try:
                    # Extract text from both PDFs
                    st.info("📖 Reading PDFs...")
                    text1 = extract_text_from_pdf(pdf1)
                    text2 = extract_text_from_pdf(pdf2)
                    combined_text = text1 + "\n\n" + text2
                    
                    # Split into chunks
                    st.info("✂️ Splitting text into chunks...")
                    chunks = split_text_into_chunks(combined_text)
                    st.session_state.chunks = chunks
                    
                    # Load embedding model
                    if st.session_state.embeddings_model is None:
                        st.info("🔧 Loading embedding model...")
                        st.session_state.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    # Create embeddings
                    st.info("🔍 Creating embeddings...")
                    embeddings = create_embeddings(chunks, st.session_state.embeddings_model)
                    
                    # Create FAISS index
                    st.info("📊 Building search index...")
                    st.session_state.index = create_faiss_index(embeddings)
                    
                    # Load QA model
                    if st.session_state.qa_model is None:
                        st.info("🤖 Loading question-answering model...")
                        st.session_state.qa_model = pipeline(
                            "text2text-generation",
                            model="google/flan-t5-base"
                        )
                    
                    st.session_state.processed = True
                    st.success(f"✅ Successfully processed {len(chunks)} chunks from both PDFs!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    if st.session_state.processed:
        st.success("✅ PDFs are ready!")
        st.info(f"📦 Total chunks: {len(st.session_state.chunks)}")
    
    st.markdown("---")
    st.markdown("""
    ### 🛠️ Tech Stack:
    - **Streamlit**: UI
    - **PyPDF2**: PDF reading
    - **Sentence Transformers**: Embeddings
    - **FAISS**: Vector search
    - **google/flan-t5-base**: Answer generation
    
    All models run locally - no API keys needed!
    """)

# Main content area
if st.session_state.processed:
    st.markdown("### 💬 Ask Questions")
    
    question = st.text_input(
        "Enter your question:",
        placeholder="What are the main topics in these documents?"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🔍 Get Answer", type="primary")
    
    if ask_button:
        if not question:
            st.warning("Please enter a question!")
        else:
            with st.spinner("Searching documents and generating answer..."):
                try:
                    # Search for relevant chunks
                    relevant_chunks = search_similar_chunks(
                        question,
                        st.session_state.embeddings_model,
                        st.session_state.index,
                        st.session_state.chunks,
                        k=3
                    )
                    
                    # Combine chunks as context
                    context = "\n\n".join(relevant_chunks)
                    
                    # Generate answer
                    answer = generate_answer(question, context, st.session_state.qa_model)
                    
                    # Display answer
                    st.markdown("### 📝 Answer:")
                    st.success(answer)
                    
                    # Show relevant chunks
                    with st.expander("📄 View source text chunks"):
                        for i, chunk in enumerate(relevant_chunks, 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(chunk[:400] + "..." if len(chunk) > 400 else chunk)
                            if i < len(relevant_chunks):
                                st.markdown("---")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
else:
    st.info("👈 Please upload two PDFs and click 'Process PDFs' to get started!")
    
    st.markdown("""
    ### 📖 How to Use:
    
    1. **Upload PDFs**: Upload two PDF documents in the sidebar <- add as much as you want
    2. **Process**: Click "Process PDFs" button (takes ~30 seconds first time because it needs to do multiple process)
    3. **Ask Questions**: Type your question and click "Get Answer"
    4. **View Sources**: Expand to see which text chunks were used
    
    ### 💡 Example Questions:
    - What are the main topics in these documents?
    - Summarize the key findings
    - What does the document say about [specific topic]?
    - List the important points mentioned
    
    ### ✨ Features:
    - ✅ 2 document processing at a time concurently
    - ✅ FAISS local searching for retrival of similar chunks
    - ✅ Open source - Uses Hugging Face models
    - ✅ Fast search - FAISS vector similarity
    """)

# Footer
st.markdown("---")
st.markdown("Built for Algorizz for Interview round using Streamlit, Sentence Transformers, FAISS, and FLAN-T5 model")
