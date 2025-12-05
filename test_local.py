"""
Simple test script to verify the RAG pipeline works
No API keys needed - uses free Hugging Face models
"""

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

def test_rag_pipeline(pdf_path1, pdf_path2, test_question):
    """Test the RAG pipeline with two PDFs"""
    
    print("📄 Reading PDFs...")
    # Read PDF 1
    pdf1 = PdfReader(pdf_path1)
    text1 = ""
    for page in pdf1.pages:
        text1 += page.extract_text()
    print(f"  PDF 1: {len(pdf1.pages)} pages, {len(text1)} characters")
    
    # Read PDF 2
    pdf2 = PdfReader(pdf_path2)
    text2 = ""
    for page in pdf2.pages:
        text2 += page.extract_text()
    print(f"  PDF 2: {len(pdf2.pages)} pages, {len(text2)} characters")
    
    # Combine texts
    combined_text = text1 + "\n\n" + text2
    
    print("\n✂️  Splitting into chunks...")
    # Split into chunks
    chunks = []
    chunk_size = 1000
    overlap = 200
    start = 0
    while start < len(combined_text):
        end = start + chunk_size
        chunk = combined_text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    print(f"  Created {len(chunks)} chunks")
    
    print("\n🔧 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("🔍 Creating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print(f"  Embeddings shape: {embeddings.shape}")
    
    print("📊 Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    print(f"  Index contains {index.ntotal} vectors")
    
    print(f"\n❓ Question: {test_question}")
    print("🔎 Finding relevant chunks...")
    
    # Search for relevant chunks
    query_embedding = model.encode([test_question])
    distances, indices = index.search(query_embedding.astype('float32'), 3)
    
    relevant_chunks = [chunks[i] for i in indices[0]]
    print(f"  Found {len(relevant_chunks)} relevant chunks")
    
    print("\n🤖 Loading QA model (FLAN-T5)...")
    qa_model = pipeline("text2text-generation", model="google/flan-t5-base")
    
    print("💭 Generating answer...")
    context = "\n\n".join(relevant_chunks)
    
    # Limit context length
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length]
    
    input_text = f"Context: {context}\n\nQuestion: {test_question}\n\nAnswer:"
    result = qa_model(input_text, max_length=200, min_length=20, do_sample=False)
    answer = result[0]['generated_text']
    
    # Extract only the answer part
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    
    print(f"\n✅ Answer:\n{answer}")
    
    print("\n📄 Source chunks used:")
    for i, chunk in enumerate(relevant_chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    
    return answer

if __name__ == "__main__":
    print("=" * 60)
    print("PDF RAG Chatbot - Test Script")
    print("100% Free - No API keys needed!")
    print("=" * 60)
    
    pdf1 = input("\nEnter path to first PDF: ")
    pdf2 = input("Enter path to second PDF: ")
    question = input("Enter your question: ")
    
    print("\n" + "=" * 60)
    
    try:
        test_rag_pipeline(pdf1, pdf2, question)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
