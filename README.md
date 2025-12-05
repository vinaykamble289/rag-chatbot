# PDF-Based RAG Chatbot

A simple, **100% free** Retrieval-Augmented Generation (RAG) chatbot that answers questions from PDF documents. No API keys required!

## 🔗 Links

- **Live Demo**: [[Deploy to get your link]](https://huggingface.co/spaces/vinaykamble289/rag-chatbot)
- **GitHub**: [[Your repository link]](https://github.com/vinaykamble289/rag-chatbot)

> After deployment, update these links with your actual URLs!

## ✨ Features

- ✅ Upload any two PDF documents
- ✅ Ask questions about the content
- ✅ 100% Free - No API keys needed
- ✅ Privacy-friendly - Everything runs locally
- ✅ Uses open-source Hugging Face models
- ✅ Fast vector search with FAISS

## 🚀 How to Use

### Online (Hugging Face Spaces)

1. Visit the deployed app
2. Upload two PDF files
3. Click "Process PDFs" (takes ~30 seconds first time)
4. Ask questions about the documents!

### Local Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
```bash
python setup.py
```
Or if you prefer:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

4. Open your browser to http://localhost:8501

**Note**: If you encounter dependency errors, see [INSTALLATION.md](INSTALLATION.md) for troubleshooting.

## 🛠️ How It Works

1. **PDF Reading**: Extract text from PDFs using PyPDF2
2. **Text Chunking**: Split documents into 1000-character chunks with 200 overlap
3. **Embeddings**: Convert chunks to vectors using Sentence Transformers
4. **Vector Search**: Store in FAISS index for fast similarity search
5. **Question Answering**: 
   - Your question is converted to a vector
   - Top 3 most similar chunks are retrieved
   - FLAN-T5 generates an answer from the context

## 💻 Tech Stack

- **Streamlit**: Simple, clean web interface
- **PyPDF2**: PDF text extraction
- **Sentence Transformers**: Text embeddings (all-MiniLM-L6-v2)
- **FAISS**: Fast vector similarity search
- **FLAN-T5**: Answer generation (google/flan-t5-base)

All models are free and open-source from Hugging Face!

## 📦 Deployment to Hugging Face Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **"Streamlit"** as the SDK
3. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `README.md`
4. The Space will automatically build and deploy!

## 💡 Example Questions

- What are the main topics in these documents?
- Summarize the key findings
- What does the document say about [specific topic]?
- List the important points mentioned

## 📚 Documentation

- **QUICKSTART.md** - Get started in 5 minutes
- **DEPLOYMENT_CHECKLIST.md** - Step-by-step deployment guide
- **test_local.py** - Test the pipeline without UI

## 🎯 Why This Stack?

- **Streamlit**: Much simpler than Gradio, easy to understand
- **PyPDF2**: Straightforward PDF reading
- **No API Keys**: Everything runs locally, completely free
- **Fast**: FAISS provides instant search results
- **Open Source**: All models from Hugging Face

## License

MIT
