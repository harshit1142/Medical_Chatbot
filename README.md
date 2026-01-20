# Medical Chatbot with RAG Architecture

A medical question-answering system that leverages Retrieval-Augmented Generation (RAG) to provide accurate and context-aware medical information. This application combines the power of large language models with vector search to deliver precise answers to medical queries.

## 🚀 Key Features

- **Advanced RAG Pipeline**: Implements a robust Retrieval-Augmented Generation system for accurate medical information retrieval
- **Multi-document Support**: Processes and indexes multiple PDF documents from the `data/` directory
- **Semantic Search**: Utilizes Pinecone's vector database for efficient similarity search across medical documents
- **State-of-the-Art LLM**: Powered by Google's Gemini model through LangChain for high-quality response generation
- **Web Interface**: User-friendly Flask-based web interface for seamless interaction
- **Scalable Architecture**: Designed for easy extension and integration with additional data sources
- **Customizable Prompts**: Easily adjustable system prompts to tailor responses to medical domain requirements
- **Efficient Chunking**: Smart text splitting to maintain context while processing large documents

## 📊 Technical Architecture

The application follows a modern microservices architecture with the following components:

1. **Frontend**: Lightweight HTML/JS interface with responsive design
2. **Backend**: Flask web server handling API requests
3. **Vector Database**: Pinecone for efficient vector similarity search
4. **Embedding Model**: `all-MiniLM-L6-v2` for creating document embeddings
5. **LLM Integration**: Google's Gemini model for generating human-like responses
6. **Document Processing**: Automated pipeline for PDF ingestion and text extraction

### Project Structure

- `app.py`: Flask app and RAG pipeline
- `store_index.py`: Builds Pinecone index from PDFs in `data/`
- `src/helper.py`: Load PDF(s), split text, and create embeddings
- `src/prompt.py`: System prompt for the assistant
- `templates/chat.html`: Frontend chat page
- `static/style.css`: Simple styles

### Prerequisites

- Python 3.10+
- A Pinecone account and API key
- A Google AI Studio API key (for Gemini)

### 1) Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2) Environment Variables

Create a `.env` file in the project root:

```ini
PINECONE_API_KEY="your_pinecone_api_key"
GOOGLE_API_KEY="your_google_api_key"
```

Notes:
- The code uses the Pinecone index name `medical-catboot` and expects embeddings of dimension 384 (`all-MiniLM-L6-v2`).
- Default serverless spec: cloud `aws`, region `us-east-1`.

### 3) Add/Update Your Data

Place your PDFs in the `data/` folder (the repo includes `data/Medical_book.pdf`). Rebuild the index after you change files.

### 4) Build/Refresh the Pinecone Index

```powershell
.\.venv\Scripts\python store_index.py
```

This will:
- Read PDFs from `data/`
- Split into chunks
- Create sentence-transformer embeddings (384 dims)
- Create or reuse Pinecone index `medical-catboot`
- Upsert embeddings

### 5) Run the App

```powershell
.\.venv\Scripts\python app.py
```

Open `http://localhost:8080` in your browser.

### Troubleshooting

- Ensure `.env` contains valid `PINECONE_API_KEY` and `GOOGLE_API_KEY`.
- If you change PDFs, rerun `store_index.py` to refresh embeddings.
- If the index doesn’t exist, the script will create it (serverless `us-east-1`).
- On corporate networks, set proxy env vars for `pip`/downloads if needed.

### Tech Stack

- Python, Flask
- LangChain (Retrieval chain)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Pinecone (Vector DB)
- Google Gemini (via `langchain-google-genai`)
