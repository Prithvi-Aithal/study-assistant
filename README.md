# AI Study Assistant 📚

A RAG-based chatbot that lets you upload any PDF and ask questions about it. Powered by Google Gemini and LlamaIndex.

## Features
- Upload any PDF (notes, textbook chapters)
- Ask questions in natural language
- Answers grounded in your document
- Chat history within session

## Tech Stack
- **LLM**: Google Gemini 2.5 Flash
- **RAG**: LlamaIndex
- **Embeddings**: BAAI/bge-small-en-v1.5
- **UI**: Streamlit

## Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Create a .env file and add your Gemini API key:
GOOGLE_API_KEY=your_key_here

## Run
streamlit run app.py
