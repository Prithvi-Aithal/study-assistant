# AI Study Assistant 📚

A retrieval-augmented generation (RAG) chatbot for your study material. Upload a PDF (lecture notes, a textbook chapter, a paper), ask questions in plain English, and get answers grounded in the document, along with the pages they came from.

**Live demo:** [prithvi-aithal-study-assistant.streamlit.app](https://prithvi-aithal-study-assistant.streamlit.app)

## Features

- Upload any text-based PDF and ask questions about it.
- Every answer cites its source pages, for example `📄 Sources: p.3, p.7`.
- Embeddings run locally, so there are no embedding API calls or costs.
- The index is built once per document and cached for the session.
- The conversation stays on screen for the whole session.

## How it works

```text
PDF ──► pypdf: one Document per page, page number kept as metadata
    ──► FastEmbed (BAAI/bge-small-en-v1.5) embeddings ──► LlamaIndex VectorStoreIndex

question ──► top-3 most similar pages ──► Gemini 2.5 Flash ──► answer + page citations
```

- **Page-level chunks** make the citations exact: each retrieved chunk is one page, so its metadata carries the page number.
- **Custom LLM wrapper:** `GeminiLLM` subclasses LlamaIndex's `CustomLLM` and calls Gemini through the `google-genai` SDK.
- **Caching:** `st.cache_resource` keeps the built index, so each later question costs one retrieval and one LLM call.

## Tech stack

| Part | Tool |
|---|---|
| LLM | Google Gemini 2.5 Flash (`google-genai`) |
| RAG framework | LlamaIndex |
| Embeddings | BAAI/bge-small-en-v1.5 via FastEmbed, run locally |
| PDF parsing | pypdf |
| UI and hosting | Streamlit, deployed on Streamlit Community Cloud |

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your_key_here" > .env   # a Gemini API key from Google AI Studio
streamlit run app.py
```

`packages.txt` lists the system package (`libgomp1`) that the Streamlit Community Cloud deployment needs.

## Limitations

- Scanned PDFs without a text layer can't be read, since there is no OCR step.
- Each question is answered on its own: earlier messages stay on screen but aren't sent as context.
