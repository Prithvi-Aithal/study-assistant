import streamlit as st
import os
import pypdf
from dotenv import load_dotenv
from google import genai
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from typing import Any, Generator

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class GeminiLLM(CustomLLM):
    model_name: str = "gemini-2.5-flash"
    context_window: int = 32000
    num_output: int = 1024

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=self.context_window, num_output=self.num_output, model_name=self.model_name)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = client.models.generate_content(model=self.model_name, contents=prompt)
        return CompletionResponse(text=response.text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> Generator:
        response = client.models.generate_content(model=self.model_name, contents=prompt)
        yield CompletionResponse(text=response.text, delta=response.text)

st.set_page_config(page_title="Study Assistant", page_icon="📚")
st.title("📚 AI Study Assistant")
st.caption("Upload your notes or a textbook chapter and ask anything.")

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded_file:
        os.makedirs("data", exist_ok=True)
        save_path = os.path.join("data", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Loaded: {uploaded_file.name}")

@st.cache_resource(show_spinner="Building index from your document...")
def build_index(filename):
    Settings.llm = GeminiLLM()
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    save_path = os.path.join("data", filename)
    reader = pypdf.PdfReader(save_path)
    text = "\n".join(page.extract_text() for page in reader.pages)
    docs = [Document(text=text)]
    index = VectorStoreIndex.from_documents(docs)
    return index.as_query_engine(similarity_top_k=3)

if uploaded_file:
    query_engine = build_index(uploaded_file.name)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask something about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_engine.query(prompt)
                answer = str(response)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Upload a PDF from the sidebar to get started.")