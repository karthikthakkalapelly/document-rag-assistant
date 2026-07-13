import os
import sys
import streamlit as st
from collections import defaultdict
from dotenv import load_dotenv
from pathlib import Path

# Load dotenv from `src/.env` if present (no debug prints)
dotenv_path = Path(__file__).resolve().parent / "src" / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.getcwd()))

CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #0b1726 0%, #121a34 45%, #151e3d 100%);
        color: #ffffff;
    }
    .stButton>button {
        background-color: #5b8def;
        color: white;
        border-radius: 12px;
        padding: 0.75rem 1.2rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #4177f2;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 12px;
        border: 1px solid #65758c;
        background: rgba(255,255,255,0.06);
        color: #ffffff;
    }
    .stFileUploader {
        border: 1px solid #4f6a9c;
        border-radius: 16px;
        background: rgba(255,255,255,0.05);
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #ffffff;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def import_rag_pipeline():
    try:
        from src.rag_pipeline import RAGPipeline
        return RAGPipeline, None
    except Exception as exc:
        return None, exc

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=api_key,
    )

@st.cache_resource(show_spinner=False)
def get_llm():
    from src.llm import load_llm
    return load_llm()

@st.cache_resource(show_spinner=False)
def get_vector_store(database_path):
    from src.retriever import load_vector_store

    return load_vector_store(database_path, embedding_model=get_embedding_model())

RAGPipeline, RAG_PIPELINE_IMPORT_ERROR = import_rag_pipeline()
if RAG_PIPELINE_IMPORT_ERROR is not None:
    st.error("Backend failed to load. Please check server logs.")
    st.exception(RAG_PIPELINE_IMPORT_ERROR)
    st.stop()

# Ensure any existing pipeline in session state is an instance of the currently
# imported RAGPipeline class. This avoids calling outdated methods on a
# stale pipeline object after code changes (hot-reload / session persistence).
if "pipeline" not in st.session_state or not isinstance(
    st.session_state.get("pipeline"), RAGPipeline
):
    try:
        st.session_state.pipeline = RAGPipeline()
    except Exception as e:
        st.error("Failed to initialize pipeline.")
        st.exception(e)
        st.stop()

# Page configuration
st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="📄",
    layout="wide",
)

st.markdown("""
    <div style='padding: 0 0.2rem;'>
        <h1 style='margin-bottom:0.15rem;'>📄 Document RAG Assistant</h1>
        <p style='font-size:1.05rem; color:#cbd5e1; line-height:1.6;'>
            Build a private document intelligence workspace that handles PDFs, OCR, and smart search without loading everything at startup.
            Designed for enterprise-grade self-service workflows.
        </p>
    </div>
""", unsafe_allow_html=True)

with st.expander("How it works", expanded=True):
    st.write(
        "1. Upload PDF documents to build your knowledge base.\n"
        "2. The app analyzes content with Gemini embeddings and Chroma retrieval.\n"
        "3. Ask natural language questions and get concise answers with sources.\n"
        "4. OCR is applied only when scanned PDF content is detected, and fallback is graceful."
    )

st.markdown("---")

if RAG_PIPELINE_IMPORT_ERROR is not None:
    st.error("Backend failed to load. Please check server logs.")
    st.exception(RAG_PIPELINE_IMPORT_ERROR)
    st.stop()

# Initialize state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "database_ready" not in st.session_state:
    st.session_state.database_ready = False

if "uploaded_pdf_names" not in st.session_state:
    st.session_state.uploaded_pdf_names = []

# Sidebar
with st.sidebar:
    st.header("📂 Upload Documents")
    st.info(
        "Upload one or more PDFs to build a searchable document database. "
        "OCR is applied automatically for scanned content."
    )
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        uploaded_names = [file.name for file in uploaded_files]

        if uploaded_names != st.session_state.uploaded_pdf_names:
            os.makedirs("data", exist_ok=True)
            pdf_paths = []
            for uploaded_file in uploaded_files:
                pdf_path = os.path.join("data", uploaded_file.name)
                with open(pdf_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())
                pdf_paths.append(pdf_path)

            if st.session_state.pipeline is None:
                RAGPipeline, RAG_PIPELINE_IMPORT_ERROR = import_rag_pipeline()
                if RAG_PIPELINE_IMPORT_ERROR is not None:
                    st.error("Backend failed to load. Please check server logs.")
                    st.exception(RAG_PIPELINE_IMPORT_ERROR)
                    st.stop()
                st.session_state.pipeline = RAGPipeline()

            with st.spinner("Building vector database..."):
                st.session_state.pipeline.build_database(
                    pdf_paths,
                    embedding_model=get_embedding_model(),
                )
                st.session_state.pipeline.vector_store = get_vector_store(
                    st.session_state.pipeline.database_path,
                )

            if st.session_state.pipeline.ocr_documents:
                st.warning(
                    "📄 OCR was used for:\n\n"
                    + "\n".join(st.session_state.pipeline.ocr_documents)
                )
            else:
                st.success("✅ Text-based PDFs detected. OCR was not required.")

            st.session_state.database_ready = True
            st.session_state.uploaded_pdf_names = uploaded_names
            st.session_state.messages = []
            st.success("Database Created Successfully!")

    st.divider()

    if st.session_state.database_ready:
        st.subheader("📄 Uploaded Documents")
        col1, col2, col3 = st.columns(3)
        col1.metric("PDFs", len(st.session_state.pipeline.pdf_names))
        col2.metric("Pages", st.session_state.pipeline.total_pages)
        col3.metric("Chunks", st.session_state.pipeline.total_chunks)

        st.write("### Files")
        for pdf in st.session_state.pipeline.pdf_names:
            st.success(pdf)

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question = st.chat_input("Enter your question about the uploaded documents")

if question:
    if not st.session_state.database_ready:
        st.warning("⚠️ Please upload a PDF first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            if st.session_state.pipeline.llm is None:
                st.session_state.pipeline.llm = get_llm()
            answer, sources, confidence = st.session_state.pipeline.ask(question)

        document_sources = defaultdict(set)
        for source in sources:
            pdf_name = os.path.basename(source.metadata["source"])
            document_sources[pdf_name].add(source.metadata["page"])

        source_lines = []
        for pdf_name, pages in document_sources.items():
            page_list = ", ".join(map(str, sorted(pages)))
            source_lines.append(f"• {pdf_name} → Page(s): {page_list}")

        if source_lines:
            sources_text = "\n".join(source_lines)
            response = f"""
{answer}

---
📊 **Confidence:** {confidence}%
📚 **Sources**
{sources_text}
"""
        else:
            response = answer

        st.markdown(response)
        with st.expander("🔍 View Retrieved Context"):
            for index, source in enumerate(sources, start=1):
                st.markdown(f"### Result {index}")
                st.write(f"**Page:** {source.metadata['page']}")
                st.write(source.page_content)
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response})