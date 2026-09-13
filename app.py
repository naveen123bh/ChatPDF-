import streamlit as st
import os
import tempfile

from src.config import Config
from src.ingestion import DocumentIngestion
from src.vector_store import VectorStore
from src.cache import RedisCache
from src.retrieval import HybridRetriever, RAGPipeline


st.set_page_config(
    page_title="ChatPDF",
    page_icon="📚",
    layout="centered"
)


st.title("📚 ChatPDF")
st.caption("Upload a PDF and ask questions about it.")


# =========================
# Session State
# =========================

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# =========================
# Upload PDF
# =========================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    # Avoid rebuilding when same PDF is already loaded
    if st.session_state.document_name != uploaded_file.name:

        with st.spinner("Processing PDF..."):

            temp_dir = tempfile.mkdtemp()

            pdf_path = os.path.join(
                temp_dir,
                uploaded_file.name
            )

            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # =========================
            # Build existing RAG pipeline
            # =========================

            config = Config()

            # Point ingestion to uploaded PDF
            config.path = temp_dir

            loader = DocumentIngestion(config)

            chunks = loader.load_and_chunk()

            cache = RedisCache(config)

            vector_store = VectorStore(
                config,
                cache
            )

            vector_store.ingest(chunks)

            retriever = HybridRetriever(
                chunks,
                vector_store,
                config
            )

            pipeline = RAGPipeline(
                config,
                retriever,
                cache,
                vector_store
            )

            st.session_state.pipeline = pipeline
            st.session_state.document_name = uploaded_file.name

        st.success(
            f"✅ {uploaded_file.name} is ready!"
        )


# =========================
# Ask Question
# =========================

if st.session_state.pipeline is not None:

    st.divider()

    st.subheader("Ask a question")

    question = st.chat_input(
        "Ask something about your PDF..."
    )

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = st.session_state.pipeline.answer(
                    question
                )

            st.write(answer)

else:

    st.info(
        "👆 Upload a PDF to start chatting."
    )