import os
import tempfile

import streamlit as st

from src.config import Config
from src.ingestion import DocumentIngestion
from src.vector_store import VectorStore
from src.cache import RedisCache
from src.retrieval import HybridRetriever, RAGPipeline


# =============================================================
# PAGE CONFIG
# =============================================================

st.set_page_config(
    page_title="ChatPDF",
    page_icon="📚",
    layout="centered"
)


st.title("📚 ChatPDF")
st.caption(
    "Upload a PDF and ask questions about it."
)


# =============================================================
# SESSION STATE
# =============================================================

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_id" not in st.session_state:
    st.session_state.document_id = None


# =============================================================
# PDF UPLOAD
# =============================================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    # Use file name + size as a simple document identity.
    # This prevents unnecessary rebuilding on normal Streamlit reruns.
    document_id = (
        f"{uploaded_file.name}:"
        f"{uploaded_file.size}"
    )

    # ---------------------------------------------------------
    # Build pipeline only when a new PDF is uploaded
    # ---------------------------------------------------------

    if st.session_state.document_id != document_id:

        with st.spinner(
            "Processing PDF..."
        ):

            # Create temporary directory
            temp_dir = tempfile.mkdtemp()

            pdf_path = os.path.join(
                temp_dir,
                uploaded_file.name
            )

            # Save uploaded PDF
            with open(
                pdf_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getbuffer()
                )

            # -------------------------------------------------
            # Configuration
            # -------------------------------------------------

            config = Config()

            # Tell ingestion where the uploaded PDF lives
            config.path = temp_dir

            # -------------------------------------------------
            # Load + chunk PDF
            # -------------------------------------------------

            loader = DocumentIngestion(
                config
            )

            chunks = loader.load_and_chunk()

            # -------------------------------------------------
            # Create memory cache
            # -------------------------------------------------

            cache = RedisCache(
                config
            )

            # -------------------------------------------------
            # Create FAISS vector store
            # -------------------------------------------------

            vector_store = VectorStore(
                config,
                cache
            )

            # Insert document chunks into FAISS
            vector_store.ingest(
                chunks
            )

            # -------------------------------------------------
            # Hybrid retriever
            # -------------------------------------------------

            retriever = HybridRetriever(
                chunks,
                vector_store,
                config
            )

            # -------------------------------------------------
            # Complete RAG pipeline
            # -------------------------------------------------

            pipeline = RAGPipeline(
                config,
                retriever,
                cache,
                vector_store
            )

            # -------------------------------------------------
            # Save everything in Streamlit session
            # -------------------------------------------------

            st.session_state.pipeline = pipeline

            st.session_state.document_name = (
                uploaded_file.name
            )

            st.session_state.document_id = (
                document_id
            )

        st.success(
            f"✅ {uploaded_file.name} is ready!"
        )


# =============================================================
# CHAT
# =============================================================

if st.session_state.pipeline is not None:

    st.divider()

    st.subheader(
        f"📖 {st.session_state.document_name}"
    )

    question = st.chat_input(
        "Ask something about your PDF..."
    )

    if question:

        # -----------------------------------------------------
        # User message
        # -----------------------------------------------------

        with st.chat_message(
            "user"
        ):

            st.write(
                question
            )

        # -----------------------------------------------------
        # Assistant response
        # -----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Searching document..."
            ):

                answer = (
                    st.session_state.pipeline.answer(
                        question
                    )
                )

            st.write(
                answer
            )


# =============================================================
# EMPTY STATE
# =============================================================

else:

    st.info(
        "👆 Upload a PDF to start chatting."
    )