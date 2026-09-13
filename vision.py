import os
import base64
import tempfile

import streamlit as st
from groq import Groq

from src.config import Config
from src.ingestion import DocumentIngestion
from src.vector_store import VectorStore
from src.cache import RedisCache
from src.retrieval import HybridRetriever, RAGPipeline


# ============================================================
# CONFIG
# ============================================================

VISION_MODEL = "qwen/qwen3.6-27b"


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="ChatPDF + Vision",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 ChatPDF + Vision")
st.caption(
    "Chat with PDFs or understand images."
)


# ============================================================
# GROQ API
# ============================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error(
        "GROQ_API_KEY is not configured."
    )
    st.stop()

groq_client = Groq(
    api_key=api_key
)


# ============================================================
# MODE SELECTOR
# ============================================================

mode = st.radio(
    "Choose what you want to use",
    [
        "📄 PDF",
        "🖼️ Image"
    ],
    horizontal=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "pdf_pipeline" not in st.session_state:
    st.session_state.pdf_pipeline = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_id" not in st.session_state:
    st.session_state.pdf_id = None

if "vision_messages" not in st.session_state:
    st.session_state.vision_messages = []


# ============================================================
# PDF MODE
# ============================================================

if mode == "📄 PDF":

    st.subheader("📄 Chat with a PDF")

    uploaded_pdf = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        key="pdf_uploader"
    )

    if uploaded_pdf is not None:

        document_id = (
            f"{uploaded_pdf.name}:"
            f"{uploaded_pdf.size}"
        )

        if (
            st.session_state.pdf_id
            != document_id
        ):

            with st.spinner(
                "Processing PDF..."
            ):

                temp_dir = tempfile.mkdtemp()

                pdf_path = os.path.join(
                    temp_dir,
                    uploaded_pdf.name
                )

                with open(
                    pdf_path,
                    "wb"
                ) as f:

                    f.write(
                        uploaded_pdf.getbuffer()
                    )

                config = Config()

                config.path = temp_dir

                loader = DocumentIngestion(
                    config
                )

                chunks = (
                    loader.load_and_chunk()
                )

                cache = RedisCache(
                    config
                )

                vector_store = VectorStore(
                    config,
                    cache
                )

                vector_store.ingest(
                    chunks
                )

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

                st.session_state.pdf_pipeline = (
                    pipeline
                )

                st.session_state.pdf_name = (
                    uploaded_pdf.name
                )

                st.session_state.pdf_id = (
                    document_id
                )

            st.success(
                f"✅ {uploaded_pdf.name} is ready!"
            )

        if st.session_state.pdf_pipeline:

            st.divider()

            st.subheader(
                f"📖 {st.session_state.pdf_name}"
            )

            question = st.chat_input(
                "Ask something about your PDF...",
                key="pdf_chat"
            )

            if question:

                with st.chat_message("user"):

                    st.write(question)

                with st.chat_message(
                    "assistant"
                ):

                    with st.spinner(
                        "Searching document..."
                    ):

                        answer = (
                            st.session_state
                            .pdf_pipeline
                            .answer(question)
                        )

                    st.write(answer)

    else:

        st.info(
            "👆 Upload a PDF to start."
        )


# ============================================================
# IMAGE MODE
# ============================================================

else:

    st.subheader("🖼️ Vision")

    st.write(
        "Upload an image or take a photo "
        "with your camera."
    )

    # --------------------------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------------------------

    uploaded_image = st.file_uploader(
        "🖼️ Upload image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        key="image_uploader"
    )

    st.write("**OR**")

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    camera_image = st.camera_input(
        "📷 Take a photo",
        key="camera_input"
    )

    # --------------------------------------------------------
    # SELECT IMAGE
    # --------------------------------------------------------

    image = None

    if camera_image is not None:

        image = camera_image

    elif uploaded_image is not None:

        image = uploaded_image

    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    if image is not None:

        st.image(
            image,
            caption="Selected image",
            use_container_width=True
        )

        st.divider()

        question = st.chat_input(
            "Ask anything about this image...",
            key="vision_chat"
        )

        if question:

            with st.chat_message("user"):

                st.write(question)

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Analyzing image..."
                ):

                    image_bytes = (
                        image.getvalue()
                    )

                    base64_image = (
                        base64.b64encode(
                            image_bytes
                        ).decode("utf-8")
                    )

                    mime_type = image.type

                    image_data = (
                        f"data:{mime_type};"
                        f"base64,{base64_image}"
                    )

                    response = (
                        groq_client
                        .chat
                        .completions
                        .create(
                            model=VISION_MODEL,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": question
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": image_data
                                            }
                                        }
                                    ]
                                }
                            ],
                            temperature=0.7,
                            max_completion_tokens=1024
                        )
                    )

                    answer = (
                        response
                        .choices[0]
                        .message
                        .content
                    )

                st.write(answer)

    else:

        st.info(
            "Choose an image from your phone "
            "or take a photo."
        )