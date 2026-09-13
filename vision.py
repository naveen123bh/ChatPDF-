import os
import base64

import streamlit as st
from groq import Groq


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "qwen/qwen3.6-27b"


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="ChatPDF Vision",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ ChatPDF Vision")
st.caption(
    "Upload an image and ask questions about it."
)


# ============================================================
# GROQ CLIENT
# ============================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error(
        "GROQ_API_KEY is not configured."
    )
    st.stop()

client = Groq(
    api_key=api_key
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_image = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# IMAGE DISPLAY
# ============================================================

if uploaded_image is not None:

    st.image(
        uploaded_image,
        caption="Uploaded image",
        use_container_width=True
    )

    question = st.chat_input(
        "Ask something about this image..."
    )

    if question:

        image_bytes = uploaded_image.getvalue()

        base64_image = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        mime_type = uploaded_image.type

        image_data = (
            f"data:{mime_type};"
            f"base64,{base64_image}"
        )

        with st.chat_message("user"):

            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Analyzing image..."
            ):

                response = client.chat.completions.create(
                    model=MODEL_NAME,
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

                answer = (
                    response
                    .choices[0]
                    .message
                    .content
                )

            st.write(answer)

else:

    st.info(
        "👆 Upload an image to start."
    )