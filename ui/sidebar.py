import streamlit as st


def render_sidebar():

    with st.sidebar:

        st.title("🤖 Hybrid RAG Chatbot")

        st.divider()

        uploaded_file = st.file_uploader(

            "Upload Document",

            type=[
                "pdf",
                "docx",
                "txt"
            ]

        )

        st.divider()

        st.subheader("Configuration")

        st.write("LLM : Groq")

        st.write("Vector DB : Pinecone")

        st.write("Search : Hybrid")

        st.write("Embeddings : BGE Small")

        st.divider()

        clear_chat = st.button(
            "🗑 Clear Chat",
            use_container_width=True
        )

    return uploaded_file, clear_chat