import tempfile
from pathlib import Path

import streamlit as st

from ingestion.ingest import IngestionPipeline
from llm.rag_chain import RAGChain

from ui.sidebar import render_sidebar
from ui.chat import (
    initialize_chat,
    display_chat,
    add_user_message,
    add_assistant_message,
)
from ui.metrics_panel import display_metrics


st.set_page_config(
    page_title="Hybrid RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

initialize_chat()

uploaded_file, clear_chat = render_sidebar()

if clear_chat:
    st.session_state.messages = []
    st.rerun()

st.title("🤖 Hybrid RAG Chatbot")

display_chat()

if uploaded_file is not None:

    if (
        "current_file" not in st.session_state
        or st.session_state.current_file != uploaded_file.name
    ):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(uploaded_file.name).suffix
        ) as tmp:

            tmp.write(uploaded_file.getbuffer())

            temp_path = tmp.name

        with st.spinner("Processing document..."):

            pipeline = IngestionPipeline()

            chunks, ingestion_metrics = pipeline.ingest(
                temp_path
            )

        st.session_state.current_file = uploaded_file.name
        st.session_state.ingested = True
        st.session_state.ingestion_metrics = ingestion_metrics

        st.sidebar.success(
            f"Indexed {ingestion_metrics['chunks']} chunks"
        )

if prompt := st.chat_input("Ask a question..."):

    if not st.session_state.get("ingested", False):

        st.warning("Please upload a document first.")

        st.stop()

    add_user_message(prompt)

    with st.chat_message("user"):

        st.markdown(prompt)

    rag = RAGChain(st.session_state.ingestion_metrics)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = rag.ask(prompt)

            answer = response["answer"]

            st.markdown(answer)

            if response["sources"]:

                st.markdown("---")

                st.markdown("### 📄 Sources")

                for source in response["sources"]:

                    st.write(
                        f"• {source['source']} (Page {source['page']})"
                    )

    add_assistant_message(answer)

    display_metrics(response["metrics"])