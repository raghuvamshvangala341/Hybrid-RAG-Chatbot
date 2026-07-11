import streamlit as st


def display_metrics(metrics: dict):

    st.divider()

    st.subheader("📊 System Metrics")

    tab1, tab2, tab3 = st.tabs(
        [
            "Retrieval",
            "LLM",
            "Ingestion"
        ]
    )

    # ==================================================
    # Retrieval
    # ==================================================

    with tab1:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Retrieval Time",
                f"{metrics.get('retrieval_time', 0):.3f} sec"
            )

            st.metric(
                "Retrieved Chunks",
                metrics.get(
                    "retrieved_chunks",
                    0
                )
            )

        with col2:

            st.metric(
                "Dense Score",
                f"{metrics.get('average_dense_score', 0):.4f}"
            )

            st.metric(
                "BM25 Score",
                f"{metrics.get('average_bm25_score', 0):.4f}"
            )

        with col3:

            st.metric(
                "Hybrid Score",
                f"{metrics.get('average_hybrid_score', 0):.4f}"
            )

            st.metric(
                "Cosine Similarity",
                f"{metrics.get('average_cosine_similarity', 0):.4f}"
            )

    # ==================================================
    # LLM
    # ==================================================

    with tab2:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "LLM Response Time",
                f"{metrics.get('llm_response_time', 0):.3f} sec"
            )

            st.metric(
                "Prompt Tokens",
                metrics.get(
                    "prompt_tokens",
                    0
                )
            )

        with col2:

            st.metric(
                "Completion Tokens",
                metrics.get(
                    "completion_tokens",
                    0
                )
            )

            st.metric(
                "Total Tokens",
                metrics.get(
                    "total_tokens",
                    0
                )
            )

    # ==================================================
    # Ingestion
    # ==================================================

    with tab3:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Documents",
                metrics.get(
                    "documents",
                    0
                )
            )

            st.metric(
                "Chunks",
                metrics.get(
                    "chunks",
                    0
                )
            )

        with col2:

            st.metric(
                "Embedding Time",
                f"{metrics.get('embedding_time', 0):.3f} sec"
            )

            st.metric(
                "Indexing Time",
                f"{metrics.get('indexing_time', 0):.3f} sec"
            )

        with col3:

            st.metric(
                "Total Ingestion",
                f"{metrics.get('total_ingestion_time', 0):.3f} sec"
            )

            st.metric(
                "Embedding Cache",
                metrics.get(
                    "embedding_cache_size",
                    0
                )
            )