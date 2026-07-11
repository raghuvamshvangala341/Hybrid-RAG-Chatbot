from llm.prompt import (
    SYSTEM_PROMPT,
    RAG_PROMPT
)

from llm.groq_client import GroqClient
from retrieval.retriever import HybridRetriever


class RAGChain:

    def __init__(self,ingestion_metrics=None):

        self.retriever = HybridRetriever()

        self.llm = GroqClient()
        
        self.ingestion_metrics = (
            ingestion_metrics or {}
        )

    def _build_context(self, results):

        context = []

        for result in results:

            source = result["source"]

            page = result["page"]

            text = result["text"]

            context.append(

                f"""Source: {source}
Page: {page}

{text}
"""

            )

        return "\n\n".join(context)

    def ask(
        self,
        question: str
    ):

        results, retrieval_metrics = (

            self.retriever.retrieve(question)

        )

        context = self._build_context(
            results
        )

        prompt = RAG_PROMPT.format(

            context=context,

            question=question

        )

        answer, llm_metrics = (

            self.llm.generate(

                SYSTEM_PROMPT,

                prompt

            )

        )

        metrics = {

            **self.ingestion_metrics,

            **retrieval_metrics,

            **llm_metrics

        }

        sources = []

        for result in results:

            source = {

                "source": result["source"],

                "page": result["page"]

            }

            if source not in sources:

                sources.append(source)

        return {

            "question": question,

            "answer": answer,

            "sources": sources,

            "context": context,

            "metrics": metrics

        }

    def stream(
        self,
        question: str
    ):

        results, retrieval_metrics = (

            self.retriever.retrieve(question)

        )

        context = self._build_context(
            results
        )

        prompt = RAG_PROMPT.format(

            context=context,

            question=question

        )

        return (

            self.llm.stream(

                SYSTEM_PROMPT,

                prompt

            ),

            results,

            retrieval_metrics

        )