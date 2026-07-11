import uuid
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config


class DocumentSplitter:

    def __init__(self):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                " ",
                ""
            ]
        )

    def split_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:

        split_docs = self.splitter.split_documents(documents)

        total_chunks = len(split_docs)

        for index, chunk in enumerate(split_docs, start=1):

            chunk.metadata.update(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_number": index,
                    "total_chunks": total_chunks,
                    "char_count": len(chunk.page_content),
                    "word_count": len(chunk.page_content.split())
                }
            )

        return split_docs