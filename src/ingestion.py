
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    DirectoryLoader,
)

from src.config import Config

load_dotenv()


class DocumentIngestion:

    def __init__(self, config: Config):
        self.config = config

    def loadDoc(self) -> list[Document]:
        path = self.config.path

        directory_loader = DirectoryLoader(
            path,
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=False,
        )

        docs = directory_loader.load()

        print(f"Loaded {len(docs)} PDF pages")

        return docs

    def chunking(self, docs: list[Document]) -> list[Document]:

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ", ",
                " ",
                "",
            ],
        )

        chunks = text_splitter.split_documents(docs)

        print(f"Created {len(chunks)} chunks")

        return chunks

    def load_and_chunk(self) -> list[Document]:

        docs = self.loadDoc()

        chunks = self.chunking(docs)

        return chunks
