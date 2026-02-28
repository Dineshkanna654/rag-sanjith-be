from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

_LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}


def load_and_split_file(path: Path, filename: str) -> list[Document]:
    suffix = Path(filename).suffix.lower()
    loader_cls = _LOADERS.get(suffix)
    if not loader_cls:
        raise ValueError(f"Unsupported file type: {suffix}")
    loader = loader_cls(str(path))
    docs = loader.load()
    return _splitter.split_documents(docs)
