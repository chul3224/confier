import os
import hashlib
from pathlib import Path

import pdfplumber
import chromadb
from chromadb.utils import embedding_functions


DOCUMENTS_DIR = Path("documents")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "recipes"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 50]


def _pdf_to_text(pdf_path: Path) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


class RecipeRAG:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def index_documents(self) -> dict:
        pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
        if not pdf_files:
            return {"indexed": 0, "files": []}

        indexed_files = []
        for pdf_path in pdf_files:
            file_id = _file_hash(pdf_path)
            existing = self._collection.get(where={"source": pdf_path.name})
            if existing["ids"]:
                continue

            text = _pdf_to_text(pdf_path)
            if not text.strip():
                continue

            chunks = _chunk_text(text)
            ids = [f"{file_id}_{i}" for i in range(len(chunks))]
            self._collection.add(
                documents=chunks,
                ids=ids,
                metadatas=[{"source": pdf_path.name, "chunk": i} for i in range(len(chunks))],
            )
            indexed_files.append(pdf_path.name)

        return {"indexed": len(indexed_files), "files": indexed_files}

    def query(self, query_text: str, n_results: int = 4) -> list[dict]:
        count = self._collection.count()
        if count == 0:
            return []

        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, count),
        )
        items = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            items.append({"text": doc, "source": meta.get("source", "")})
        return items

    def get_indexed_files(self) -> list[str]:
        if self._collection.count() == 0:
            return []
        results = self._collection.get()
        return list({m["source"] for m in results["metadatas"]})

    def reset(self):
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
