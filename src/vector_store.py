"""Step 2b - Turn chunks into embeddings and keep them in a FAISS vector database."""
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.chunking import load_chapters, make_chunks
from src.config import CHUNK_STRATEGIES, DEFAULT_STRATEGY, EMBEDDING_MODEL, INDEX_ROOT


@lru_cache(maxsize=1)
def get_embeddings():
    """Load the embedding model once (the first run downloads it)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})


def build_index(strategy):
    """Chunk the book with one strategy, embed the chunks, and save the FAISS index."""
    chunks = make_chunks(load_chapters(), **CHUNK_STRATEGIES[strategy])
    print(f"[{strategy}] embedding {len(chunks)} chunks ...")
    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(str(INDEX_ROOT / strategy))
    print(f"[{strategy}] saved to {INDEX_ROOT / strategy}")


def load_index(strategy=DEFAULT_STRATEGY):
    path = INDEX_ROOT / strategy
    if not path.exists():
        raise FileNotFoundError(f"Vector index '{strategy}' not found. Run `python -m src.build_index` first.")
    # The index is a file we created ourselves, so loading it is safe.
    return FAISS.load_local(str(path), get_embeddings(), allow_dangerous_deserialization=True)
