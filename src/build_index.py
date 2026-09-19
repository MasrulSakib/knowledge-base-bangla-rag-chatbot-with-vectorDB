"""Step 2 - Build one FAISS index for every chunking strategy listed in config.py.

Run:  python -m src.build_index
"""
from src.config import CHUNK_STRATEGIES
from src.vector_store import build_index

if __name__ == "__main__":
    for strategy in CHUNK_STRATEGIES:
        build_index(strategy)
