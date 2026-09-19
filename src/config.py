"""All settings live here. Change a value once and every step picks it up."""
import os
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY from the .env file

ROOT = Path(__file__).resolve().parent.parent

# ---- The book ----
BOOK_TITLE = "অব্যক্ত"
BOOK_AUTHOR = "জগদীশচন্দ্র বসু"
WIKISOURCE_HOST = "https://bn.wikisource.org"
WIKISOURCE_API = f"{WIKISOURCE_HOST}/w/api.php"
BOOK_URL = f"{WIKISOURCE_HOST}/wiki/{quote(BOOK_TITLE)}"

# ---- Files and folders ----
CHAPTERS_FILE = ROOT / "data" / "chapters.json"
INDEX_ROOT = ROOT / "vector_store"
QUESTIONS_FILE = ROOT / "tests" / "test_questions.json"
RESULTS_FILE = ROOT / "RESULTS.md"
README_FILE = ROOT / "README.md"

# ---- Chunking (sizes are in characters) ----
# Two strategies are built so the bonus comparison can measure which one retrieves better.
CHUNK_STRATEGIES = {
    "small": {"chunk_size": 500, "chunk_overlap": 100},
    "large": {"chunk_size": 1000, "chunk_overlap": 200},
}
DEFAULT_STRATEGY = "small"  # the strategy the chatbot uses

# ---- Embeddings, retrieval, LLM ----
EMBEDDING_MODEL = "BAAI/bge-m3"  # multilingual, supports Bengali
TOP_K = 4  # how many chunks the retriever returns for each question
LLM_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")  # override in .env if Groq retires it
