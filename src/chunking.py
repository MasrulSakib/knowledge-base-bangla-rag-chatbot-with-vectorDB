"""Step 2a - Clean the chapter text and split it into small overlapping chunks."""
import json
import re
import unicodedata

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import BOOK_TITLE, CHAPTERS_FILE


def load_chapters():
    if not CHAPTERS_FILE.exists():
        raise FileNotFoundError("data/chapters.json not found. Run `python -m src.ingest` first.")
    return json.loads(CHAPTERS_FILE.read_text(encoding="utf-8"))


def clean_text(text):
    text = unicodedata.normalize("NFC", text)  # one consistent form for Bengali letters
    text = re.sub(r"[\u200b\ufeff]", "", text)  # invisible characters (ZWJ and ZWNJ stay: Bengali needs them)
    text = re.sub(r"\[[0-9০-৯]+\]", "", text)  # footnote markers such as [১]
    text = re.sub(r"[ \t\u00a0]+", " ", text)  # repeated spaces
    text = re.sub(r" *\n *", "\n", text)  # spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)  # blank lines
    return text.strip()


def make_chunks(chapters, chunk_size, chunk_overlap):
    """Split every chapter into chunks. Each chunk remembers where it came from."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "।", " ", ""],  # try paragraph, then line, then sentence end
        keep_separator="end",
    )
    chunks = []
    for chapter in chapters:
        pieces = splitter.split_text(clean_text(chapter["text"]))
        for number, piece in enumerate(pieces, start=1):
            metadata = {
                "book": BOOK_TITLE,
                "chapter": chapter["chapter"],
                "section": f"অংশ {number}/{len(pieces)}",
                "source_url": chapter["url"],
            }
            chunks.append(Document(page_content=piece, metadata=metadata))
    return chunks
