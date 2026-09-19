"""Step 1 - Crawl every chapter of the book from Bengali Wikisource and save it as JSON.

Run:  python -m src.ingest
"""
import json
import time
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from src.config import BOOK_TITLE, CHAPTERS_FILE, WIKISOURCE_API, WIKISOURCE_HOST

# Wikimedia asks bots to identify themselves: add your email or GitHub link inside the brackets.
HEADERS = {"User-Agent": "AbyaktaRagChatbot/1.0 (university assignment)"}
REQUEST_DELAY = 1.0  # seconds to wait between requests
MAX_RETRIES = 5  # how many times to retry when Wikisource says "too many requests"
MIN_CHARS = 200  # pages shorter than this are not real chapters
# Parts of a Wikisource page that are not book text (page numbers, edit links, footnotes...)
JUNK_SELECTORS = ".noprint, .ws-noexport, .mw-editsection, .pagenum, .ws-pagenum, sup.reference, style, script"


def call_api(**params):
    """Send one request to the Wikisource API and return the JSON answer."""
    params.update(format="json", formatversion=2)
    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.get(WIKISOURCE_API, params=params, headers=HEADERS, timeout=30)
        if response.status_code != 429:
            break
        retry_after = response.headers.get("Retry-After", "")
        wait = min(int(retry_after) if retry_after.isdigit() else 10 * attempt, 60)
        print(f"  Wikisource asked us to slow down. Waiting {wait} seconds (try {attempt}/{MAX_RETRIES}) ...")
        time.sleep(wait)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"Wikisource API error: {data['error']['info']}")
    time.sleep(REQUEST_DELAY)  # be polite to the Wikisource servers
    return data


def list_subpage_titles():
    """Return every page whose title starts with 'অব্যক্ত/' (these are the chapters)."""
    titles, extra = [], {}
    while True:
        data = call_api(action="query", list="allpages", apprefix=f"{BOOK_TITLE}/", aplimit=500, **extra)
        titles += [page["title"] for page in data["query"]["allpages"]]
        if "continue" not in data:
            return titles
        extra = {"apcontinue": data["continue"]["apcontinue"]}


def fetch_text(title):
    """Download one page and return only its readable text."""
    data = call_api(action="parse", page=title, prop="text", redirects=1)
    soup = BeautifulSoup(data["parse"]["text"], "html.parser")
    for tag in soup.select(JUNK_SELECTORS):
        tag.decompose()
    for line_break in soup.find_all("br"):
        line_break.replace_with("\n")
    blocks = soup.select(".prp-pages-output") or [soup]  # the proofread book text, if marked
    return "\n\n".join(block.get_text() for block in blocks)


def page_url(title):
    return f"{WIKISOURCE_HOST}/wiki/{quote(title.replace(' ', '_'))}"


def crawl_book():
    """Download the main page and every chapter subpage."""
    chapters = []
    for title in [BOOK_TITLE] + list_subpage_titles():
        text = fetch_text(title)
        chapter = "মূল পাতা" if title == BOOK_TITLE else title.removeprefix(f"{BOOK_TITLE}/")
        if len(text.strip()) < MIN_CHARS:
            print(f"  skipped '{chapter}' (only {len(text.strip())} characters)")
            continue
        print(f"  {chapter}: {len(text):,} characters")
        chapters.append({"chapter": chapter, "url": page_url(title), "text": text})
    return chapters


def main():
    print(f"Crawling '{BOOK_TITLE}' from {WIKISOURCE_HOST} ...")
    chapters = crawl_book()
    CHAPTERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHAPTERS_FILE.write_text(json.dumps(chapters, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(chapter["text"]) for chapter in chapters)
    print(f"Saved {len(chapters)} pages ({total:,} characters) to {CHAPTERS_FILE}")


if __name__ == "__main__":
    main()