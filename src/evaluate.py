"""Step 4 - Run the 10 test questions and compare the chunking strategies (bonus).

Run:  python -m src.evaluate                    (full run, needs the Groq key)
      python -m src.evaluate --retrieval-only   (only the chunking comparison, no key needed)

Results are written to RESULTS.md and to the marked block in README.md.
"""
import argparse
import json
import re
import unicodedata

from src.chunking import clean_text, load_chapters
from src.config import CHUNK_STRATEGIES, QUESTIONS_FILE, README_FILE, RESULTS_FILE, TOP_K
from src.rag_chain import ask, build_rag_chain
from src.vector_store import load_index


def normalize(text):
    return unicodedata.normalize("NFC", text)


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        cells = (str(cell).replace("\n", " ").replace("|", "/") for cell in row)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_questions():
    return json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))


def check_questions(questions):
    """Warn early if a test question does not match the crawled book (wrong chapter name or spelling)."""
    book = {normalize(c["chapter"]): normalize(clean_text(c["text"])) for c in load_chapters()}
    for q in questions:
        if not q["chapter"]:
            continue
        texts = [text for name, text in book.items() if normalize(q["chapter"]) in name]
        if not texts:
            print(f"WARNING: question {q['id']}: chapter '{q['chapter']}' was not crawled. Fix tests/test_questions.json.")
        elif not any(normalize(word) in texts[0] for word in q["keywords"]):
            print(f"WARNING: question {q['id']}: none of its keywords appear in chapter '{q['chapter']}'.")


def passage_hit(docs, question):
    """True if a retrieved chunk comes from the right chapter and contains an answer keyword."""
    return any(
        normalize(question["chapter"]) in normalize(doc.metadata["chapter"])
        and any(normalize(word) in normalize(doc.page_content) for word in question["keywords"])
        for doc in docs
    )


def run_test_questions(questions):
    """Ask the chatbot every question. Returns (number passed, markdown table)."""
    chain = build_rag_chain()
    rows, passed = [], 0
    for q in questions:
        result = ask(chain, q["question"])
        if q["chapter"]:
            retrieved = "yes" if passage_hit(result["docs"], q) else "no"
            ok = result["found"] and any(normalize(word) in normalize(result["answer"]) for word in q["keywords"])
        else:  # the book has no answer, so the chatbot must say so
            retrieved = "-"
            ok = not result["found"]
        passed += ok
        rows.append([q["id"], q["question"], q["expected_answer"], q["chapter"] or "(not in book)",
                     result["answer"], retrieved, "PASS" if ok else "FAIL"])
    headers = ["#", "Question", "Expected answer", "Chapter", "Chatbot answer", "Right passage retrieved", "Result"]
    return passed, markdown_table(headers, rows)


def compare_chunking(questions):
    """Bonus: for each chunking strategy, how often does the retriever find the answer passage?"""
    answerable = [q for q in questions if q["chapter"]]
    rows = []
    for strategy, settings in CHUNK_STRATEGIES.items():
        store = load_index(strategy)
        top1 = sum(passage_hit(store.similarity_search(q["question"], k=1), q) for q in answerable)
        topk = sum(passage_hit(store.similarity_search(q["question"], k=TOP_K), q) for q in answerable)
        rows.append([strategy, f"{settings['chunk_size']} / {settings['chunk_overlap']}", store.index.ntotal,
                     f"{top1}/{len(answerable)}", f"{topk}/{len(answerable)}"])
    headers = ["Strategy", "Chunk size / overlap", "Chunks", "Hit@1", f"Hit@{TOP_K}"]
    note = "A hit means a retrieved chunk is from the right chapter and contains the answer keyword."
    return markdown_table(headers, rows) + f"\n\n{note}"


def update_readme(summary):
    """Replace the block between the RESULTS markers in README.md with fresh results."""
    pattern = r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)"
    text = re.sub(pattern, lambda m: f"{m.group(1)}\n{summary}\n{m.group(2)}", README_FILE.read_text(encoding="utf-8"), flags=re.S)
    README_FILE.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM and only compare chunking")
    args = parser.parse_args()

    questions = load_questions()
    check_questions(questions)

    summary = f"### Chunking strategy comparison\n\n{compare_chunking(questions)}"
    details = ""
    if not args.retrieval_only:
        passed, table = run_test_questions(questions)
        summary = f"### Test questions\n\n**{passed} of {len(questions)} passed.**\n\n{summary}"
        details = f"\n\n### Every test question\n\n{table}"

    RESULTS_FILE.write_text(f"# Evaluation results\n\n{summary}{details}\n", encoding="utf-8")
    update_readme(summary)
    print(f"Done. See {RESULTS_FILE.name} and the results block in README.md")


if __name__ == "__main__":
    main()
