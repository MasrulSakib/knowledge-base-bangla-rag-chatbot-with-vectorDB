"""Step 3 - The RAG pipeline: question -> retriever -> prompt -> LLM -> answer + citations."""
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq

from src.config import BOOK_AUTHOR, BOOK_TITLE, DEFAULT_STRATEGY, LLM_MODEL, TOP_K
from src.vector_store import load_index

NOT_FOUND_MESSAGE = "দুঃখিত, এই প্রশ্নের উত্তর বইটিতে পাওয়া যায়নি।"

PROMPT = ChatPromptTemplate.from_template(
    """You answer questions about the Bengali book "{book}" by {author}.
Use ONLY the context below. Do not use outside knowledge and do not guess.
Write the answer in Bengali.
If the context does not contain the answer, reply with exactly: {not_found}

Context:
{context}

Question: {question}"""
).partial(book=BOOK_TITLE, author=BOOK_AUTHOR, not_found=NOT_FOUND_MESSAGE)


def format_context(docs):
    """Put the retrieved chunks into one text block, each labelled with its chapter."""
    return "\n\n".join(f"[{doc.metadata['chapter']}, {doc.metadata['section']}]\n{doc.page_content}" for doc in docs)


def format_sources(docs):
    """Turn retrieved chunks into unique markdown links: chapter (section) -> Wikisource page."""
    links = (f"[{doc.metadata['chapter']} ({doc.metadata['section']})]({doc.metadata['source_url']})" for doc in docs)
    return list(dict.fromkeys(links))


def build_rag_chain(strategy=DEFAULT_STRATEGY):
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing. Copy .env.example to .env and paste your key.")

    retriever = load_index(strategy).as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatGroq(model=LLM_MODEL, temperature=0)
    write_answer = (
        RunnableLambda(lambda step: {"context": format_context(step["docs"]), "question": step["question"]})
        | PROMPT
        | llm
        | StrOutputParser()
    )
    # 1) retrieve the chunks   2) add the LLM answer next to them
    return {"docs": retriever, "question": RunnablePassthrough()} | RunnablePassthrough.assign(answer=write_answer)


def ask(chain, question):
    """Answer one question. Sources are only returned when the book really contained the answer."""
    result = chain.invoke(question)
    found = NOT_FOUND_MESSAGE not in result["answer"]
    return {
        "answer": result["answer"] if found else NOT_FOUND_MESSAGE,
        "found": found,
        "docs": result["docs"],
        "sources": format_sources(result["docs"]) if found else [],
    }
