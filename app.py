"""Chat interface.  Run:  streamlit run app.py"""
import streamlit as st

from src.config import BOOK_AUTHOR, BOOK_TITLE
from src.rag_chain import ask, build_rag_chain

st.set_page_config(page_title=f"{BOOK_TITLE} চ্যাটবট", page_icon="📖")
st.title(f"📖 {BOOK_TITLE}")
st.caption(f"{BOOK_AUTHOR}-এর এই বই নিয়ে প্রশ্ন করুন। উত্তর শুধু বই থেকেই দেওয়া হবে।")


@st.cache_resource(show_spinner="বইয়ের ডেটাবেস লোড হচ্ছে...")
def load_chain():
    return build_rag_chain()


def show_message(role, text, sources=()):
    with st.chat_message(role):
        st.markdown(text)
        if sources:
            st.markdown("**উৎস:**\n" + "\n".join(f"- {source}" for source in sources))


try:
    chain = load_chain()
except (FileNotFoundError, RuntimeError) as error:
    st.error(str(error))
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for message in st.session_state.history:
    show_message(**message)

if question := st.chat_input("বই সম্পর্কে একটি প্রশ্ন লিখুন"):
    show_message("user", question)
    try:
        with st.spinner("উত্তর খোঁজা হচ্ছে..."):
            result = ask(chain, question)
    except Exception as error:  # network problems, wrong API key, retired model...
        st.error(f"উত্তর আনা যায়নি: {error}")
        st.stop()
    show_message("assistant", result["answer"], result["sources"])
    st.session_state.history += [
        {"role": "user", "text": question},
        {"role": "assistant", "text": result["answer"], "sources": result["sources"]},
    ]
