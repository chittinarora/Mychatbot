"""

app.py - RAG Study Assistant using:

- ChatGoogleGenerativeAI (Gemini)

- HuggingFaceEmbeddings (sentence-transformers all-MiniLM-L6-v2)

- Chroma vectorstore

- Multi-file upload: PDF, DOCX, TXT

- Summarize & Generate Quiz (10 MCQs)

"""
 
import os

import shutil

import io

from dotenv import load_dotenv

import streamlit as st

import numpy as np

from typing import List
 
# LangChain imports

from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
 
# --------------- Config & paths ---------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
 
DATA_DIR = "./data"

PERSIST_DIR = "./chroma_db"
 
os.makedirs(DATA_DIR, exist_ok=True)
 
# --------------- Safe removal of chroma db (Windows-safe) ---------------

def safe_delete_chroma(persist_dir: str):

    if not os.path.exists(persist_dir):

        return

    try:

        shutil.rmtree(persist_dir)

    except PermissionError:

        # If it is locked, try to gently close/delete via Chroma API

        try:

            vs = Chroma(persist_directory=persist_dir, embedding_function=None)

            # some Chroma wrappers may have delete_collection() or similar

            try:

                vs.delete_collection()

            except Exception:

                pass

            del vs

            shutil.rmtree(persist_dir)

        except Exception as e:

            st.warning(f"Could not fully delete {persist_dir}: {e}. Continuing without deletion.")
 
# NOTE: we delete only on cold start if user explicitly requests; do NOT auto-delete every page rerun.

# We will provide a button in UI to "Reset DB" which calls this function safely.
 
# --------------- Embeddings & Vector Store (session-managed) ---------------

def get_embeddings():

    # using sentence-transformers model (all-MiniLM-L6-v2)

    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
def init_vector_store(persist_dir: str, embeddings):

    # create Chroma instance (it will create directory if missing)

    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
 
# Keep in session_state to avoid reinitializing between reruns

if "embeddings" not in st.session_state:

    st.session_state["embeddings"] = get_embeddings()
 
if "vector_store" not in st.session_state:

    # create directory but don't delete by default

    st.session_state["vector_store"] = init_vector_store(PERSIST_DIR, st.session_state["embeddings"])
 
vector_store: Chroma = st.session_state["vector_store"]

embeddings = st.session_state["embeddings"]
 
# --------------- LLM setup ---------------

# Use ChatGoogleGenerativeAI from langchain_google_genai

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.2)
 
# Simple prompt template: we will append context retrieved from vector DB

system_prompt = "You are a helpful, concise study assistant. Answer using only the provided context. If the answer is not present, say you don't know."

prompt = ChatPromptTemplate.from_messages([

    ("system", system_prompt),

    ("user", "Question: {question}")

])

output_parser = StrOutputParser()
 
# --------------- Utilities: load files, index, retrieve ---------------

from langchain.text_splitter import RecursiveCharacterTextSplitter
 
def save_uploaded_file(file) -> str:

    target_path = os.path.join(DATA_DIR, file.name)

    with open(target_path, "wb") as f:

        f.write(file.getbuffer())

    return target_path
 
def load_documents_from_files(filepaths: List[str]):

    docs = []

    for path in filepaths:

        lower = path.lower()

        try:

            if lower.endswith(".pdf"):

                loader = PyPDFLoader(path)

            elif lower.endswith(".txt"):

                loader = TextLoader(path, encoding="utf-8")

            elif lower.endswith(".docx"):

                loader = Docx2txtLoader(path)

            else:

                st.warning(f"Skipping unsupported file: {path}")

                continue

            docs.extend(loader.load())

        except Exception as e:

            st.warning(f"Failed loading {path}: {e}")

    return docs
 
def index_files_and_persist(filepaths: List[str]):

    docs = load_documents_from_files(filepaths)

    if not docs:

        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    chunks = splitter.split_documents(docs)

    vector_store.add_documents(chunks)

    vector_store.persist()

    return len(chunks)
 
def retrieve_context(query: str, k: int = 4):

    try:

        results = vector_store.similarity_search(query, k=k)

    except Exception:

        # fallback: empty list

        results = []

    context = "\n\n".join([d.page_content for d in results])

    return context, results
 
# --------------- LLM call helpers ---------------

def llm_invoke_with_context(question: str, context: str):

    # Build prompt (user message includes question + context)

    user_text = f"Question: {question}\n\nContext:\n{context}\n\nAnswer concisely."

    # As examples in LangChain docs, we can call llm.invoke with messages list

    messages = [

        ("system", system_prompt),

        ("human", user_text)

    ]

    try:

        resp = llm.invoke(messages)

        # resp may be an AIMessage-like object or have .content

        if hasattr(resp, "content"):

            return resp.content

        # some variants return string

        return str(resp)

    except Exception as e:

        st.error(f"LLM invocation failed: {e}")

        return "Error: LLM invocation failed."
 
# --------------- Streamlit UI ---------------

st.set_page_config(page_title="📚 RAG Study Assistant", layout="wide")
 
# Basic CSS for color

st.markdown(

    """
<style>

      .stApp { background: linear-gradient(180deg,#ffffff 0%, #f7fbff 100%); }

      .big-title {font-size:30px; font-weight:700;}

      .accent { color: #0b57d0; font-weight:700; }

      .card { background: white; padding: 16px; border-radius:12px; box-shadow: 0 6px 18px rgba(13,38,76,0.08); }

      .small-muted { color: #6b7280; font-size:12px; }

      .btn-blue > button { background-color: #0b57d0; color: white; font-weight:600; }
</style>

    """, unsafe_allow_html=True

)
 
st.markdown("<div class='big-title'>📚 RAG Study Assistant <span class='accent'> — Gemini + Sentence-Transformers</span></div>", unsafe_allow_html=True)

st.write("Upload PDFs / DOCX / TXT, index them for this session, ask questions, summarize, or generate a 10-question multiple-choice quiz.")
 
# Sidebar controls

with st.sidebar:

    st.header("Session Controls")

    st.write("Upload files to index for this session (PDF / DOCX / TXT).")

    uploaded_files = st.file_uploader("Upload files", type=["pdf","docx","txt"], accept_multiple_files=True)
 
    if st.button("Reset / Delete chroma_db"):

        safe_delete_chroma(PERSIST_DIR)

        # re-initialize vector store in session

        st.session_state["vector_store"] = init_vector_store(PERSIST_DIR, st.session_state["embeddings"])

        st.success("Attempted reset. If DB was in use, you may need to close other Python processes.")
 
    st.markdown("---")

    st.markdown("**Notes**")

    st.markdown("- DB persists on disk until reset. Use Reset button to try to delete it.")

    st.markdown("- Each Streamlit run uses indexed files present in `./data` + `chroma_db` unless reset.")
 
# Main columns

col_main, col_right = st.columns([3,1])
 
with col_main:

    # Upload handling

    if uploaded_files:

        # save to disk and index

        saved_paths = []

        for f in uploaded_files:

            path = save_uploaded_file(f)

            saved_paths.append(path)

        with st.spinner("Indexing uploaded files..."):

            chunks_added = index_files_and_persist(saved_paths)

        st.success(f"Indexed {len(saved_paths)} file(s) → {chunks_added} chunks added.")
 
    st.markdown("### Ask a question")

    question = st.text_input("Enter question to ask about your uploaded documents", key="question_input")
 
    col_a, col_b, col_c = st.columns([1,1,1])

    with col_a:

        if st.button("Get Answer", key="answer_btn"):

            if not question or question.strip() == "":

                st.warning("Please type a question above.")

            else:

                with st.spinner("Searching documents and getting answer..."):

                    ctx, docs = retrieve_context(question, k=4)

                    if not ctx.strip():

                        st.info("No indexed content found — upload files first.")

                    else:

                        answer = llm_invoke_with_context(question, ctx)

                        st.markdown("#### Answer")

                        st.write(answer)

                        # show sources

                        st.markdown("**Sources / retrieved snippets:**")

                        for i, d in enumerate(docs):

                            st.markdown(f"- **Chunk {i+1}** (source: {d.metadata.get('source','unknown')}):")

                            st.write(d.page_content[:600] + ("..." if len(d.page_content) > 600 else ""))
 
    with col_b:

        if st.button("Summarize Documents", key="summarize_btn"):

            with st.spinner("Generating summary from indexed documents..."):

                # we will construct a summary prompt that asks to summarize retrieved context

                ctx, _ = retrieve_context("summarize the documents", k=8)

                if not ctx.strip():

                    st.info("No indexed content found — upload files first.")

                else:

                    summary = llm_invoke_with_context("Please produce a concise summary of the uploaded documents.", ctx)

                    st.markdown("#### Summary")

                    st.write(summary)
 
    with col_c:

        if st.button("Generate 10-question Quiz (MCQ)", key="quiz_btn"):

            with st.spinner("Generating quiz from documents..."):

                ctx, _ = retrieve_context("generate quiz", k=8)

                if not ctx.strip():

                    st.info("No indexed content found — upload files first.")

                else:

                    quiz_prompt = (

                        "From the following content, generate 10 multiple-choice questions. "

                        "Each question should have 4 options labeled A-D and indicate the correct option. "

                        "Keep questions clear and focused."

                    )

                    quiz_text = llm_invoke_with_context(quiz_prompt, ctx)

                    st.markdown("#### Quiz (10 MCQs)")

                    st.write(quiz_text)
 
with col_right:

    st.markdown("## Session Info")

    st.write(f"- Embeddings model: `all-MiniLM-L6-v2`")

    st.write(f"- Vector DB path: `{PERSIST_DIR}`")

    st.write(f"- Indexed chunks (approx): not exact; use logs to inspect")

    st.markdown("---")

    st.markdown("### Files in `./data`")

    try:

        files = os.listdir(DATA_DIR)

        if files:

            for f in files:

                st.write(f"- {f}")

        else:

            st.write("No files uploaded yet.")

    except Exception as e:

        st.write(f"Error listing files: {e}")
 
st.markdown("---")

st.markdown("**Tips:** Upload documents, use the Summarize button to get a short summary, or Generate Quiz to create practice MCQs. Use Reset to delete the stored Chroma DB. ")
 