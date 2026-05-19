import streamlit as st
import tempfile
from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Optional PDF Support
try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    PyPDFLoader = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None


# -------------------- Helper Functions --------------------

def load_pdf_from_bytes(file_bytes: bytes) -> List[Document]:
    """Robust PDF loader with fallback support"""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp.flush()  # ✅ IMPORTANT FIX
        tmp_path = tmp.name

    docs = []

    # --- Try PyPDFLoader ---
    if PyPDFLoader is not None:
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        except Exception as e:
            print(f"❌ PyPDFLoader failed: {e}")

    # --- Fallback: pdfplumber ---
    if not docs and pdfplumber is not None:
        try:
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.append(Document(page_content=text))
        except Exception as e:
            print(f"❌ pdfplumber failed: {e}")

    # --- Final fallback ---
    if not docs:
        docs = [Document(page_content="No readable content found in PDF.")]

    return docs


def load_url_documents(urls: List[str]) -> List[Document]:
    docs = [WebBaseLoader(url).load() for url in urls]
    return [item for sublist in docs for item in sublist]


def build_vectorstore(doc_splits, embedding_model_name="embeddinggemma:300m"):
    embedding = OllamaEmbeddings(model=embedding_model_name)
    return FAISS.from_documents(documents=doc_splits, embedding=embedding)


# -------------------- RAG Core --------------------

PROMPT_TEMPLATE = """You are a strict assistant for question-answering tasks.
Use the following documents to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise (max 2 sentences):

Question: {question}
Documents: {documents}
Answer:"""

rag_prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["question", "documents"]
)


class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question):
        documents = self.retriever.invoke(question)
        doc_texts = "\n".join([doc.page_content for doc in documents])
        return self.rag_chain.invoke({
            "question": question,
            "documents": doc_texts
        })


# -------------------- Page Config --------------------

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -------------------- Session State --------------------

if "history" not in st.session_state:
    st.session_state.history = []
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "rag_app" not in st.session_state:
    st.session_state.rag_app = None


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:

    st.title("📚 RAG Chatbot")
    st.caption("Fully local · Powered by FAISS + Ollama")

    st.divider()

    option = st.radio(
        "Choose input type:",
        ("📄 Upload PDF", "🌐 Enter URLs"),
        label_visibility="collapsed"
    )

    st.divider()

    docs_list = []

    if option == "📄 Upload PDF":
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            st.success(f"✅ File ready: **{uploaded_file.name}**")

    elif option == "🌐 Enter URLs":
        urls_input = st.text_area(
            "One URL per line",
            placeholder="https://example.com",
            height=120,
            label_visibility="collapsed"
        )

    st.divider()

    process_button = st.button(
        "⚡ Ingest / Rebuild Index",
        type="primary",
        use_container_width=True
    )

    if process_button:
        if option == "📄 Upload PDF":
            if not uploaded_file:
                st.warning("⚠️ Upload a PDF first.")
            else:
                with st.spinner("Processing PDF..."):
                    docs_list = load_pdf_from_bytes(uploaded_file.read())

        elif option == "🌐 Enter URLs":
            if not urls_input.strip():
                st.warning("⚠️ Enter at least one URL.")
            else:
                urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
                with st.spinner("Loading URLs..."):
                    docs_list = load_url_documents(urls)

        if docs_list:
            with st.status("🔨 Building Knowledge Base...", expanded=True) as status:

                splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=500,
                    chunk_overlap=150
                )

                doc_splits = splitter.split_documents(docs_list)

                try:
                    vs = build_vectorstore(doc_splits)
                    retriever = vs.as_retriever(search_kwargs={"k": 4})

                    llm = OllamaLLM(
                        model="llama3.2:1b",
                        temperature=0
                    )

                    rag_chain = rag_prompt | llm | StrOutputParser()

                    st.session_state.rag_app = RAGApplication(
                        retriever,
                        rag_chain
                    )
                    st.session_state.ingested = True

                    status.update(
                        label="✅ Ready!",
                        state="complete",
                        expanded=False
                    )

                    st.toast("🎉 Knowledge base built!")

                except Exception as e:
                    status.update(label=f"❌ Failed: {e}", state="error")
                    st.error(str(e))

    st.divider()

    keep_history = st.checkbox("Keep conversation history", value=True)

    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []
        st.rerun()

    st.divider()

    if st.session_state.ingested:
        st.success("🟢 Ready")
    else:
        st.info("⚪ No data yet")


# ================================================================
# MAIN CHAT
# ================================================================

st.header("💬 DocuMIND")
st.caption("Ask questions about your documents")

st.divider()

if not st.session_state.ingested:
    st.info("👈 Upload PDF or URLs and click Ingest")

# Chat History
for msg in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(msg["question"])
    with st.chat_message("assistant"):
        st.markdown(msg["answer"])

# Input
prompt = st.chat_input(
    "Ask something...",
    disabled=not st.session_state.ingested
)

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.rag_app.run(prompt)
                st.markdown(answer)

                if keep_history:
                    st.session_state.history.append({
                        "question": prompt,
                        "answer": answer
                    })

            except Exception as e:
                st.error(str(e))