from dotenv import load_dotenv
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile

# ================== LOAD ENV ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, ".env")

load_dotenv(env_path)
api_key = os.getenv("GOOGLE_API_KEY")

# Debug (temporary)
print("ENV PATH:", env_path)
print("API KEY:", api_key)

# Safe check
if not api_key:
    raise ValueError("❌ API key not loaded. Fix your .env file.")

# Set environment variable
os.environ["GOOGLE_API_KEY"] = api_key

# ================== LLM ==================
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)

# ================== UI ==================
st.set_page_config(page_title="Assist.AI", page_icon="🤖")
st.title("🤖 Assist.AI")

# ================== FILE UPLOAD ==================
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# ================== LOAD PDF ==================
@st.cache_data
def load_pdf_content(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    return "\n\n".join([p.page_content for p in pages])

pdf_text = None
if uploaded_file:
    pdf_text = load_pdf_content(uploaded_file)

# ================== SESSION ==================
if "history" not in st.session_state:
    st.session_state.history = []

# ================== QA FUNCTION ==================
def answer_question(context, question):
    messages = [
        ("system", "Answer using this document:\n\n" + context),
        ("user", question)
    ]
    return llm.invoke(messages).content

# ================== CHAT UI ==================
if pdf_text is None:
    st.info("Upload a PDF to start.")
else:
    for msg in st.session_state.history:
        with st.chat_message("user"):
            st.write(msg["q"])
        with st.chat_message("assistant"):
            st.write(msg["a"])

    prompt = st.chat_input("Ask something...")

    if prompt:
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ans = answer_question(pdf_text, prompt)
                st.write(ans)

        st.session_state.history.append({"q": prompt, "a": ans})