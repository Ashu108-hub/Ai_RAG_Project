# 🤖 AI RAG ChatBot

An AI-powered chatbot that allows users to upload PDF files and ask questions directly from the document content using Retrieval-Augmented Generation (RAG).

This project contains two different implementations:
- Gemini API based chatbot
- LLaMA based chatbot

Built using Python, Streamlit, LangChain, and modern LLM technologies.

---

## 🚀 Features

- Upload PDF documents
- Ask questions from uploaded PDFs
- Context-aware AI responses
- Interactive Streamlit interface
- Gemini API integration
- LLaMA model implementation
- Simple and clean UI

---

## 🧠 Models Used

### 1. Gemini Model

**File:** `gemini_rag_chatbot.py`

Uses Google Gemini API for generating responses from document context.

Run command:

```bash
streamlit run gemini_rag_chatbot.py
```

---

### 2. LLaMA Model

**File:** `Local_rag_chatbot.py`

Uses LLaMA-based architecture for document question answering.

Run command:

```bash
streamlit run Local_rag_chatbot.py
```

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini API
- LLaMA
- PyPDF
- dotenv

---

## 📂 Project Structure

```bash
Ai_RAG_ChatBot/
│
├── gemini_rag_chatbot.py
├── Local_rag_chatbot.py
├── README.md
├── .gitignore
└── requirements.txt
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Ashu108-hub/Ai_RAG_ChatBot.git
```

Go to the project folder:

```bash
cd Ai_RAG_ChatBot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory and add your API key:

```env
GOOGLE_API_KEY=your_api_key
```

---

## ▶️ Run the Project

For Gemini version:

```bash
streamlit run gemini_rag_chatbot.py
```

For LLaMA version:

```bash
streamlit run Local_rag_chatbot.py
```

---

## 💡 Future Improvements

- Chat memory support
- Multiple PDF handling
- Vector database integration
- Improved UI design
- Cloud deployment

---

## 👨‍💻 Developer

Ashutosh

Interested in AI, Generative AI, Python Development, and LLM-based applications.
