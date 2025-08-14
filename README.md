# 📚 RAG Study Assistant

A study assistant chatbot using **LangChain**, **Gemini (ChatGoogleGenerativeAI)**, and **HuggingFace embeddings**.  
Supports multi-file uploads (PDF, DOCX, TXT), document summarization, and automatic 10-question quizzes.

---

## Features

- Chatbot powered by Gemini LLM (`gemini-2.5-flash`)  
- Document embeddings using `sentence-transformers all-MiniLM-L6-v2`  
- Vector storage with Chroma  
- Multi-file upload: PDF, DOCX, TXT  
- Generate concise summaries from documents  
- Generate 10-question multiple-choice quizzes (MCQs)  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/chittinarora/Mychatbot.git
cd Mychatbot
