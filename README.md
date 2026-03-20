# 🤖 AI CV Assistant (RAG Chatbot)

An intelligent AI-powered chatbot that allows users to upload their CV (PDF) and interact with it using natural language questions.

Built using **Retrieval-Augmented Generation (RAG)**, this system extracts structured information from resumes and provides accurate, concise answers.

---

## Features

* Upload CV in PDF format
* Automatic text extraction and section detection
* Smart semantic search using embeddings
* AI-powered question answering
* Interactive chat interface (Streamlit)
* Fast responses with caching
* Predefined quick questions for easy access

---

## How It Works

1. **PDF Processing**

   * Extracts text using `PyPDF2`
   * Detects sections like:

     * Experience
     * Skills
     * Education
     * Projects

2. **Text Structuring**

   * Splits CV into meaningful chunks
   * Each section stored as a document

3. **Embeddings**

   * Uses `BAAI/bge-m3` model for semantic understanding

4. **Vector Database**

   * Stores embeddings using `ChromaDB`

5. **RAG Pipeline**

   * Retrieves relevant CV sections
   * Sends them to LLM for answering

6. **LLM**

   * Uses `Together AI` (Apriel model)
   * Generates concise answers (3–4 sentences)

---

## Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **LLM:** Together AI
* **Embeddings:** HuggingFace (`bge-m3`)
* **Vector DB:** ChromaDB
* **Framework:** LangChain

---

## Installation

```bash
git clone https://github.com/your-username/ai-cv-assistant.git
cd ai-cv-assistant
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file and add:

```env
api_key=YOUR_TOGETHER_API_KEY
```

---

## Example Questions

* What are my key skills?
* Summarize my work experience.
* What certifications do I have?
* Describe my education.
* What are my main achievements?

---

## Demo

> Upload your CV and start chatting instantly!

---

## 📂 Project Structure

```
├── app.py
├── requirements.txt
├── .env
├── cv_vdb_chroma/
└── README.md
```

---

## Performance Optimizations

* Uses `@st.cache_resource` to avoid rebuilding pipeline
* Efficient document chunking
* Top-K retrieval for fast responses

---

## Future Improvements
* Multilingual support (Arabic/English)
* Resume evaluation and feedback 
* Export responses as a report
  
---


## 👨‍💻 Author

**Mohamed Ahmed Hemdan**

AI Engineer | Python Instructor | Teaching Assistant
