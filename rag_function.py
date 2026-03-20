import os
import PyPDF2
import re
import time
import streamlit as st

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import LLMChain, RetrievalQA
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.prompts import PromptTemplate
from langchain_together import ChatTogether
from dotenv import load_dotenv

load_dotenv()

# ===== Page Configuration =====
st.set_page_config(page_title="AI CV Assistant", layout="wide")

# ===== Custom Styling =====
st.markdown("""
<style>
.block-container { 
    padding-top: 2rem;
    padding-left: 2rem; 
    padding-right: 2rem; 
}
h1, h2, h3 { font-family: 'Arial', sans-serif; }
.stButton>button { background-color: #4CAF50; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ========================= Load and Build RAG Pipeline =========================
@st.cache_resource(show_spinner=False)
def load_pipeline(file):
    text = ""
    reader = PyPDF2.PdfReader(file)

    # Extract text from PDF
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    # Define CV sections
    Sections = ["EXPERIENCE", "PROFILE", "ACHIEVEMENTS", "SKILLS",
                "EDUCATION", "CERTIFICATIONS", "PROJECTS",
                "CERTIFICATES", "OBJECTIVES", "OBJECTIVE"]

    # Insert line breaks before section headers
    pattern = r"\b(" + "|".join(Sections) + r")\b"
    text_mod = re.sub(pattern, r"\n\1\n", text)

    Sections_upper = [s.upper() for s in Sections]

    current_header = "Header"
    content = ""
    documents = []

    # Split text into structured documents
    for line in text_mod.split("\n"):
        line = line.strip()
        if line.upper() in Sections_upper:
            if content:
                documents.append(Document(
                    page_content=content.strip(),
                    metadata={"Section": current_header}
                ))
            current_header = line
            content = ""
        else:
            content += line + "\n"

    if content:
        documents.append(Document(
            page_content=content.strip(),
            metadata={"Section": current_header}
        ))

    # Create embeddings and vector database
    embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')

    vdb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory="cv_vdb_chroma"
    )

    # Prompt template for QA
    prompt_template = """Use The Following Document To Answer The Question concisely in 3-4 sentences:
{context}
Question: {question}
Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Initialize LLM
    llm = ChatTogether(
        model="ServiceNow-AI/Apriel-1.6-15b-Thinker",
        temperature=0,
        api_key=os.getenv("api_key")
    )

    # Build RAG chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    combine_docs_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )

    retriever = vdb.as_retriever(search_kwargs={"k": 3})

    rag_pipeline = RetrievalQA(
        combine_documents_chain=combine_docs_chain,
        retriever=retriever
    )

    return rag_pipeline


# ========================= Session State Initialization =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

# ========================= Sidebar =========================
st.sidebar.title("AI CV Assistant")

uploaded_file = st.sidebar.file_uploader("📄 Upload CV", type="pdf")

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    st.sidebar.success("File uploaded ✅")

# Predefined quick questions
st.sidebar.markdown("### 💡 Quick Questions")

quick_questions = [
    "What are my key skills?",
    "Summarize my work experience.",
    "List my achievements.",
    "What certifications do I have?",
    "Describe my education.",

]

# ========================= Main UI =========================
st.title("🤖 CV RAG Chatbot")

chat_placeholder = st.container()

# Display chat history
for msg in st.session_state.messages:
    with chat_placeholder:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👨‍💻"):
            st.write(msg["content"])


# ========================= Question Handler =========================
def ask_question(question_text):

    # Display user message
    with chat_placeholder:
        with st.chat_message("user", avatar="👨‍💻"):
            st.write(question_text)

    st.session_state.messages.append({"role": "user", "content": question_text})

    # Load pipeline only once
    if st.session_state.rag_pipeline is None and st.session_state.uploaded_file:
        with st.spinner("⚡ Processing CV..."):
            st.session_state.rag_pipeline = load_pipeline(st.session_state.uploaded_file)

    # Generate assistant response
    if st.session_state.rag_pipeline:

        # Run model with external spinner (avoids empty bubbles)
        with st.spinner("🤖 Thinking..."):
            result = st.session_state.rag_pipeline.invoke({"query": question_text})
            answer = result["result"]

        # Display assistant response after generation
        with chat_placeholder:
            with st.chat_message("assistant", avatar="🤖"):

                message_placeholder = st.empty()

                # Typing animation
                full_response = ""
                for word in answer.split():
                    full_response += word + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)

                message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    else:
        with chat_placeholder:
            with st.chat_message("assistant", avatar="🤖"):
                st.write("Upload your CV first to get answers.")

        st.session_state.messages.append({
            "role": "assistant",
            "content": "Upload your CV first to get answers."
        })


# ========================= Quick Question Buttons =========================
for q in quick_questions:
    if st.sidebar.button(q):
        ask_question(q)

# ========================= Chat Input =========================
user_input = st.chat_input("💬 Ask about your CV...")

if user_input:
    ask_question(user_input)