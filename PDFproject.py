import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_classic.chains import RetrievalQA

# 🔥 LOCAL LLM (NO API)
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline


# -------------------------------
# 🔑 PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="PDF Chat (FREE Local AI)", layout="wide")
st.title("📄 Chat with your PDF (FREE AI 🤖)")

# -------------------------------
# 💬 CHAT MEMORY
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# 📂 FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ PDF uploaded successfully!")

    # -------------------------------
    # 📄 LOAD PDF
    # -------------------------------
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # -------------------------------
    # ✂️ SPLIT TEXT
    # -------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    st.info(f"📊 Total chunks: {len(chunks)}")

    # -------------------------------
    # 🧠 EMBEDDINGS (LOCAL)
    # -------------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # -------------------------------
    # 🗄 VECTOR STORE
    # -------------------------------
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        location=":memory:"
    )

    retriever = vector_store.as_retriever()

    # -------------------------------
    # 🤖 LOCAL LLM (NO API 🔥)
    # -------------------------------
    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=512
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # -------------------------------
    # 🔎 QA CHAIN
    # -------------------------------
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    # -------------------------------
    # 💬 CHAT UI
    # -------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask something about your PDF...")

    if user_input:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤔"):
                response = qa_chain.run(user_input)
                st.write(response)

        # Save response
        st.session_state.messages.append({"role": "assistant", "content": response})