import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# --- Load API key ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Streamlit Setup ---
st.set_page_config(page_title="UChicago ADS Chatbot (Fine-Tuned)", layout="wide")
st.title("💬 UChicago MS in Applied Data Science Chatbot")

# --- Load fine-tuned embedding model ---
from langchain_community.embeddings import HuggingFaceEmbeddings

os.environ["HF_HUB_OFFLINE"] = "1"
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "fine_tuned_qa_embedding_model"))
embedding_model = HuggingFaceEmbeddings(model_name=model_path)

# --- Load Chroma vector store ---
persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "chroma_v2"))
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

# --- Retriever and Generator ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

# --- RAG Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# --- Streamlit UI ---
query = st.text_input("Ask a question about the Master's in Applied Data Science program at the University of Chicago:")

if query:
    with st.spinner("Thinking... 🤔"):
        result = qa_chain({"query": query})

        st.subheader("📌 Answer")
        st.write(result["result"])

