import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load secrets from environment or .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Streamlit Setup ---
st.set_page_config(page_title="UChicago ADS Chatbot", layout="wide")
st.title("💬 The University of Chicago Master's in Applied Data Science Chatbot")

# --- Load fine-tuned embedding model ---
os.environ["HF_HUB_OFFLINE"] = "1"

# Get the absolute path to the model directory
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "fine_tuned_qa_embedding_model"))

# Load the embedding model from local fine-tuned directory
embedding_model = HuggingFaceEmbeddings(model_name=model_path)

# --- Load persisted vectorstore ---
persist_dir = "chroma_store"
try:
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
except Exception as e:
    st.error(f"Failed to load vectorstore: {e}")
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- LLM setup ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

# --- RAG Pipeline ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# --- UI ---
query = st.text_input("Ask a question about the Master's in Applied Data Science program at the University of Chicago:")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain({"query": query})
        st.subheader("📌 Answer")
        st.write(result["result"])

        st.subheader("📚 Source Documents")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Source {i+1}**")
            st.code(doc.page_content[:500])





