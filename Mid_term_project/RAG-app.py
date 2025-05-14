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
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load secrets from environment or .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Streamlit Setup ---
st.set_page_config(page_title="UChicago ADS Chatbot", layout="wide")
st.title("💬 The University of Chicago Master's in Applied Data Science Chatbot")

# load .txt file
try:
    text_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "uchicago_datascience_cleaned.txt"))
    loader = TextLoader(text_path)
    raw_documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(raw_documents)
except Exception as e:
    st.warning(f"Loaded app but failed to read .txt file for reference: {e}")

# --- Load fine-tuned embedding model ---
os.environ["HF_HUB_OFFLINE"] = "1"

# Get the absolute path to the model directory
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "app_model"))

# Load the embedding model from local fine-tuned directory
embedding_model = HuggingFaceEmbeddings(model_name=model_path)

# --- Load persisted vectorstore ---
persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "chroma_v3"))

try:
    if not os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        # If no vectorstore exists, create it from the docs
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
    else:
        # Load existing vectorstore
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
except Exception as e:
    st.error(f"Failed to load or build vectorstore: {e}")
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

