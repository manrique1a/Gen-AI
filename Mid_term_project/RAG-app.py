import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# --- Load secrets from Streamlit (or .env if running locally) ---
load_dotenv()  # Optional for local dev
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Streamlit Setup ---
st.set_page_config(page_title="Applied Data Science Q&A", layout="wide")
st.title("💬 The University of Chicago's Master's in Applied Data Science Chatbot")

# --- Load Vectorstore ---
persist_dir = "chroma_store"
embedding_model = HuggingFaceEmbeddings(model_name="fine_tuned_qa_embedding_model")

try:
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
except Exception as e:
    st.error(f"Failed to load vectorstore: {e}")
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- LLM Setup (GPT-3.5 Turbo) ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

# --- RAG Chain ---
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




