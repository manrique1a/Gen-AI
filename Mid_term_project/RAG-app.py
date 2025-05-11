import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# --- Streamlit Setup ---
st.set_page_config(page_title="Applied Data Science Q&A", layout="wide")
st.title("💬 Applied Data Science RAG Assistant")

# --- Load Vectorstore ---
persist_dir = "./chroma_store"  # adjust if different
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
except Exception as e:
    st.error(f"Failed to load vectorstore: {e}")
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Placeholder Model ---
placeholder_pipeline = pipeline("text-generation", model="gpt2", max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=placeholder_pipeline)

# --- RAG Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# --- UI ---
query = st.text_input("Ask a question about the Applied Data Science program:")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain({"query": query})
        st.subheader("📌 Answer")
        st.write(result["result"])

        st.subheader("📚 Source Documents")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Source {i+1}**")
            st.code(doc.page_content[:500])

