__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="LLM Guide Assistant", page_icon="🤖")
st.title("📚 Chat with the LLM Guide")

# Use Streamlit Secrets for the API Key (set this in Streamlit dashboard later)
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please add your GOOGLE_API_KEY to Streamlit Secrets!")
    st.stop()

# --- 2. THE BRAIN (Loading & Indexing) ---
@st.cache_resource # This keeps the DB in memory so it doesn't reload every click
def prepare_vector_db():
    pdf_path = "compact-guide-to-large-language-models.pdf"
    persist_dir = "./chroma_db"
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    
    # If DB doesn't exist locally, create it
    if not os.path.exists(persist_dir):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(docs)
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=persist_dir
        )
    else:
        vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    return vector_db

vector_db = prepare_vector_db()

# --- 3. THE LOGIC (RAG Chain) ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
template = """Answer strictly based on the context provided. 
If the answer isn't there, say "NOT_FOUND".

Context: {context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

def get_answer(query):
    docs_with_scores = vector_db.similarity_search_with_score(query, k=5)
    
    # Change 0.6 to 1.0 (Chroma distance: higher is more lenient)
    if not docs_with_scores or docs_with_scores[0][1] > 1.0:
        return "NOT_FOUND", []
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(query), docs_with_scores

# --- 4. THE INTERFACE ---
if query:
    with st.spinner("Searching..."):
        answer, sources = get_answer(query)
        
        if answer == "NOT_FOUND" or "NOT_FOUND" in answer:
            st.warning("I couldn't find a direct match in the guide.")
            
            # Create two columns for the buttons
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("✉️ Contact Vivek"):
                    with st.form("service_form"):
                        msg = st.text_area("Your message:")
                        if st.form_submit_button("Submit"):
                            st.success("Sent!")
            with col2:
                # This button just refreshes the page to clear the state
                if st.button("🔄 Ask Another Question"):
                    st.rerun()
        else:
            st.markdown(f"### Answer\n{answer}")
            # Show sources...
