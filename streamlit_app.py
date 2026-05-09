__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import requests
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
import requests # Add this to the top of your file with other imports

# Function to create a GitHub Issue
def log_question_to_github(user_msg, original_query):
    repo = "vivsharma5/pdf-assistant" # Use your actual username/repo name
    token = st.secrets["GITHUB_TOKEN"]
    url = f"https://api.github.com/repos/{repo}/issues"
    
    header = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    data = {
        "title": f"New Inquiry: {original_query[:30]}...",
        "body": f"**User Question:** {original_query}\n\n**Message for Vivek:** {user_msg}"
    }
    
    response = requests.post(url, headers=header, json=data)
    return response.status_code == 201

# --- 4. THE INTERFACE ---
# FIXED: We define 'query' here before using it!
query = st.text_input("What would you like to know from the guide?", placeholder="e.g., What is an LLM?")

if query:
    with st.spinner("Searching..."):
        answer, sources = get_answer(query)
        
        if answer == "NOT_FOUND" or "NOT_FOUND" in answer:
            st.warning("I couldn't find a direct match in the guide.")
            
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("✉️ Notify Vivek"):
                    with st.form("service_form"):
                        msg = st.text_area("Leave a message for Vivek:")
                        if st.form_submit_button("Submit"):
                            # This sends the data to your GitHub Issues
                            if log_question_to_github(msg, query):
                                st.success("Vivek has been notified via GitHub!")
                            else:
                                st.error("Failed to log. Please try again.")
            with col2:
                if st.button("🔄 Ask Another Question"):
                    st.rerun()
        else:
            st.markdown(f"### Answer\n{answer}")
            st.markdown("---")
            st.markdown("**Sources:**")
            for doc, score in sources:
                st.caption(f"📍 Page {doc.metadata.get('page')}: {doc.page_content[:150]}...")
            
            if st.button("Ask New Question"):
                st.rerun()
