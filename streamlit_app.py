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
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold

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
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.3,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }
)
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
# --- 4. THE INTERFACE ---

# 1. Define the clear function at the top of this section
def clear_text():
    st.session_state["user_query"] = ""

# 2. Setup the text input with a KEY
# The 'key' connects this box to the clear_text function
query = st.text_input(
    "What would you like to know from the guide?", 
    key="user_query", 
    placeholder="e.g., What is an LLM?"
)

# --- 4. THE INTERFACE ---

# Function to create a GitHub Issue with Error Reporting
def log_to_github(query, user_msg):
    repo = "vivsharma5/pdf-assistant"
    token = st.secrets["GITHUB_TOKEN"]
    url = f"https://api.github.com/repos/{repo}/issues"
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {
        "title": f"Inquiry: {query[:30]}...",
        "body": f"**Question:** {query}\n\n**User Message:** {user_msg}"
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    # This will show you EXACTLY why it fails (e.g., 401 or 404)
    if response.status_code != 201:
        st.error(f"GitHub Error {response.status_code}: {response.text}")
        return False
    return True

# --- LOGIC ---
if query:
    try:
        with st.spinner("Searching..."):
            answer, sources = get_answer(query)
            
            if answer == "NOT_FOUND" or "NOT_FOUND" in answer:
                st.warning("I couldn't find that in the guide.")
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.popover("📩 Notify Vivek"):
                        # Wrapping in a form ensures the 'u_msg' is sent correctly
                        with st.form("github_form"):
                            u_msg = st.text_area("Your message:")
                            submit_github = st.form_submit_button("Send to GitHub")
                            
                            if submit_github:
                                if u_msg:
                                    if log_to_github(query, u_msg):
                                        st.success("✅ Logged to GitHub Issues!")
                                else:
                                    st.warning("Please type a message first.")
                with col2:
                    st.button("🔄 Ask Another", on_click=clear_text)
            else:
                st.markdown(f"### Answer\n{answer}")
                st.button("➕ Ask New Question", on_click=clear_text)
                
    except Exception as e:
        st.error(f"The AI is having a moment: {str(e)[:100]}")I key or safety settings.")
