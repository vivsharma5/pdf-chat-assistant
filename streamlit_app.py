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
    # Get matches to check similarity
    docs_with_scores = vector_db.similarity_search_with_score(query, k=3)
    
    # If the match is too weak (distance is too high), trigger customer service
    if not docs_with_scores or docs_with_scores[0][1] > 0.6:
        return "NOT_FOUND", []
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(query), docs_with_scores

# --- 4. THE INTERFACE ---
query = st.text_input("What would you like to know from the guide?")

if query:
    with st.spinner("Searching the guide..."):
        answer, sources = get_answer(query)
        
        if answer == "NOT_FOUND":
            st.warning("I couldn't find that in the guide. Would you like to contact Vivek?")
            with st.form("contact_form"):
                msg = st.text_area("Your message:")
                if st.form_submit_button("Send to Customer Service"):
                    # For now, we print to console. 
                    # In a real app, you'd use smtplib here!
                    st.success("Thanks! Vivek has been notified.")
        else:
            st.markdown(f"### Answer\n{answer}")
            st.markdown("### Sources")
            for doc, score in sources:
                st.caption(f"- Page {doc.metadata.get('page')}: {doc.page_content[:100]}...")
