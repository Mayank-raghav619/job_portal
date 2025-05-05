import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pymongo import MongoClient
import http.client
import json
import os
import re
import asyncio
import nest_asyncio
import warnings
from datetime import datetime

# Load environment variables
load_dotenv()

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress torch warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# Apply nest_asyncio for async compatibility
nest_asyncio.apply()

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["job_assistant"]
logs_collection = db["user_logs"]

# Initialize Groq LLM
def init_groq_model():
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return ChatGroq(
        groq_api_key=groq_api_key, model_name="llama-3-3-70b-versatile", temperature=0.2
    )

llm_groq = init_groq_model()

def init_async():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# PDF Text Extractor
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

# Text Splitter
def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=3000, chunk_overlap=200, length_function=len
    )
    return splitter.split_text(text)

# Vectorstore Generator
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embeddings)

# Conversation Chain
def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

# Feature Extractor
def extract_job_features(text):
    skills = re.findall(r'\b(Java|Python|Data Science|Machine Learning|Deep Learning|Software Engineer|Data Engineer|AI|NLP|C\+\+|SQL|TensorFlow|Keras)\b', text, re.IGNORECASE)
    titles = re.findall(r'\b(Engineer|Data Scientist|Developer|Manager|Analyst|Consultant)\b', text, re.IGNORECASE)
    return list(set(skills + titles)) or ["General"]

# Job Recommender
def get_job_recommendations(features):
    host = "jooble.org"
    jooble_api_key = os.getenv("JOOBLE_API_KEY")
    connection = http.client.HTTPConnection(host)
    headers = {"Content-type": "application/json"}
    body = json.dumps({"keywords": ", ".join(features), "location": "Remote"})

    try:
        connection.request("POST", f"/api/{jooble_api_key}", body, headers)
        response = connection.getresponse()
        jobs = json.loads(response.read()).get("jobs", [])
        return [{
            "title": job.get("title", "Job Title"),
            "company": job.get("company", "Company"),
            "link": job.get("link", "#"),
            "description": clean_job_description(job.get("snippet", "No Description"))
        } for job in jobs]
    except Exception as e:
        st.error(f"Error fetching job data: {e}")
        return []

# Job Description Cleaner
def clean_job_description(desc):
    desc = re.sub(r'&nbsp;|&#39;|<[^>]+>', '', desc)
    keywords = re.findall(r'\b(Python|Java|TensorFlow|Keras|Machine Learning|AI|NLP|Deep Learning|Engineer|Data Scientist|Developer|Analyst)\b', desc, re.IGNORECASE)
    for word in keywords:
        desc = re.sub(rf'\b{word}\b', f"**{word}**", desc, flags=re.IGNORECASE)
    return desc

# Chatbot Handler
def handle_userinput(question):
    if question:
        try:
            response = st.session_state.conversation.invoke({"question": question})
            answer = response.get('answer', 'No response')
            st.write(answer)

            logs_collection.insert_one({
                "query": question,
                "response": answer,
                "timestamp": datetime.now()
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Main App
def main():
    st.set_page_config(page_title="Job Assistant Chatbot", page_icon="💼")
    st.title("💬 Smart Job Assistant")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "job_recommendations" not in st.session_state:
        st.session_state.job_recommendations = []

    init_async()
    tab = st.sidebar.radio("Choose Tab", ["Chatbot", "Job Recommendations", "Logs Viewer"])

    if tab == "Chatbot":
        user_input = st.text_input("Ask something about your resume:")
        if user_input:
            handle_userinput(user_input)

        st.sidebar.subheader("Upload Resume")
        pdfs = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if st.sidebar.button("Process"):
            if pdfs:
                with st.spinner("Processing documents..."):
                    text = get_pdf_text(pdfs)
                    chunks = get_text_chunks(text)
                    vectorstore = get_vectorstore(chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    features = extract_job_features(text)
                    st.session_state.job_recommendations = get_job_recommendations(features)
                    st.success("Processing complete.")
            else:
                st.warning("Upload at least one PDF.")

    elif tab == "Job Recommendations":
        st.header("Recommended Jobs 🔍")
        if st.session_state.job_recommendations:
            for job in st.session_state.job_recommendations:
                st.markdown(f"**[{job['title']}]({job['link']})** at **{job['company']}**")
                st.markdown(f"{job['description']}", unsafe_allow_html=True)
        else:
            st.info("Upload a resume first to get recommendations.")

    elif tab == "Logs Viewer":
        st.header("📜 Chat & Job Query Logs")
        logs = list(logs_collection.find().sort("_id", -1).limit(100))  # recent 100 logs

        if logs:
            for i, log in enumerate(logs):
                with st.expander(f"Log #{len(logs) - i}"):
                    st.markdown(f"**User Query:** {log.get('query', 'N/A')}")
                    st.markdown(f"**Model Response:** {log.get('response', 'N/A')}")
                    if "timestamp" in log:
                        st.markdown(f"🕒 {log['timestamp']}")
        else:
            st.info("No logs available yet.")

if __name__ == "__main__":
    main()
