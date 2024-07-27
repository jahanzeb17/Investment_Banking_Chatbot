import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai


from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


st.title("Investment Banking Chatbot")

llm = ChatGroq(model_name='Llama3-70b-8192')

prompt_template = """
    Use the following pieces of information to answer the user's questions.
    If you don't know the answer, just say you don't know the answer, don't try to make up an answer.

    Context:{context}
    Question:{input}

    Only return helpful answer noting else.
    Helpful answer: 
    """

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.loader = PyPDFDirectoryLoader("./data")
        st.session_state.data = st.session_state.loader.load()

        st.session_state.spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        st.session_state.chunks = st.session_state.spliter.split_documents(st.session_state.data)

        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

        st.session_state.vectors = Chroma.from_documents(st.session_state.chunks, st.session_state.embeddings)
        st.write("Vector Store is Ready")

if st.button("Process Data"):
    with st.spinner("Processing"):
        vector_embedding()

input_text = st.text_input("Enter your query")
button = st.button("Submit")

prompt = PromptTemplate(template=prompt_template)

if "vectors" in st.session_state:
    retriever = st.session_state.vectors.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    if button and input_text:
        with st.spinner("Processing"):
            response = retriever_chain.invoke({"input": input_text})
            st.write(response['answer'])
else:
    st.write("Please process the data first.")
