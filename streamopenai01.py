from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.documents import Document
from PyPDF2 import PdfReader
import streamlit as st
import os

st.title("RAG Made Easy")
st.write("This is a simple RAG app, using OPENAI. Please use your own OpenAI API Key.")

api_key = os.environ.get("OPENAI_API_KEY")
embeddings_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4", max_tokens=1000)

# Initialize session state
if 'reader_init' not in st.session_state:
    st.session_state['reader_init'] = False

# Initialize session state
if 'reader_init' not in st.session_state:
    st.session_state['reader_init'] = False

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if st.button("Read PDF") and uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Convert the text string into a Document object with metadata
    documents = [Document(page_content=text, metadata={"source": "local"})]

    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    split_documents = text_splitter.split_documents(documents)

    # Initialize Document Store and Retriever
    db = Chroma.from_documents(split_documents, embedding=embeddings_model, persist_directory="./test_index")
    db.persist()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    chain = load_qa_chain(llm, chain_type="stuff")
    st.session_state['retriever'] = retriever
    st.session_state['chain'] = chain
    st.session_state['reader_init'] = True

if st.session_state['reader_init']:
    question = st.text_input("Ask a question")
    if st.button("Ask"):
        if question:
            docs = st.session_state['retriever'].get_relevant_documents(question)
            answer = st.session_state['chain'].run(input_documents=docs, question=question)
            st.write(answer)
        else:
            st.write("Please enter a question")
else:
    st.write("Please upload a file and read the PDF to proceed")


