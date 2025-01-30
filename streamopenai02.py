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

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if st.button("Read PDF") and uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    documents = [Document(page_content=text, metadata={"source": "local"})]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20, length_function=len)
    split_documents = text_splitter.split_documents(documents)
    db = Chroma.from_documents(split_documents, embedding=embeddings_model, persist_directory="./test_index")
    db.persist()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    chain = load_qa_chain(llm, chain_type="stuff")
    st.session_state['retriever'] = retriever
    st.session_state['chain'] = chain
    st.session_state['reader_init'] = True

def ask(question):
    context = st.session_state['retriever'].get_relevant_documents(question)
    prompt_template = """Answer the following question based only on the provided context.
    Your answers must be based on the document and please provide detailed answers. 
    If you don't know the answer, just say you don't know. Don't try to make up an answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer: Let's approach this step by step:"""
    prompt = prompt_template.format(context=context, question=question)
    answer = st.session_state['chain'].run(input_documents=context, question=prompt)
    return answer

if st.session_state['reader_init']:
    user_question = st.text_input("Ask a question")
    if st.button("Ask"):
        if user_question:
            answer = ask(user_question)
            st.write("Answer:", answer)
        else:
            st.write("Please enter a question")
else:
    st.write("Please upload a file and read the PDF to proceed")
