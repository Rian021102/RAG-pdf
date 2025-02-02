import streamlit as st
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from PyPDF2 import PdfReader  # Ensure you have this library installed
from langchain_core.documents import Document  # Import Document from langchain_core

st.title("Ask Anything About Your Documents!")
st.header("Extract information directly from your reports or PDFs")
# @st.experimental_singleton
def get_embeddings():
    return OllamaEmbeddings(model="bge-m3")

def ask(question):
    if 'chain' in st.session_state:
        response = st.session_state['chain'].invoke({"question": question})
        if isinstance(response, dict):
            return response.get('answer', 'No answer found in response')
        else:
            return response
    return "The processing chain is not initialized. Please upload a file and initialize."

def main():
    # Initialize session state
    if 'reader_init' not in st.session_state:
        st.session_state['reader_init'] = False

    # File upload handling
    uploaded_file = st.file_uploader("Upload a Document (CSV or PDF)", type=["pdf", "csv"])
    if st.button("Process File") and uploaded_file is not None:
        file_type = uploaded_file.type
        if file_type == "application/pdf":
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() if page.extract_text() else ""
            documents = [Document(page_content=text, metadata={"source": "local"})]
    
        embeddings = get_embeddings()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20,length_function=len)
        split_documents = text_splitter.split_documents(documents)
        db = Chroma.from_documents(documents=split_documents, embedding=embeddings, persist_directory='./Chroma_DB')
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # Set up the local model:
        local_model = "llama3.2"
        llm = ChatOllama(model=local_model, num_predict=400,
                         temperature=0.7, top_p=0.8,
                        stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"])
        
        prompt_template = """
        <|start_header_id|>user<|end_header_id|>
        Answer the following question based only on the provided context.
        Your answers must be based on the document and please provide detailed answers. 
        If you don't know the answer, just say you don't know. Don't try to make up an answer".
        Question: {question}
        Context: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        st.session_state['chain'] = chain
        st.session_state['reader_init'] = True

    if st.session_state['reader_init']:
        user_question = st.text_input("Ask a question")
        if st.button("Ask"):
            if user_question:
                answer = ask(user_question)
                st.write("Answer:", answer)
            else:
                st.write("Please enter a question")
    
if __name__ == "__main__":
    main()