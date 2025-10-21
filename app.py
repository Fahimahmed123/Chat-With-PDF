import streamlit as st
from dotenv import load_dotenv

import os
from htmlTemplets import css, bot_template, user_template

#reading pdf text
from PyPDF2 import PdfReader

# making text chunks
from langchain.text_splitter import CharacterTextSplitter


# for getting embeddings 
# from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain.embeddings import HuggingFaceEmbeddings


#vector store

from langchain.vectorstores import FAISS

# for conversational chain

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from langchain.llms import HuggingFaceHub
from transformers import pipeline
from langchain.llms import HuggingFacePipeline



load_dotenv()  # loads .env locally

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")


# reading raw text from the pdf

def get_pdf_texts(pdf_files):
    raw_text = ""

    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    
    return raw_text

# making chunks from the raw text we found by using the raw texts

def get_text_chunks(raw_text):
    text_spliter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len

    )
    text_chunks = text_spliter.split_text(raw_text)

    return text_chunks

def get_vector_to_store(text_chunks):
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversational_chain(vector_store):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=512,
    temperature=0.5
)
    llm = HuggingFacePipeline(pipeline=pipe)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)






def main():
    # load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.write(css, unsafe_allow_html=True)

    st.header("Chat with multiple PDFs :books:")
    user_questions = st.text_input("Enter your questions about the documents: ")
    if user_questions:
        handle_user_input(user_questions)

    with st.sidebar:
        st.subheader("Your Documents : ")
        pdf_docs = st.file_uploader("Upload your files here" , accept_multiple_files=True)
        # st.write(pdf_docs)


        if st.button("Process"):
            with st.spinner("Processing"):

                # get pdf texts
                raw_text = get_pdf_texts(pdf_docs)
                

                #get text chunks for saving the vector DB

                text_chunks = get_text_chunks(raw_text)


                # get the embeddings in the vector database

                vector_store = get_vector_to_store(text_chunks)

                # making conversational chain

                st.session_state.conversation = get_conversational_chain(vector_store)





    
if __name__ == '__main__':
    main()