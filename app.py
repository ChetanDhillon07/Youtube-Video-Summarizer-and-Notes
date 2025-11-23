import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

parser=StrOutputParser()
embeddings=HuggingFaceEmbeddings(model='all-MiniLM-L6-v2')
llm=ChatGroq(model='llama-3.1-8b-instant')

def youtube(link:str)->Document:
    doc=YoutubeLoader(link.split('=')[1]).load()
    doc=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50).split_documents(doc)
    return doc

def retriever(doc):
    db=FAISS.from_documents(doc,embeddings)
    retriever=db.as_retriever()
    return retriever

prompt=ChatPromptTemplate.from_messages(
    [
        ('system','You are a good assistant that can help in creating notes from the provided context. create detailed notes with each topic helping the user understand with minimal usage of emojis, dont use very complex language keep it good and easy to understand. \n\n context:{context}'),
        ('human','{input}')
    ]
)


st.header('YouTube Video Summarizer and detailed notes')
link=st.text_input('Input the YouTube Link')
if st.button('Submit'):
    ## Summary
    doc=youtube(link)
    chain=load_summarize_chain(llm,chain_type='stuff')
    summary=chain.run(doc)
    ## Notes
    ret=retriever(doc)
    qa_chain=create_stuff_documents_chain(llm,prompt)
    rag=create_retrieval_chain(ret,qa_chain)
    note=rag.invoke({'input':'Create notes of the provided context in length with detail.'})
    
    st.subheader('Summary')
    st.write(summary)

    st.subheader('Notes')
    st.write(note['answer'])

