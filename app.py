import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd


from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

loader = TextLoader("pf.txt")
documents = loader.load()


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
docs = text_splitter.split_documents(documents)

embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")#, all-MiniLM-L6-v2

db = Chroma.from_documents(docs, embedding_function)

#query = "working in teams"
query = st.text_area('Text to analyze', '''
    working in teams, (...)
    ''')
docs = db.similarity_search(query)

st.text(docs[0].page_content)
