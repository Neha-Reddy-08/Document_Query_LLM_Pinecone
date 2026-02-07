#!/usr/bin/env python
# coding: utf-8



import openai
import langchain
import pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter #convert into chunks
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAI



from dotenv import load_dotenv
load_dotenv() #this will load all ur env variables


import os


## Read the document

def read_doc(directory):
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    return documents



doc= read_doc('documents/')
len(doc)


# chunk_size
# Definition: Maximum number of characters (or tokens, depending on splitter) per chunk.
# It determines how big each chunk of text will be.
# 
# chunk_overlap
# Definition: Number of characters to repeat between consecutive chunks
# Keeps context across chunk boundaries
# Prevents cutting off important sentences in retrieval
# 
# Chunk size should be chosen based on the embedding model you’re using, because embeddings have practical token limits and performance considerations. Let me explain carefully.


# Divide the document into chunks

def chunk_data(docs, chunk_size=8000, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    return chunks


# Chunking is per Document
# The splitter works on each Document independently
# In your case, the page (Document) is only 900 chars
# Since 900 < 8000
# The entire page becomes a single chunk
# No text from the next page is added
# So you don’t “spill over” to the next page
# Overlap is irrelevant here
# Overlap only matters if the Document produces multiple chunks
# For a single-chunk Document, nothing is repeated



documents=chunk_data(docs=doc)
len(documents)



#Embedings of OPen Ai
embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
embeddings




vectors = embeddings.embed_query("hello world")
len(vectors)



#create vector search DB in pinecone

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index = pc.Index("langchainvector")



from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(index=index, embedding=embeddings)



from uuid import uuid4
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)


#Cosine Similarity retrive results from vector DB
def retrive_query(query, k=2):
    matching_results = vector_store.similarity_search_with_score(query, k=k)
    return matching_results


# Explanation (plain English)
# query → user question
# k=2 → return the top 2 most similar documents
# similarity_search_with_score:
# Converts the query into an embedding
# Compares it with stored vectors using cosine similarity
# Returns:
# [
#   (Document, similarity_score),
#   ...
# ]
# So this function retrieves the most relevant documents from the vector database.
# 
# Why cosine similarity?
# 
# Measures the angle between vectors
# Ignores magnitude, focuses on semantic meaning
# Best for text embeddings
# Default metric for most LLM embeddings


from langchain_classic.chains.question_answering import load_qa_chain #Takes retrieved documents,Injects them into an LLM prompt,Produces a final answer
from langchain_openai import OpenAI



load_dotenv()  # MUST call this first
# Verify the key is loaded
print(os.getenv("OPENAI_API_KEY"))  # Should print your key



llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0.5
)
chain=load_qa_chain(llm, chain_type="stuff")




def retrive_answer(query):
    doc_search = retrive_query(query)
    print(doc_search)
    docs = [doc for doc, score in doc_search]
    response=chain.run(input_documents=docs, question=query)
    return response




our_query = "How much the agriculture target will be increased by how many crore?"
answer = retrive_answer(our_query)




our_query = "how much the agriculture target will be increased by how many crore?"
answer = retrive_answer(our_query)






