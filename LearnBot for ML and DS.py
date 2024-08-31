import streamlit as st
import openai
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import json

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "Enter your OpenAI API key"
os.environ['LANGCHAIN_API_KEY'] = "Enter your Langchain API Key"
os.environ['LANGCHAIN_TRACING_V2'] = "true"

# Load the JSON file
with open('database.json', 'r') as file:
    data = json.load(file)

# Combine all patterns and responses for the intents
all_data = []
for i in range(len(data.get("intents"))):
    dicts = data.get("intents")[i]
    all_data.extend(dicts["patterns"])
    all_data.extend(dicts["responses"])

# Split the texts into chunks for better vectorization
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
docs = text_splitter.create_documents(all_data)

# Initialize the embeddings and vector store
embedding_model = OpenAIEmbeddings()
vector_store = Chroma.from_documents(docs, embedding_model)

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get the relevant information to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(vector_store, user_input, chat_history):
    retriever_chain = get_context_retriever_chain(vector_store)
    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversational_rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    return response['answer']

# Initialize chat history
chat_history = [AIMessage(content="Hello, I am Lipa. How can I help you?")]

# Streamlit Interface
st.title("Lipa - Your ML & DS Chatbot")

st.write("Ask me anything about Machine Learning or Data Science:")

user_query = st.text_input("Enter Your Query:", "")

if user_query:
    response = get_response(vector_store, user_query, chat_history)
    st.write(f"Lipa: {response}")
    chat_history.append(HumanMessage(content=user_query))
    chat_history.append(AIMessage(content=response))
