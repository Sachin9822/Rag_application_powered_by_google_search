from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st 
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper

from dotenv import load_dotenv
load_dotenv(override=True)


def get_response(user_query):
    return "I dont know"

def get_search_results_urls(user_query):

    search = GoogleSearchAPIWrapper(k=2)
    def top5_results(query):
        return search.results(query, 2)
    tool= Tool(
        name="google search",
        description="Search google for Finance related news",
        func=top5_results,
    )
    response = tool.run(user_query)
    urls = [i['link'] for i in response]
    return urls

def get_vector_from_url(urls):
    # st.write(urls)
    loader = WebBaseLoader(urls)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(documents)
        
    with st.sidebar:
        st.write(urls)
    # create vector store from chunks 
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents=document_chunks,embedding=embeddings)
    return vector_store
    
def get_context_retriever_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-pro",convert_system_message_to_human=True)
    
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input} \n Given the above conversation, generate a search query for lookup inorder to get information relevant to the converation")
    ])

    retriever_chain = create_history_aware_retriever(llm,retriever,prompt)
    return retriever_chain

def get_conversation_rag_chain(retriver_chain):
    llm = ChatGoogleGenerativeAI(model="gemini-pro",convert_system_message_to_human=True,temperature=0.5)
    prompt = ChatPromptTemplate.from_messages([
        ("system","You are a Expert CA Who will give me advice on how to save the tax and answer only finance related question. Answer the user's question based on the below context\n\n{context}"),
        # MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input} make sure the answer is well formated")
        ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)

    return create_retrieval_chain(retriever_chain,stuff_documents_chain)

# initialise chat history 

st.title("Chat with websites")

# with st.sidebar:
#     website = st.text_input("Enter the website")

# if website is None or website == "":
#     st.info("Please enter the url ")
# else:
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
    AIMessage(content="Hello how can i assist you? ") 
    ]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = []


user_query = st.chat_input("Type your message here ..")
if user_query is not None and user_query != "":

    # response = get_response(user_query)
    urls = get_search_results_urls(user_query)
    st.session_state.vector_store = get_vector_from_url(urls)
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)

    conversation_rag_chain = get_conversation_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        # "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    
    # st.write(response['answer'])
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response['answer']))

    # retrieved_documents = retriever_chain.invoke({
    #     "chat_history": st.session_state.chat_history,
    #     "input": user_query
    # })
    # with st.chat_message("Human"):
    #     st.write(user_query)
    # with st.chat_message("AI"):
    #     st.write(response)

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)