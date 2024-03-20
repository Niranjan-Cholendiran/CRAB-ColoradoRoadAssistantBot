from langchain_google_genai import GoogleGenerativeAIEmbeddings # To convert the text chunks into tokens (or vectors or embeddings)
from langchain_google_genai import ChatGoogleGenerativeAI # The LLM model used to build the application
from langchain_pinecone import PineconeVectorStore # To retive data from existing pinecone indexes
from langchain.prompts import PromptTemplate # Langchain's prompt template to format the LLM's input prompt
from langchain.chains.conversation.memory import ConversationBufferWindowMemory # Langchain's memory component to store chat history
import streamlit as st # Streamlit to build the frontend application
from dotenv import load_dotenv # To load environment variables
import os
load_dotenv()

# Define the embedder, Pinecone vector store, LLM model, prompt template and the memory.

## Initiate the embedder and VectorDB
if "embedder" not in st.session_state:
    st.session_state.embedder= GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key= os.environ['GOOGLE_API_KEY'])
if "db" not in st.session_state:
    st.session_state.db= PineconeVectorStore(index_name=os.environ['PINECONE_INDEX_NAME'], embedding=st.session_state.embedder)

## Create a function to retive data from Pinecone
def retrieve_query(query,k=2):
    """
    Funtion that takes the user query and returns top K matching text chunks from the Pinecone Vector DB using Cosine Similarity score.
    """
    matching_results= st.session_state.db.similarity_search(query,k=k)
    return matching_results

## Initiate the LLM model
if "llm" not in st.session_state:
    st.session_state.llm= ChatGoogleGenerativeAI(google_api_key= os.environ['GOOGLE_API_KEY'], model="gemini-pro", temperature=0.5, convert_system_message_to_human=True)

## Define the prompt template
template = """You are a Colorado driving instructor named CRAB- Colorado Road Assistant Bot, who helps in answering any road rules related questions in English based on the information present in Colorado DMV handbook. Answer the questions concisely and reply in points for any question asking to summarize. 
Try to answer any question based on the below provided data, if not available, then use your pre-trained data.

This is your previous chat history with this human who's asking the question. Use this information to answer any follow-up questions: {chat_history}

Also use these texts as an additional reference to answer the questions: {relevant_docs}

The question is: {question}"""

prompt= PromptTemplate(input_variables=["relevant_docs", "chat_history", "question"], template=template)

## Initiate the memory to remember last few interactions
if "memory" not in st.session_state:
    num_int= 5 # Number of interactions to remember 
    st.session_state.memory= ConversationBufferWindowMemory(k=num_int, memory_key='chat_history')

# Built a function to generate query response
def query_response_with_memory(query):
    """
    Funtion that calls the LLM model to respond the query.
    """
    # Collect the docs relevant to the query from Pinecone DB using similarity search
    matching_results= retrieve_query(query)
    relevant_docs=""
    for relevant_doc in matching_results:
        relevant_docs=relevant_docs + relevant_doc.page_content +" "
    
    # Prepare the query template with system message, relevant docs from Vector DB, chat memory and user's query
    question_prompt= prompt.format(relevant_docs= relevant_docs, chat_history= st.session_state.memory.buffer, question= query)

    # Call the LLM model and generate response
    try:
        response= st.session_state.llm.invoke(question_prompt).content
    except:
        response= "Please rephrase the question and try again."

    # Update the memory
    st.session_state.memory.save_context({"input": query}, {"output": str(response)})
    
    return str(response)

# Build the Streamlit UI
col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    st.image("resources\CRAB Logo WithBg2.jpg")
with col3:
    st.write(' ')
st.title("CRAB- Colorado Road Assistant Bot")
st.caption("Ask any questions related to Colorado road rules, and I will provide accurate answers sourced from the Colorado Driver's Handbook.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages= []

# Display the chat messages history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Respond to user's input with Gemini
query= st.chat_input("Type your question here.")
if query:
    # Display the user's prompt in the chat container
    with st.chat_message("human"):
        st.markdown(query)
    
    # Add user message to streamlit's chat history
    st.session_state.messages.append({'role':'user', 'content':query})

    # Get response from Gemini LLM
    response= query_response_with_memory(query)

    # Display the response in chat container 
    with st.chat_message("assistant"):
        st.markdown(response) 

    # Add user message to streamlit's chat history
    st.session_state.messages.append({'role':'assistant', 'content':response})