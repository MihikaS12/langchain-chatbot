import streamlit as st
import os
from dotenv import load_dotenv

# --- LangChain & LangGraph Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages

# 1. Load environment variables
load_dotenv()

# --- LANGGRAPH CORE LOGIC ---

# 2. DEFINE THE STATE
# The "State" is a shared notebook nodes can read from and write to.
# Annotated[List[BaseMessage], add_messages] tells LangGraph 
# to "append" new messages into the history instead of overwriting.
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# 3. DEFINE THE CHAT NODE
# A node is just a function that takes the current state and returns an update.
def chat_node(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a highly capable, professional, and friendly AI assistant built with LangGraph. You provide clear, well-structured, and helpful answers."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    llm = ChatGroq(model="llama-3.1-8b-instant")
    chain = prompt | llm
    
    # We pass the entire list of messages from the state to the chain
    response = chain.invoke({"messages": state["messages"]})
    
    # We return the new message to be "added" to the state
    return {"messages": [response]}

# 4. BUILD THE GRAPH
workflow = StateGraph(State)

# Add our node to the graph
workflow.add_node("chat", chat_node)

# Define the flow (Start -> Chat -> End)
workflow.add_edge(START, "chat")
workflow.add_edge("chat", END)

# Compile the graph into a runnable app
app_graph = workflow.compile()

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Smart Assistant (LangGraph)", page_icon="🕸️")
st.title("🕸️ LangGraph AI Assistant")
st.caption("A professional agent with structured state management")

# 5. INITIALIZE CHAT HISTORY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display all previous chat messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# 6. CHAT INPUT BAR
user_input = st.chat_input("How can I help you today?")

if user_input:
    # Immediately show the user's message in the UI
    with st.chat_message("user"):
        st.write(user_input)
    
    # Prepare the input message for the graph
    input_message = HumanMessage(content=user_input)
    
    # RUN THE GRAPH
    with st.chat_message("assistant"):
        with st.spinner("Processing through LangGraph..."):
            # We pass the history + the new message to the graph
            output = app_graph.invoke({
                "messages": st.session_state.chat_history + [input_message]
            })
            
            # The output contains the updated list of messages.
            # We just need the last one (the AI's response).
            response_message = output["messages"][-1]
            st.write(response_message.content)
    
    # SAVE TO SESSION STATE
    st.session_state.chat_history.append(input_message)
    st.session_state.chat_history.append(response_message)

