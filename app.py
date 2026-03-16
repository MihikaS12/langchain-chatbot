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
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# 3. DEFINE THE CHAT NODE
def chat_node(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a highly capable, professional, and friendly AI assistant. You provide clear, well-structured, and helpful answers."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    llm = ChatGroq(model="llama-3.1-8b-instant")
    chain = prompt | llm
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}

# 4. BUILD THE GRAPH
workflow = StateGraph(State)
workflow.add_node("chat", chat_node)
workflow.add_edge(START, "chat")
workflow.add_edge("chat", END)
app_graph = workflow.compile()

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Smart AI Assistant", page_icon="✨")
st.title("✨ Smart AI Assistant")
st.caption("A professional conversational agent powered by LangGraph & Groq")

# INITIALIZE SESSION STATE
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display messages from session state
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# CHAT INPUT
user_input = st.chat_input("How can I help you today?")

if user_input:
    # 1. Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 2. Add to local history
    input_message = HumanMessage(content=user_input)
    st.session_state.chat_history.append(input_message)
    
    # 3. Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Run the graph with the current history
            output = app_graph.invoke({"messages": st.session_state.chat_history})
            
            # The last message in the output is the AI's response
            ai_response = output["messages"][-1]
            st.markdown(ai_response.content)
            
            # 4. Save AI response to history
            st.session_state.chat_history.append(ai_response)

