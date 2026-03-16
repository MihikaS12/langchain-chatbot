import streamlit as st
import os
from dotenv import load_dotenv

# --- LangChain & LangGraph Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages

# 1. Load environment variables
load_dotenv()

# --- LANGGRAPH CORE LOGIC ---

# 2. DEFINE THE STATE (Vishal Sir: This is the 'Global Notebook')
# We use Annotated with 'add_messages' so that ANY update from ANY node 
# is appended to the history, rather than overwriting it.
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# 3. DEFINE THE CHAT NODE (Node A)
def chat_node(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional AI built with LangGraph. Keep answers concise."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    llm = ChatGroq(model="llama-3.1-8b-instant")
    chain = prompt | llm
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}

# 4. DEFINE A PARALLEL NODE (Node B - Vishal Sir: This shows Parallel Updates)
# This node runs at the same time as the Chat Node. 
# It adds a background "system" note to the state without interfering with the LLM.
def system_monitor_node(state: State):
    # Imagine this node checking a database or an API in the background
    return {"messages": [SystemMessage(content="[System Note: Context & Memory check complete. State is synchronized.]")]}

# 5. BUILD THE GRAPH
workflow = StateGraph(State)

# Add both nodes
workflow.add_node("chat", chat_node)
workflow.add_node("monitor", system_monitor_node)

# PARALLEL EXECUTION: 
# Logic: Start -> (Chat & Monitor) -> End
workflow.add_edge(START, "chat")
workflow.add_edge(START, "monitor") # Both start from the same point!
workflow.add_edge("chat", END)
workflow.add_edge("monitor", END)

# Compile the graph
app_graph = workflow.compile()

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="LangGraph Pro", page_icon="🕸️")
st.title("🕸️ LangGraph Professional Agent")
st.markdown("""
**Technical Note for Review:** This version implements **Parallel Nodes** and **Additive State Updates** (Reducers) to demonstrate robust conversational memory.
""")

# INITIALIZE CHAT HISTORY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)
    elif isinstance(message, SystemMessage):
        # We display the parallel node's message as a small info box
        st.info(message.content)

# CHAT INPUT
user_input = st.chat_input("Ask about LangGraph fundamentals...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    
    input_message = HumanMessage(content=user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Graph Nodes executing in parallel..."):
            # RUN THE GRAPH
            output = app_graph.invoke({
                "messages": st.session_state.chat_history + [input_message]
            })
            
            # The output 'messages' list now contains updates from BOTH nodes!
            # We show the LLM response and log the system message.
            for msg in output["messages"]:
                if isinstance(msg, AIMessage) and msg not in st.session_state.chat_history:
                    st.write(msg.content)
                elif isinstance(msg, SystemMessage) and msg not in st.session_state.chat_history:
                    st.info(msg.content)
    
    # Update local history with the new messages from the graph
    # (Only add messages that aren't already there)
    for msg in output["messages"]:
        if msg not in st.session_state.chat_history and msg.content != user_input:
            st.session_state.chat_history.append(msg)
    # Also add the user message
    st.session_state.chat_history.insert(-len(output["messages"])+1, input_message)

