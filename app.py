import streamlit as st
import os
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# 1. Load environment variables
load_dotenv()

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Smart Assistant", page_icon="✨")
st.title("✨ Smart AI Assistant")
st.caption("A helpful conversational agent built with LangChain & Groq")

# 2. INITIALIZE CHAT HISTORY (MEMORY)
# Streamlit re-runs the whole script every time you type something.
# We must use st.session_state to store the history so it isn't erased.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display all previous chat messages in the UI
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# 3. CHAT INPUT BAR
# st.chat_input creates a nice message bar at the bottom of the screen
user_input = st.chat_input("How can I help you today?")

if user_input:
    # Immediately show the user's message in the UI
    with st.chat_message("user"):
        st.write(user_input)
    
    # --- LANGCHAIN LOGIC ---
    
    # 4. PROMPT TEMPLATE WITH MEMORY
    # MessagesPlaceholder tells LangChain where to inject our past conversation history!
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a highly capable, professional, and friendly AI assistant. You provide clear, well-structured, and helpful answers on a variety of topics. You are NOT just a coding mentor, but a general-purpose intelligent agent."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}")
        ]
    )
    
    llm = ChatGroq(model="llama-3.1-8b-instant")
    output_parser = StrOutputParser()
    
    chain = prompt | llm | output_parser
    
    # 5. RUN THE CHAIN & DISPLAY AI RESPONSE
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # We must pass BOTH the new question AND the history to the chain
            response = chain.invoke({
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })
            st.write(response)
    
    # 6. SAVE TO MEMORY
    # Save the current interaction so it is remembered for the NEXT question
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))
