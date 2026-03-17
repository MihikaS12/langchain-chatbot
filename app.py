import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from graph import app_graph

# ─────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────
st.set_page_config(page_title="Smart AI Assistant", page_icon="✨", layout="centered")
st.title("✨ Smart AI Assistant")

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.title("🛠️ AI Toolbox")
st.sidebar.markdown("Switch tools on or off below.")
use_search = st.sidebar.checkbox("🌐 Web Search", value=False,
                                  help="Search the internet for latest news, weather, facts.")
use_wiki   = st.sidebar.checkbox("📚 Wikipedia", value=False,
                                  help="Look up any topic, person, or place.")
use_calc   = st.sidebar.checkbox("🔢 Calculator", value=False,
                                  help="Perform any math calculation accurately.")

st.sidebar.markdown("---")

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# ─────────────────────────────────────────
# DISPLAY CHAT HISTORY
# ─────────────────────────────────────────
TOOL_TAGS = ["<google_search>", "<wikipedia_search>", "<calculator>",
             "<function", "function=", "</function>"]
TOOL_NAMES = {"google_search", "web_search", "wikipedia_search",
              "calculator", "get_current_time"}

def is_internal_message(msg: AIMessage) -> bool:
    """Returns True if this message should be hidden from the user."""
    # Has structured tool calls
    if msg.tool_calls:
        return True
    content = msg.content.strip()
    # Empty
    if not content:
        return True
    # Contains raw tool XML tags
    if any(tag in content for tag in TOOL_TAGS):
        return True
    # Is just a bare tool name
    if content.lower() in TOOL_NAMES:
        return True
    return False

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        if not is_internal_message(message):
            with st.chat_message("assistant"):
                st.markdown(message.content)

# ─────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                output = app_graph.invoke({
                    "messages": st.session_state.chat_history,
                    "use_search": use_search,
                    "use_wiki":   use_wiki,
                    "use_calc":   use_calc,
                })

                # Get the last meaningful (non-tool-call) AI message
                final_response = ""
                for msg in reversed(output["messages"]):
                    if isinstance(msg, AIMessage) and not is_internal_message(msg):
                        final_response = msg.content
                        break

                if final_response:
                    st.markdown(final_response)
                    st.session_state.chat_history.append(AIMessage(content=final_response))
                else:
                    fallback = "I processed your request. Please try asking again if you need more details."
                    st.markdown(fallback)
                    st.session_state.chat_history.append(AIMessage(content=fallback))

            except Exception as e:
                print(f"[Internal Error]: {e}")
                friendly = "I'm sorry, I ran into a small hiccup. Please try asking again!"
                st.markdown(friendly)
                st.session_state.chat_history.append(AIMessage(content=friendly))
