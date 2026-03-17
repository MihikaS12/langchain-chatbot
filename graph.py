import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from duckduckgo_search import DDGS
from datetime import datetime
import wikipedia
import pytz

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────

@tool
def google_search(query: str) -> str:
    """Search the web for current news, facts, weather, prices, or any real-time information."""
    try:
        with DDGS() as ddgs:
            # Try news search first (best for current events)
            news_results = list(ddgs.news(query, max_results=5))
            if news_results:
                lines = []
                for i, r in enumerate(news_results, 1):
                    lines.append(f"{i}. **{r.get('title', '')}**")
                    lines.append(f"   {r.get('body', r.get('excerpt', ''))}")
                    lines.append(f"   Source: {r.get('url', r.get('source', ''))}")
                return "\n".join(lines)

            # Fallback: regular web text search
            text_results = list(ddgs.text(query, max_results=5))
            if text_results:
                lines = []
                for i, r in enumerate(text_results, 1):
                    lines.append(f"{i}. **{r.get('title', '')}**")
                    lines.append(f"   {r.get('body', '')}")
                    lines.append(f"   Source: {r.get('href', '')}")
                return "\n".join(lines)

        return "No results found. Try a more specific search query."
    except Exception as e:
        return f"Search encountered an issue: {str(e)[:100]}. Please try rephrasing your query."

@tool
def calculator(expression: str) -> str:
    """Calculate any math expression like '12 * 45', '100 / 4', '2 ** 10'."""
    try:
        # Safe evaluation — only math, no builtins
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception:
        return "Could not evaluate that expression. Please write it as a math expression like '12 * 45'."

@tool
def wikipedia_search(query: str) -> str:
    """Look up any topic, person, place, or concept on Wikipedia."""
    try:
        result = wikipedia.summary(query, sentences=3)
        return result
    except wikipedia.exceptions.DisambiguationError as e:
        # Try the first suggestion
        try:
            return wikipedia.summary(e.options[0], sentences=3)
        except Exception:
            return f"Found multiple results for '{query}'. Try being more specific."
    except Exception:
        return f"No Wikipedia article found for '{query}'."

# ─────────────────────────────────────────
# LANGGRAPH STATE
# ─────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    use_search: bool
    use_wiki: bool
    use_calc: bool

# ─────────────────────────────────────────
# CHAT NODE
# ─────────────────────────────────────────

def chat_node(state: State):
    # Always inject current real time so bot never says "I don't know the time"
    now_ist = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%I:%M %p, %A %d %B %Y (IST)")
    now_nyc = datetime.now(pytz.timezone('America/New_York')).strftime("%I:%M %p (NYC/ET)")
    now_lon = datetime.now(pytz.timezone('Europe/London')).strftime("%I:%M %p (London/GMT)")

    system_msg = f"""You are a professional AI Assistant.

CONTEXT:
Current Time: {now_ist}

RULES:
1. TOOL USE: If a tool (Web Search, Wikipedia, or Calculator) is enabled and relevant to the user's query, you MUST use it immediately.
2. DISABILITY: If a user asks for information requiring a tool that is currently DISABLED in the state, politely inform them: "I'm sorry, that capability is currently turned off. Please enable it in the sidebar to proceed."
3. ACCURACY: Always base your answers on the tool results. Never ignore or second-guess facts returned by a tool.
4. PROFESSIONALISM: Keep responses helpful, concise, and professional. Never show raw code or internal tool names."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        MessagesPlaceholder(variable_name="messages"),
    ])

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # Bind only enabled tools
    enabled_tools = []
    if state.get("use_search", False):
        enabled_tools.append(google_search)
    if state.get("use_wiki", False):
        enabled_tools.append(wikipedia_search)
    if state.get("use_calc", False):
        enabled_tools.append(calculator)

    if enabled_tools:
        chain = prompt | llm.bind_tools(enabled_tools)
    else:
        chain = prompt | llm

    response = chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}

# ─────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────

workflow = StateGraph(State)

all_tools = [google_search, calculator, wikipedia_search]
workflow.add_node("chat", chat_node)
workflow.add_node("tools", ToolNode(all_tools))

workflow.add_edge(START, "chat")
workflow.add_conditional_edges("chat", tools_condition)
workflow.add_edge("tools", "chat")

app_graph = workflow.compile()
