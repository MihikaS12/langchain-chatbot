import os
from dotenv import load_dotenv
from graph import app_graph
from langchain_core.messages import HumanMessage

load_dotenv()

def test_search():
    print("Testing graph with a search query...")
    inputs = {"messages": [HumanMessage(content="What is the current price of Bitcoin?")]}
    
    # We use stream to see the transitions
    for output in app_graph.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}':")
            # Print last message content if available
            if "messages" in value:
                last_msg = value["messages"][-1]
                print(f"  Content: {last_msg.content[:200]}...")
                if hasattr(last_msg, 'tool_calls'):
                    print(f"  Tool Calls: {last_msg.tool_calls}")

if __name__ == "__main__":
    test_search()
