from langchain_community.tools import DuckDuckGoSearchRun
import traceback

try:
    search = DuckDuckGoSearchRun()
    print("Success initialize DuckDuckGoSearchRun")
except Exception as e:
    print("Failed initialize DuckDuckGoSearchRun")
    traceback.print_exc()
