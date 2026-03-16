# Smart AI Assistant 🤖

A highly capable, professional, and friendly conversational AI assistant built with [LangChain](https://python.langchain.com/) and powered by [Groq](https://groq.com/)'s ultra-fast LPU inference engine using the Llama 3 model. The user interface is built with [Streamlit](https://streamlit.io/).

## Features ✨

- **Llama 3 Powered**: Utilizing the `llama-3.1-8b-instant` model for fast and intelligent responses.
- **Conversational Memory**: The chatbot remembers previous parts of the conversation to provide contextual answers.
- **Interactive UI**: A clean, easy-to-use chat interface built entirely with Streamlit.

## Setup Instructions 🚀

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd my-chatbot
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```

Activate the environment:
- On **Windows**:
  ```bash
  .\venv\Scripts\activate
  ```
- On **Mac/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setting up API Keys
Create a `.env` file in the root directory (same level as `app.py`) and add your Groq API key:
```ini
GROQ_API_KEY=gsk_your_actual_groq_key_here
```
*(Get a free Groq API key at [console.groq.com](https://console.groq.com/))*

### 5. Run the Application
```bash
streamlit run app.py
```
The application will automatically open in your web browser. 
