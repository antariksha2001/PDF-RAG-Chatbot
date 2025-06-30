# PDF RAG Chatbot with LangChain, OpenAI, and Streamlit

This project lets you upload a PDF and chat with it using Retrieval-Augmented Generation (RAG) powered by LangChain, OpenAI, and Streamlit.

## Features
- Upload any PDF and ask questions about its content
- Uses OpenAI's LLM for answers
- Retrieves relevant PDF chunks using vector search (FAISS)
- Simple Streamlit web interface

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app**
   ```bash
   streamlit run app.py
   ```
4. **Enter your OpenAI API key** in the sidebar
5. **Upload a PDF** and start chatting!

## Notes
- You need an OpenAI API key ([get one here](https://platform.openai.com/account/api-keys)).
- All processing is done locally except for the LLM and embedding calls to OpenAI.

---

Made with ❤️ using LangChain, OpenAI, and Streamlit. 