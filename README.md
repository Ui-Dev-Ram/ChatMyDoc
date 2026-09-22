# Chat My Doc

Chat with your PDF files using AI.

Upload one or more PDFs, then ask questions from the uploaded documents. The app gives answers only from the PDF context.

## Features

- Upload multiple PDF files
- Ask questions in a simple chat screen
- Uses Gemini embeddings for document search
- Uses Groq for AI answers
- Keeps chat memory during the session

## Setup

Clone this project and create a virtual environment.

```bash
git clone <your-repository-url>
cd rag-pdf-chatbot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file from `.env.example` and add your keys.

```env
GROQ_API_KEY=your_groq_api_key
```

Run the app.

```bash
streamlit run app/rag_agent.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new app on Streamlit Community Cloud.
3. Select `app/rag_agent.py` as the main file.
4. In **Advanced settings → Secrets**, add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

5. Deploy.

Never commit your `.env` file or API keys.

## Tech used

- Streamlit
- LangChain and LangGraph
- Google Gemini embeddings
- Groq
- PyPDF

## Note

Uploaded PDFs are kept only for the running app session. Do not upload private documents to a public deployment unless you are comfortable with the hosting environment.
