# Chat With My Resume

RAG (Retrieval Augmented Generation) chatbot that answers questions about your resume using LangGraph, OpenAI embeddings, and Chroma vector database.

## What It Does

- Loads and processes PDF resume into vector embeddings
- Stores resume content in Chroma vector database
- Answers questions about your resume using RAG
- Tracks messages and retrieval-tool calls within each LangGraph question/answer run
- Uses retrieved resume passages and a prompt that asks the model to acknowledge missing information

Controls:

- Install `requirements.txt`, set `OPENAI_API_KEY`, and update `pdf_path` and `persist_directory` in the script for your machine
- Run script: `python rag_chatbot.py`
- Enter questions about your resume when prompted
- The model can call a retrieval tool to find resume sections before answering
- The default `force_recreate = True` rebuilds the configured vector database on startup

## How It Works

1. **PDF Loading**: Loads resume PDF using PyPDFLoader
2. **Text Chunking**: Splits the resume using an 800-character chunk size and 50-character overlap
3. **Embedding**: Creates embeddings using OpenAI text-embedding-3-small
4. **Vector Storage**: Stores embeddings in Chroma vector database
5. **Query Processing**: A similarity-search tool retrieves up to three relevant chunks when called by the model
6. **Answer Generation**: Uses GPT-4o-mini with retrieved context to generate answers
7. **State Management**: Tracks model/tool messages within a run; the CLI starts fresh state for each question

## Dependencies

- `langchain` - LLM framework and tools
- `langchain-openai` - OpenAI integration
- `langchain-chroma` - Vector database
- `langgraph` - State graph for conversation flow
- `openai` - GPT-4o-mini and embeddings

## Technical Details

- LLM Model: GPT-4o-mini with `temperature=0`; this setting does not guarantee deterministic or factually correct answers
- Embedding Model: text-embedding-3-small
- Vector Database: Chroma
- Chunk Size: 800 characters with 50 character overlap
- Framework: LangGraph for model/tool orchestration within each question
- Evaluation status: No automated grounding or hallucination benchmark is included

