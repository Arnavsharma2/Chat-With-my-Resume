# Chat With My Resume

RAG (Retrieval Augmented Generation) chatbot that answers questions about your resume using LangGraph, OpenAI embeddings, and Chroma vector database.

## What It Does

- Loads and processes PDF resume into vector embeddings
- Stores resume content in Chroma vector database
- Answers questions about your resume using RAG
- Maintains conversation context with LangGraph state management
- Minimizes hallucinations with temperature=0 and structured prompts

Controls:
- Run script: `python rag_chatbot.py`
- Enter questions about your resume when prompted
- Chatbot retrieves relevant resume sections and generates answers

## How It Works

1. **PDF Loading**: Loads resume PDF using PyPDFLoader
2. **Text Chunking**: Splits resume into optimized chunks (800 chars, 50 overlap)
3. **Embedding**: Creates embeddings using OpenAI text-embedding-3-small
4. **Vector Storage**: Stores embeddings in Chroma vector database
5. **Query Processing**: Retrieves relevant chunks based on question similarity
6. **Answer Generation**: Uses GPT-4o-mini with retrieved context to generate answers
7. **State Management**: Maintains conversation history with LangGraph

## Dependencies

- `langchain` - LLM framework and tools
- `langchain-openai` - OpenAI integration
- `langchain-chroma` - Vector database
- `langgraph` - State graph for conversation flow
- `openai` - GPT-4o-mini and embeddings

## Technical Details

- LLM Model: GPT-4o-mini (temperature=0 for deterministic output)
- Embedding Model: text-embedding-3-small
- Vector Database: Chroma
- Chunk Size: 800 characters with 50 character overlap
- Framework: LangGraph for stateful conversation management

