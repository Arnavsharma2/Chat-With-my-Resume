"""
Minimal Resume Chatbot - Simplified version to avoid threading issues
"""

import streamlit as st
import os
import PyPDF2
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Resume Chatbot - Professional AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

def extract_resume_text(pdf_path: str) -> str:
    """Extract text from PDF resume"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def find_relevant_chunks(query: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Simple keyword-based search for relevant chunks"""
    query_words = set(query.lower().split())
    chunk_scores = []
    
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk.lower().split())
        score = len(query_words.intersection(chunk_words))
        chunk_scores.append((score, i, chunk))
    
    # Sort by score and return top chunks
    chunk_scores.sort(reverse=True)
    return [chunk for _, _, chunk in chunk_scores[:top_k]]

def generate_response(query: str, context: str, client: OpenAI) -> str:
    """Generate response using OpenAI"""
    system_prompt = """You are an AI assistant representing a professional candidate's resume. 
    Provide detailed, impressive, and contextually relevant responses about the candidate's background, 
    skills, and experience. Be professional, confident, and highlight achievements and technical expertise."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Based on this resume information: {context}\n\nAnswer this question: {query}"}
            ],
            max_tokens=600,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

def main():
    """Main application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Resume Chatbot</h1>
        <p>AI-Powered Professional Assistant | Simplified Version</p>
        <p>Ask me anything about this candidate's background, skills, and experience!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'resume_chunks' not in st.session_state:
        st.session_state.resume_chunks = []
    if 'client' not in st.session_state:
        st.session_state.client = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🚀 Setup")
        
        # Check API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == "your_openai_api_key_here":
            st.error("Please set your OPENAI_API_KEY in the .env file")
            st.stop()
        
        # Initialize OpenAI client
        if not st.session_state.client:
            try:
                st.session_state.client = OpenAI(api_key=api_key)
                st.success("✅ OpenAI client initialized")
            except Exception as e:
                st.error(f"Error initializing OpenAI: {str(e)}")
                st.stop()
        
        # Process resume
        if not st.session_state.resume_chunks:
            if st.button("📄 Process Resume", use_container_width=True):
                with st.spinner("Processing resume..."):
                    text = extract_resume_text('resume.pdf')
                    if text:
                        chunks = chunk_text(text)
                        st.session_state.resume_chunks = chunks
                        st.success(f"✅ Processed {len(chunks)} chunks")
                    else:
                        st.error("Failed to process resume")
        
        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main interface
    if not st.session_state.resume_chunks:
        st.info("Please process the resume first using the button in the sidebar.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        message_class = "user-message" if message["role"] == "user" else "assistant-message"
        st.markdown(f'<div class="chat-message {message_class}">{message["content"]}</div>', 
                    unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the candidate..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Find relevant chunks
        relevant_chunks = find_relevant_chunks(prompt, st.session_state.resume_chunks)
        context = "\n\n".join(relevant_chunks)
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            response = generate_response(prompt, context, st.session_state.client)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Suggested questions
    if not st.session_state.messages:
        st.markdown("### 💡 Suggested Questions")
        suggested_questions = [
            "Tell me about this candidate's technical skills",
            "What are their most significant achievements?",
            "Describe their educational background",
            "What projects have they worked on?",
            "Tell me about their work experience"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(suggested_questions):
            with cols[i % 2]:
                if st.button(question, key=f"suggested_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.rerun()

if __name__ == "__main__":
    main()
