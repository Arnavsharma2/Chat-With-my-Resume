"""
Resume Chatbot Application
A professional RAG-powered chatbot for resume interactions
"""

import streamlit as st
import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

from pdf_processor import PDFProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine


# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Resume Chatbot - Professional AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
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
    
    .suggested-question {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .suggested-question:hover {
        background-color: #e9ecef;
    }
    
    .stats-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .error-message {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 5px;
        padding: 1rem;
        color: #c62828;
    }
    
    .success-message {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 5px;
        padding: 1rem;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize the RAG system with caching"""
    try:
        # Check for API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            st.error("Please set your OPENAI_API_KEY in the .env file")
            return None
        
        # Initialize components
        pdf_processor = PDFProcessor('resume.pdf')
        vector_store = VectorStore()
        rag_engine = RAGEngine(api_key, vector_store)
        
        # Process resume
        with st.spinner("Processing resume and building knowledge base..."):
            resume_data = pdf_processor.process_resume()
            vector_store.build_index(resume_data['chunks'])
        
        return {
            'pdf_processor': pdf_processor,
            'vector_store': vector_store,
            'rag_engine': rag_engine,
            'resume_data': resume_data
        }
    
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return None


def display_chat_message(message: str, is_user: bool = True):
    """Display a chat message with appropriate styling"""
    message_class = "user-message" if is_user else "assistant-message"
    st.markdown(f'<div class="chat-message {message_class}">{message}</div>', 
                unsafe_allow_html=True)


def display_suggested_questions(questions: List[str], rag_engine: RAGEngine):
    """Display suggested questions as clickable buttons"""
    st.markdown("### 💡 Suggested Questions")
    
    cols = st.columns(2)
    for i, question in enumerate(questions):
        with cols[i % 2]:
            if st.button(question, key=f"suggested_{i}", use_container_width=True):
                # Add question to chat
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()


def display_resume_stats(resume_data: Dict[str, Any], vector_store: VectorStore):
    """Display resume statistics"""
    st.markdown("### 📊 Resume Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sections", len(resume_data['sections']))
    
    with col2:
        st.metric("Content Chunks", resume_data['total_chunks'])
    
    with col3:
        stats = vector_store.get_stats()
        st.metric("Vector Dimensions", stats['dimension'])


def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Resume Chatbot</h1>
        <p>AI-Powered Professional Assistant | RAG Technology</p>
        <p>Ask me anything about this candidate's background, skills, and experience!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🚀 Quick Start")
        
        if st.button("🔄 Refresh System", use_container_width=True):
            st.cache_resource.clear()
            st.session_state.system_initialized = False
            st.rerun()
        
        st.markdown("---")
        
        # Initialize system
        if not st.session_state.system_initialized:
            system = initialize_system()
            if system:
                st.session_state.system = system
                st.session_state.system_initialized = True
                st.success("✅ System initialized successfully!")
            else:
                st.error("❌ Failed to initialize system")
                st.stop()
        else:
            system = st.session_state.system
        
        # Display resume stats
        if system:
            display_resume_stats(system['resume_data'], system['vector_store'])
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    if st.session_state.system_initialized:
        system = st.session_state.system
        rag_engine = system['rag_engine']
        
        # Display chat messages
        for message in st.session_state.messages:
            display_chat_message(
                message["content"], 
                message["role"] == "user"
            )
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about the candidate..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_chat_message(prompt, True)
            
            # Generate response
            with st.spinner("🤔 Thinking..."):
                response_data = rag_engine.process_query(
                    prompt, 
                    st.session_state.messages
                )
            
            # Display response
            display_chat_message(response_data['response'], False)
            
            # Show context sources if available
            if response_data['context_sources'] > 0:
                with st.expander(f"📚 Sources ({response_data['context_sources']} found)"):
                    for i, result in enumerate(response_data['search_results'], 1):
                        st.markdown(f"**Source {i}** (Section: {result.section})")
                        st.markdown(f"*Relevance: {result.score:.3f}*")
                        st.markdown(f"{result.text[:200]}...")
                        st.markdown("---")
        
        # Suggested questions
        if not st.session_state.messages:
            suggested_questions = rag_engine.get_suggested_questions()
            display_suggested_questions(suggested_questions[:6], rag_engine)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            Powered by OpenAI GPT-4 & RAG Technology | Built with Streamlit
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.error("System not initialized. Please check the sidebar for errors.")


if __name__ == "__main__":
    main()
