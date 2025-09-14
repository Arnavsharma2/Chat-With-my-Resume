"""
Simple Resume Chatbot - Working Version
A streamlined resume chatbot that works without complex ML dependencies
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

def clean_text(text: str) -> str:
    """Clean and normalize extracted text"""
    # Remove extra whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'[^\w\s@\.\-\+\/\(\)\[\]{}:;,\'\"!?]', '', text)
    
    return text.strip()

def identify_sections(text: str) -> List[Dict[str, str]]:
    """Identify resume sections using keyword matching"""
    sections = []
    
    # Common resume section patterns
    section_patterns = {
        'Contact': r'(?i)(contact|personal information|personal details)',
        'Summary': r'(?i)(summary|profile|objective|about)',
        'Experience': r'(?i)(experience|work experience|employment|professional experience)',
        'Education': r'(?i)(education|academic|qualifications)',
        'Skills': r'(?i)(skills|technical skills|competencies)',
        'Projects': r'(?i)(projects|portfolio|key projects)',
        'Certifications': r'(?i)(certifications|certificates|licenses)',
        'Achievements': r'(?i)(achievements|awards|honors|recognition)',
        'Publications': r'(?i)(publications|papers|research)',
        'Languages': r'(?i)(languages|language skills)'
    }
    
    # Split text into lines for processing
    lines = text.split('\n')
    current_section = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line matches a section header
        section_found = False
        for section_name, pattern in section_patterns.items():
            if re.search(pattern, line):
                # Save previous section if exists
                if current_section and current_content:
                    sections.append({
                        'title': current_section,
                        'content': ' '.join(current_content),
                        'type': current_section.lower()
                    })
                
                # Start new section
                current_section = section_name
                current_content = []
                section_found = True
                break
        
        if not section_found and current_section:
            current_content.append(line)
    
    # Add the last section
    if current_section and current_content:
        sections.append({
            'title': current_section,
            'content': ' '.join(current_content),
            'type': current_section.lower()
        })
    
    # If no sections found, create a general section
    if not sections:
        sections.append({
            'title': 'Resume Content',
            'content': text,
            'type': 'general'
        })
    
    return sections

def chunk_text(text: str, chunk_size: int = 800) -> List[str]:
    """Split text into manageable chunks"""
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
        # Calculate relevance score
        common_words = query_words.intersection(chunk_words)
        score = len(common_words) + (len(common_words) / len(query_words)) * 0.5
        chunk_scores.append((score, i, chunk))
    
    # Sort by score and return top chunks
    chunk_scores.sort(reverse=True)
    return [chunk for _, _, chunk in chunk_scores[:top_k]]

def generate_response(query: str, context: str, client: OpenAI) -> str:
    """Generate response using OpenAI with retrieved context"""
    
    system_prompt = """You are an AI assistant representing a professional candidate's resume. Your role is to provide detailed, impressive, and contextually relevant responses about the candidate's background, skills, and experience.

Key guidelines:
1. Always be professional, confident, and articulate
2. Use specific details and quantifiable achievements when available
3. Highlight technical skills, leadership experience, and unique accomplishments
4. Structure responses clearly with bullet points or numbered lists when appropriate
5. Show enthusiasm for the candidate's work and potential
6. If asked about specific technologies or skills, provide detailed explanations of the candidate's experience
7. Always maintain a positive, professional tone that would impress recruiters
8. If information isn't available in the context, politely state that and offer to discuss related topics
9. Use industry-standard terminology and demonstrate understanding of professional contexts
10. End responses with engaging questions or offers to discuss specific aspects further

Remember: You are representing a highly qualified professional. Make them sound impressive!"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""Based on the following resume information, please answer this question: "{query}"

Resume Context:
{context}

Please provide a comprehensive, professional response that would impress recruiters. Include specific details, achievements, and technical expertise where relevant."""}
            ],
            max_tokens=800,
            temperature=0.7,
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"I apologize, but I encountered an error generating a response: {str(e)}. Please try rephrasing your question."

def display_chat_message(message: str, is_user: bool = True):
    """Display a chat message with appropriate styling"""
    message_class = "user-message" if is_user else "assistant-message"
    st.markdown(f'<div class="chat-message {message_class}">{message}</div>', 
                unsafe_allow_html=True)

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
    if 'resume_data' not in st.session_state:
        st.session_state.resume_data = None
    if 'client' not in st.session_state:
        st.session_state.client = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🚀 Setup")
        
        # Check API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == "your_openai_api_key_here":
            st.error("Please set your OPENAI_API_KEY in the .env file")
            st.markdown("**Steps to fix:**")
            st.markdown("1. Edit the `.env` file")
            st.markdown("2. Replace `your_openai_api_key_here` with your actual API key")
            st.markdown("3. Save the file and refresh this page")
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
        if not st.session_state.resume_data:
            if st.button("📄 Process Resume", use_container_width=True):
                with st.spinner("Processing resume..."):
                    # Extract text
                    text = extract_resume_text('resume.pdf')
                    if not text:
                        st.error("Failed to extract text from resume.pdf")
                        st.stop()
                    
                    # Clean text
                    cleaned_text = clean_text(text)
                    
                    # Identify sections
                    sections = identify_sections(cleaned_text)
                    
                    # Create chunks
                    chunks = chunk_text(cleaned_text)
                    
                    st.session_state.resume_data = {
                        'raw_text': text,
                        'cleaned_text': cleaned_text,
                        'sections': sections,
                        'chunks': chunks
                    }
                    
                    st.success(f"✅ Processed resume successfully!")
                    st.markdown(f"**Sections found:** {len(sections)}")
                    st.markdown(f"**Content chunks:** {len(chunks)}")
        
        # Display resume stats
        if st.session_state.resume_data:
            st.markdown("### 📊 Resume Statistics")
            data = st.session_state.resume_data
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sections", len(data['sections']))
            with col2:
                st.metric("Chunks", len(data['chunks']))
            
            # Show sections
            st.markdown("**Resume Sections:**")
            for section in data['sections']:
                st.markdown(f"• {section['title']}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    if not st.session_state.resume_data:
        st.info("Please process the resume first using the button in the sidebar.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        display_chat_message(message["content"], message["role"] == "user")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the candidate..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message(prompt, True)
        
        # Find relevant chunks
        relevant_chunks = find_relevant_chunks(prompt, st.session_state.resume_data['chunks'])
        context = "\n\n".join(relevant_chunks)
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            response = generate_response(prompt, context, st.session_state.client)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        display_chat_message(response, False)
    
    # Suggested questions
    if not st.session_state.messages:
        st.markdown("### 💡 Suggested Questions")
        suggested_questions = [
            "Tell me about this candidate's technical skills and programming languages",
            "What are their most significant professional achievements?",
            "Describe their experience with machine learning and AI projects",
            "What leadership roles have they held?",
            "Tell me about their educational background and certifications",
            "What projects have they worked on that they're most proud of?",
            "How do they stay updated with the latest technology trends?",
            "Describe a challenging problem they solved in their career",
            "What are their career goals and aspirations?",
            "Tell me about their experience with data analysis and visualization"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(suggested_questions):
            with cols[i % 2]:
                if st.button(question, key=f"suggested_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Powered by OpenAI GPT-4 | Built with Streamlit | Resume Chatbot v1.0
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
