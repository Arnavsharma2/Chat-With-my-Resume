"""
Simple RAG Resume Chatbot
Uses OpenAI embeddings instead of sentence-transformers to avoid threading issues
"""

import os
import PyPDF2
import re
import numpy as np
import textwrap
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

class SimpleRAGChatbot:
    """Simple RAG-powered resume chatbot using OpenAI embeddings"""
    
    def __init__(self):
        self.client = None
        self.resume_data = None
        self.chunks = []
        self.embeddings = None
        self.conversation_history = []
        self.terminal_width = 80  # Default terminal width
        
    def wrap_text(self, text: str, width: int = None) -> str:
        """Wrap text to fit terminal width"""
        if width is None:
            width = self.terminal_width - 4  # Leave some margin
        
        # Split into lines and wrap each line
        lines = text.split('\n')
        wrapped_lines = []
        
        for line in lines:
            if len(line) <= width:
                wrapped_lines.append(line)
            else:
                wrapped_lines.extend(textwrap.wrap(line, width=width))
        
        return '\n'.join(wrapped_lines)
    
    def print_wrapped(self, text: str, prefix: str = ""):
        """Print text with proper wrapping"""
        wrapped_text = self.wrap_text(text)
        for line in wrapped_text.split('\n'):
            print(f"{prefix}{line}")
        
    def initialize_openai(self):
        """Initialize OpenAI client"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == "your_openai_api_key_here":
            print("❌ Please set your OPENAI_API_KEY in the .env file")
            return False
        
        try:
            self.client = OpenAI(api_key=api_key)
            print("✅ OpenAI client initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Error initializing OpenAI: {str(e)}")
            return False
    
    def extract_resume_text(self, pdf_path: str) -> str:
        """Extract text from PDF resume"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"❌ Error reading PDF: {str(e)}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s@\.\-\+\/\(\)\[\]{}:;,\'\"!?]', '', text)
        
        return text.strip()
    
    def identify_sections(self, text: str) -> List[Dict[str, str]]:
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
    
    def chunk_text(self, text: str, chunk_size: int = 500) -> List[Dict[str, Any]]:
        """Split text into manageable chunks for vector storage"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': f"chunk_{i//chunk_size}",
                    'metadata': {
                        'source': 'resume',
                        'chunk_index': i//chunk_size
                    }
                })
        
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            print(f"❌ Error getting embeddings: {str(e)}")
            return []
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)
    
    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Perform semantic search using OpenAI embeddings"""
        if not self.embeddings or not self.chunks:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.get_embeddings([query])[0]
            
            # Calculate similarities
            similarities = []
            for i, chunk_embedding in enumerate(self.embeddings):
                similarity = self.cosine_similarity(query_embedding, chunk_embedding)
                similarities.append((similarity, i))
            
            # Sort by similarity and return top results
            similarities.sort(reverse=True)
            
            results = []
            for similarity, idx in similarities[:top_k]:
                results.append({
                    'text': self.chunks[idx]['text'],
                    'score': similarity,
                    'metadata': self.chunks[idx]['metadata']
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Error during semantic search: {str(e)}")
            return []
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using OpenAI with retrieved context"""
        
        system_prompt = """You are Arnav Sharma speaking directly to a recruiter. Respond naturally and conversationally as if you're in an interview. 

Key guidelines:
1. Speak in first person (I, me, my) as Arnav
2. Keep responses to 2-3 lines maximum - be concise and natural
3. Be confident, enthusiastic, and professional
4. Use specific details and achievements from the resume
5. Sound like a real person talking, not a formal document
6. If asked about something not in the resume, say "I'd be happy to discuss that further" or "That's a great question"
7. Answer questions directly - don't ask questions back to the recruiter
8. Be conversational and engaging, like you're having a real conversation

Remember: You ARE Arnav Sharma talking to a recruiter right now. Answer their questions professionally!"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""The recruiter is asking: "{query}"

Here's my resume information for context:
{context}

Respond naturally as Arnav in 2-3 lines, being conversational and confident."""}
                ],
                max_tokens=200,
                temperature=0.7,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"I apologize, but I encountered an error generating a response: {str(e)}. Please try rephrasing your question."
    
    def process_resume(self, pdf_path: str):
        """Process the resume PDF with RAG"""
        print("📄 Processing resume with RAG...")
        
        # Extract text
        text = self.extract_resume_text(pdf_path)
        if not text:
            print("❌ Failed to extract text from resume.pdf")
            return False
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Identify sections
        sections = self.identify_sections(cleaned_text)
        
        # Create chunks
        chunks = self.chunk_text(cleaned_text)
        
        # Get embeddings for chunks
        print("🧠 Generating embeddings for semantic search...")
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.get_embeddings(chunk_texts)
        
        if not embeddings:
            print("❌ Failed to generate embeddings")
            return False
        
        self.chunks = chunks
        self.embeddings = embeddings
        
        self.resume_data = {
            'raw_text': text,
            'cleaned_text': cleaned_text,
            'sections': sections,
            'chunks': chunks
        }
        
        print(f"✅ Resume processed successfully with RAG!")
        print(f"   Sections found: {len(sections)}")
        print(f"   Content chunks: {len(chunks)}")
        
        # Show sections
        print("\n📋 Resume Sections:")
        for section in sections:
            print(f"   • {section['title']}")
        
        return True
    
    def display_welcome(self):
        """Display welcome message"""
        print("\n" + "="*self.terminal_width)
        print("🤖 RAG-POWERED RESUME CHATBOT - AI Assistant".center(self.terminal_width))
        print("="*self.terminal_width)
        self.print_wrapped("Ask me anything about Arnav's background, skills, and experience! I use advanced AI to find the most relevant information and provide smart responses.")
        print("="*self.terminal_width)
    
    def display_suggested_questions(self):
        """Display suggested questions"""
        questions = [
            "Tell me about your technical skills and programming languages",
            "What are your most significant professional achievements?",
            "Describe your experience with machine learning and AI projects",
            "What leadership roles have you held?",
            "Tell me about your educational background and certifications",
            "What projects have you worked on that you're most proud of?",
            "How do you stay updated with the latest technology trends?",
            "Describe a challenging problem you solved in your career",
            "What are your career goals and aspirations?",
            "Tell me about your experience with data analysis and visualization"
        ]
        
        print("\n💡 Suggested Questions:")
        for i, question in enumerate(questions, 1):
            wrapped_question = self.wrap_text(question, self.terminal_width - 6)
            lines = wrapped_question.split('\n')
            print(f"   {i:2d}. {lines[0]}")
            for line in lines[1:]:
                print(f"       {line}")
    
    def chat_loop(self):
        """Main chat loop with RAG"""
        while True:
            print("\n" + "-"*self.terminal_width)
            user_input = input("🤔 Your question (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using RAG Resume Chatbot! Goodbye!")
                break
            
            if not user_input:
                print("Please enter a question or 'quit' to exit.")
                continue
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Perform semantic search using RAG
            search_results = self.semantic_search(user_input, top_k=3)
            
            if search_results:
                # Format context from search results
                context_parts = []
                for i, result in enumerate(search_results, 1):
                    context_parts.append(f"Source {i}: {result['text']}")
                context = "\n\n".join(context_parts)
            else:
                # Fallback to simple keyword search
                relevant_chunks = self.find_relevant_chunks_simple(user_input, [chunk['text'] for chunk in self.chunks])
                context = "\n\n".join(relevant_chunks)
            
            # Generate response
            print("\n🤖 Arnav's Response:")
            print("-" * (self.terminal_width - 2))
            response = self.generate_response(user_input, context)
            self.print_wrapped(response, "   ")
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
    
    def find_relevant_chunks_simple(self, query: str, chunks: List[str], top_k: int = 3) -> List[str]:
        """Fallback simple keyword search"""
        query_words = set(query.lower().split())
        chunk_scores = []
        
        for i, chunk in enumerate(chunks):
            chunk_words = set(chunk.lower().split())
            common_words = query_words.intersection(chunk_words)
            score = len(common_words) + (len(common_words) / len(query_words)) * 0.5
            chunk_scores.append((score, i, chunk))
        
        chunk_scores.sort(reverse=True)
        return [chunk for _, _, chunk in chunk_scores[:top_k]]
    
    def run(self):
        """Run the RAG chatbot"""
        # Initialize OpenAI
        if not self.initialize_openai():
            return
        
        # Process resume
        if not self.process_resume('resume.pdf'):
            return
        
        # Display welcome and suggestions
        self.display_welcome()
        self.display_suggested_questions()
        
        # Start chat loop
        self.chat_loop()

def main():
    """Main function"""
    chatbot = SimpleRAGChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()
