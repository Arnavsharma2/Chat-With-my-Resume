"""
Terminal-based Resume Chatbot
A command-line interface for the resume chatbot
"""

import os
import PyPDF2
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

class TerminalResumeChatbot:
    """Terminal-based resume chatbot"""
    
    def __init__(self):
        self.client = None
        self.resume_data = None
        self.conversation_history = []
        
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
    
    def chunk_text(self, text: str, chunk_size: int = 800) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def find_relevant_chunks(self, query: str, chunks: List[str], top_k: int = 3) -> List[str]:
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
7. End with a brief follow-up or show interest in the role
8. Be conversational and engaging, like you're having a real conversation

Remember: You ARE Arnav Sharma talking to a recruiter right now. Be natural and impressive!"""

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
        """Process the resume PDF"""
        print("📄 Processing resume...")
        
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
        
        self.resume_data = {
            'raw_text': text,
            'cleaned_text': cleaned_text,
            'sections': sections,
            'chunks': chunks
        }
        
        print(f"✅ Resume processed successfully!")
        print(f"   Sections found: {len(sections)}")
        print(f"   Content chunks: {len(chunks)}")
        
        # Show sections
        print("\n📋 Resume Sections:")
        for section in sections:
            print(f"   • {section['title']}")
        
        return True
    
    def display_welcome(self):
        """Display welcome message"""
        print("\n" + "="*60)
        print("🤖 RESUME CHATBOT - AI-Powered Professional Assistant")
        print("="*60)
        print("Ask me anything about this candidate's background,")
        print("skills, and experience! I'll provide detailed, impressive")
        print("responses that will wow any recruiter.")
        print("="*60)
    
    def display_suggested_questions(self):
        """Display suggested questions"""
        questions = [
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
        
        print("\n💡 Suggested Questions:")
        for i, question in enumerate(questions, 1):
            print(f"   {i:2d}. {question}")
    
    def chat_loop(self):
        """Main chat loop"""
        while True:
            print("\n" + "-"*60)
            user_input = input("🤔 Your question (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using Resume Chatbot! Goodbye!")
                break
            
            if not user_input:
                print("Please enter a question or 'quit' to exit.")
                continue
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Find relevant chunks
            relevant_chunks = self.find_relevant_chunks(user_input, self.resume_data['chunks'])
            context = "\n\n".join(relevant_chunks)
            
            # Generate response
            print("\n🤖 AI Response:")
            print("-" * 40)
            response = self.generate_response(user_input, context)
            print(response)
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
    
    def run(self):
        """Run the chatbot"""
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
    chatbot = TerminalResumeChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()
