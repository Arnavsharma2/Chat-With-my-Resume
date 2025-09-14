#!/usr/bin/env python3
"""
Chat with Resume - An intelligent resume chatbot built with RAG technology
Uses Gemini API for natural language processing and retrieval-augmented generation
"""

import os
import sys
import json
import pickle
from typing import List, Dict, Any
from pathlib import Path

import google.generativeai as genai
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from colorama import init, Fore, Style, Back
from dotenv import load_dotenv

# Initialize colorama for colored terminal output
init(autoreset=True)

class ResumeChatBot:
    """Intelligent resume chatbot using RAG technology with Gemini API"""
    
    def __init__(self, resume_path: str, api_key: str = None):
        """Initialize the chatbot with resume PDF and API key"""
        self.resume_path = resume_path
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY environment variable or pass it directly.")
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Initialize components
        self.chunks = []
        self.embeddings = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Initialize sentence transformer with error handling
        print(f"{Fore.YELLOW}Loading sentence transformer model...")
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"{Fore.GREEN}Sentence transformer loaded successfully!")
        except Exception as e:
            print(f"{Fore.RED}Error loading sentence transformer: {e}")
            print(f"{Fore.YELLOW}Falling back to TF-IDF only...")
            self.sentence_model = None
        
        # Load or process resume
        self._load_resume()
    
    def _load_resume(self):
        """Load and process the resume PDF"""
        print(f"{Fore.YELLOW}Loading resume from {self.resume_path}...")
        
        # Check if processed data exists
        cache_file = "resume_cache.pkl"
        if os.path.exists(cache_file):
            print(f"{Fore.GREEN}Loading cached resume data...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.chunks = cache_data['chunks']
                self.embeddings = cache_data['embeddings']
            print(f"{Fore.GREEN}Resume loaded successfully! Found {len(self.chunks)} chunks.")
            return
        
        # Extract text from PDF
        text = self._extract_pdf_text()
        
        # Chunk the text
        self.chunks = self._chunk_text(text)
        
        # Generate embeddings
        self.embeddings = self._generate_embeddings(self.chunks)
        
        # Cache the processed data
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings
            }, f)
        
        print(f"{Fore.GREEN}Resume processed and cached successfully! Found {len(self.chunks)} chunks.")
    
    def _extract_pdf_text(self) -> str:
        """Extract text from PDF file"""
        try:
            with open(self.resume_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better retrieval"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def _generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks using sentence transformers or TF-IDF"""
        print(f"{Fore.YELLOW}Generating embeddings for {len(chunks)} chunks...")
        
        if self.sentence_model is not None:
            # Use sentence transformers for better semantic similarity
            embeddings = self.sentence_model.encode(chunks)
            print(f"{Fore.GREEN}Generated sentence transformer embeddings!")
        else:
            # Fallback to TF-IDF embeddings
            print(f"{Fore.YELLOW}Using TF-IDF embeddings as fallback...")
            tfidf_matrix = self.vectorizer.fit_transform(chunks)
            embeddings = tfidf_matrix.toarray()
            print(f"{Fore.GREEN}Generated TF-IDF embeddings!")
        
        return embeddings
    
    def _find_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find most relevant chunks for a given query"""
        if self.sentence_model is not None:
            # Use sentence transformers for better semantic similarity
            query_embedding = self.sentence_model.encode([query])
        else:
            # Use TF-IDF for query embedding
            query_embedding = self.vectorizer.transform([query]).toarray()
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_chunks = []
        for idx in top_indices:
            relevant_chunks.append({
                'text': self.chunks[idx],
                'similarity': similarities[idx]
            })
        
        return relevant_chunks
    
    def _create_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Create context from relevant chunks for the LLM"""
        context = "Based on the following resume information:\n\n"
        for i, chunk in enumerate(relevant_chunks, 1):
            context += f"Section {i}:\n{chunk['text']}\n\n"
        return context
    
    def chat(self, user_input: str) -> str:
        """Process user input and generate response using RAG with professional prompting"""
        try:
            # Find relevant chunks
            relevant_chunks = self._find_relevant_chunks(user_input)
            
            # Create context from resume chunks
            context = self._create_context(relevant_chunks)
            
            # Professional system prompt optimized for career contexts
            system_prompt = f"""You are Arnav Sharma's professional AI assistant. You represent Arnav in conversations with potential employers, recruiters, and industry professionals. 

CRITICAL INSTRUCTIONS:
- You are speaking to career professionals who may be evaluating Arnav for opportunities
- Present Arnav as a highly capable, ambitious, and results-driven professional
- Emphasize his technical expertise, leadership potential, and innovative thinking
- Highlight quantifiable achievements and impact wherever possible
- Use confident, professional language that showcases his strengths
- Always respond in first person as Arnav ("I", "my", "me")
- Be specific about technologies, methodologies, and business impact
- Show passion for technology and continuous learning
- MAINTAIN CONVERSATION CONTEXT - carefully read and reference the PREVIOUS CONVERSATION section to understand what the user has already said
- When user asks about "my previous message" or "what did I say", refer to their actual previous messages in the conversation history

ARNAV'S RESUME CONTENT (extracted from PDF):
{context}

RESPONSE GUIDELINES:
- Keep responses concise but impactful (2-4 sentences typically)
- Use specific examples and metrics when possible
- Show enthusiasm for technology and problem-solving
- Demonstrate leadership and initiative
- If asked about something not in the resume, politely redirect to relevant experience
- Always end responses that could lead to follow-up questions with a question to keep the conversation engaging
- CONTINUE THE CONVERSATION TOPIC - don't restart with generic responses
- IMPORTANT: When user asks "what was my previous message" or similar, look at the PREVIOUS CONVERSATION section and tell them exactly what they said in their last message

Remember: You are representing Arnav to potential employers and industry professionals. Make him look exceptional."""

            # Create the final prompt
            prompt = f"{system_prompt}\n\nUser Question: {user_input}\n\nResponse:"
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            # Enhanced error handling with professional fallback responses
            error_message = str(e)
            
            # Handle quota/credit limit errors (including 429 status codes)
            if (any(keyword in error_message.lower() for keyword in ['quota', 'limit', 'exceeded', 'billing', 'credit', '429']) or 
                '429' in error_message):
                return "I apologize, but I've reached my daily API usage limit. Please try again tomorrow, or feel free to reach out to me directly at aqs7726@psu.edu to discuss my experience with machine learning, software development, or my current internship at Wefire. What would you like to know about my background?"
            
            # Handle authentication errors
            if any(keyword in error_message.lower() for keyword in ['api_key', 'authentication', 'unauthorized', '401', '403']):
                return "I apologize, but there's a temporary issue with my AI service. Please try again later, or feel free to reach out to me directly at aqs7726@psu.edu to discuss my experience with machine learning, software development, or my current internship at Wefire. What would you like to know about my background?"
            
            # Generic fallback response
            return "I apologize, but I'm having trouble processing your request right now. I'd be happy to discuss my experience with machine learning, software development, or my current internship at Wefire. What would you like to know about my background?"
    
    def start_chat(self):
        """Start the interactive chat session"""
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}{'CHAT WITH ARNAV SHARMA - PROFESSIONAL AI ASSISTANT':^70}")
        print(f"{Fore.CYAN}{'='*70}")
        print(f"\n{Fore.GREEN}Hello! I'm Arnav Sharma's AI assistant. I can discuss:")
        print(f"{Fore.WHITE}• My current role as Software Engineering Intern at Wefire")
        print(f"{Fore.WHITE}• My technical skills in Python, ML, and AI development")
        print(f"{Fore.WHITE}• My projects including LSTM stock prediction and customer churn analysis")
        print(f"{Fore.WHITE}• My education at Penn State University (Computer Science)")
        print(f"{Fore.WHITE}• My experience with data analysis, web development, and AI applications")
        print(f"\n{Fore.YELLOW}Type 'quit', 'exit', or 'bye' to end the conversation.")
        print(f"{Fore.CYAN}{'='*70}\n")
        
        while True:
            try:
                # Get user input
                user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print(f"\n{Fore.GREEN}Thank you for chatting! Have a great day! 👋")
                    break
                
                if not user_input:
                    continue
                
                # Generate and display response
                print(f"\n{Fore.MAGENTA}Arnav: {Style.RESET_ALL}", end="")
                response = self.chat(user_input)
                print(response)
                print()  # Add spacing
                
            except KeyboardInterrupt:
                print(f"\n\n{Fore.GREEN}Thank you for chatting! Have a great day! 👋")
                break
            except Exception as e:
                print(f"\n{Fore.RED}Error: {str(e)}")
                print(f"{Fore.YELLOW}Please try again or type 'quit' to exit.\n")

def main():
    """Main function to run the chatbot"""
    # Load environment variables
    load_dotenv()
    
    # Configuration
    resume_path = "resume.pdf"
    api_key = os.getenv('GEMINI_API_KEY')
    
    # Check if resume file exists
    if not os.path.exists(resume_path):
        print(f"{Fore.RED}Error: Resume file '{resume_path}' not found!")
        print(f"{Fore.YELLOW}Please make sure the resume PDF is in the same directory as this script.")
        sys.exit(1)
    
    # Check for API key
    if not api_key:
        print(f"{Fore.RED}Error: Gemini API key not found!")
        print(f"{Fore.YELLOW}Please set your GEMINI_API_KEY environment variable.")
        print(f"{Fore.WHITE}You can get your API key from: https://makersuite.google.com/app/apikey")
        print(f"{Fore.WHITE}Then run: export GEMINI_API_KEY='your_api_key_here'")
        sys.exit(1)
    
    try:
        # Initialize and start chatbot
        chatbot = ResumeChatBot(resume_path, api_key)
        chatbot.start_chat()
        
    except Exception as e:
        print(f"{Fore.RED}Failed to initialize chatbot: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
