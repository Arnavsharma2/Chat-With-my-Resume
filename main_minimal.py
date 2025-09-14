#!/usr/bin/env python3
"""
Chat with Resume - Minimal version without external ML libraries
Uses Gemini API for natural language processing and simple text matching
"""

import os
import sys
import json
import pickle
import re
from typing import List, Dict, Any
from pathlib import Path

import google.generativeai as genai
import PyPDF2
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
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Initialize components
        self.chunks = []
        self.full_text = ""
        
        # Load or process resume
        self._load_resume()
    
    def _load_resume(self):
        """Load and process the resume PDF"""
        print(f"{Fore.YELLOW}Loading resume from {self.resume_path}...")
        
        # Check if processed data exists
        cache_file = "resume_cache_minimal.pkl"
        if os.path.exists(cache_file):
            print(f"{Fore.GREEN}Loading cached resume data...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.chunks = cache_data['chunks']
                self.full_text = cache_data['full_text']
            print(f"{Fore.GREEN}Resume loaded successfully! Found {len(self.chunks)} chunks.")
            return
        
        # Extract text from PDF
        self.full_text = self._extract_pdf_text()
        
        # Chunk the text
        self.chunks = self._chunk_text(self.full_text)
        
        # Cache the processed data
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'full_text': self.full_text
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
    
    def _find_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find most relevant chunks using simple keyword matching"""
        query_words = set(query.lower().split())
        chunk_scores = []
        
        for i, chunk in enumerate(self.chunks):
            chunk_words = set(chunk.lower().split())
            # Calculate simple word overlap score
            overlap = len(query_words.intersection(chunk_words))
            score = overlap / len(query_words) if query_words else 0
            chunk_scores.append((i, score, chunk))
        
        # Sort by score and get top-k
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = chunk_scores[:top_k]
        
        relevant_chunks = []
        for idx, score, chunk in top_chunks:
            relevant_chunks.append({
                'text': chunk,
                'similarity': score
            })
        
        return relevant_chunks
    
    def _create_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Create context from relevant chunks for the LLM"""
        context = "Based on the following resume information:\n\n"
        for i, chunk in enumerate(relevant_chunks, 1):
            context += f"Section {i}:\n{chunk['text']}\n\n"
        return context
    
    def chat(self, user_input: str) -> str:
        """Process user input and generate response using RAG"""
        try:
            # Find relevant chunks
            relevant_chunks = self._find_relevant_chunks(user_input)
            
            # Create context
            context = self._create_context(relevant_chunks)
            
            # Create prompt for Gemini
            prompt = f"""
            {context}
            
            Question: {user_input}
            
            Please provide a helpful and accurate response based on the resume information above. 
            If the question cannot be answered from the resume, please say so politely.
            Keep your response conversational and professional.
            """
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def start_chat(self):
        """Start the interactive chat session"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}{'CHAT WITH RESUME - RAG POWERED CHATBOT':^60}")
        print(f"{Fore.CYAN}{'='*60}")
        print(f"\n{Fore.GREEN}Hello! I'm your AI resume assistant. I can answer questions about:")
        print(f"{Fore.WHITE}• Professional background and experience")
        print(f"{Fore.WHITE}• Skills and qualifications")
        print(f"{Fore.WHITE}• Education and certifications")
        print(f"{Fore.WHITE}• Projects and achievements")
        print(f"\n{Fore.YELLOW}Type 'quit', 'exit', or 'bye' to end the conversation.")
        print(f"{Fore.CYAN}{'='*60}\n")
        
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
                print(f"\n{Fore.MAGENTA}Resume Assistant: {Style.RESET_ALL}", end="")
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
