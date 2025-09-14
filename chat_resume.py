#!/usr/bin/env python3
"""
Chat with Resume - Streamlined version
Uses Gemini API with simple text processing
"""

import os
import sys
import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv

def extract_pdf_text(pdf_path):
    """Extract text from PDF"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def chat_with_resume():
    """Main chat function"""
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found!")
        print("Please set your API key: export GEMINI_API_KEY='your_key_here'")
        return
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # Load resume
    resume_path = "resume.pdf"
    if not os.path.exists(resume_path):
        print(f"❌ Error: {resume_path} not found!")
        return
    
    print("📄 Loading resume...")
    resume_text = extract_pdf_text(resume_path)
    if not resume_text:
        return
    
    print("✅ Resume loaded successfully!")
    print("\n" + "="*50)
    print("🤖 CHAT WITH RESUME")
    print("="*50)
    print("Ask me anything about the resume!")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Create prompt
            prompt = f"""
            Based on this resume information:
            
            {resume_text}
            
            Question: {user_input}
            
            Please provide a helpful response based on the resume. If the question can't be answered from the resume, say so politely.
            """
            
            print("🤔 Thinking...")
            response = model.generate_content(prompt)
            print(f"Resume Assistant: {response.text}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    chat_with_resume()
