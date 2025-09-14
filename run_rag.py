#!/usr/bin/env python3
"""
RAG Resume Chatbot Launcher
Launcher for the RAG-powered version
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if RAG requirements are met"""
    print("🔍 Checking RAG requirements...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        return False
    
    # Check if resume.pdf exists
    if not os.path.exists('resume.pdf'):
        print("❌ resume.pdf not found!")
        print("   Please place your resume PDF in the project directory")
        return False
    
    # Check if requirements are installed
    try:
        import openai
        import PyPDF2
        import sentence_transformers
        import faiss
        import numpy
        print("✅ All RAG requirements satisfied!")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("   Installing RAG requirements...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("✅ Requirements installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements")
            return False

def main():
    """Main launcher function"""
    print("🤖 RAG Resume Chatbot - Smart AI Assistant")
    print("=" * 50)
    
    if not check_requirements():
        print("\n❌ Setup incomplete. Please fix the issues above.")
        return
    
    print("\n🚀 Starting RAG Resume Chatbot...")
    print("   This version uses semantic search for smarter responses!")
    print("-" * 50)
    
    try:
        # Run the RAG chatbot
        subprocess.run([sys.executable, "rag_chatbot.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting application: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
