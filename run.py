#!/usr/bin/env python3
"""
Resume Chatbot Launcher
Quick start script for the resume chatbot application
"""

import subprocess
import sys
import os
from pathlib import Path


def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("   Please copy env_example.txt to .env and add your OpenAI API key")
        return False
    
    # Check if resume.pdf exists
    if not os.path.exists('resume.pdf'):
        print("❌ resume.pdf not found!")
        print("   Please place your resume PDF in the project directory")
        return False
    
    # Check if requirements are installed
    try:
        import streamlit
        import openai
        import PyPDF2
        import sentence_transformers
        import faiss
        print("✅ All requirements satisfied!")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("   Please run: pip install -r requirements.txt")
        return False


def main():
    """Main launcher function"""
    print("🤖 Resume Chatbot Launcher")
    print("=" * 40)
    
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix the issues above.")
        return
    
    print("\n🚀 Starting Resume Chatbot...")
    print("   The application will open in your default browser")
    print("   Press Ctrl+C to stop the application")
    print("-" * 40)
    
    try:
        # Run streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting application: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
