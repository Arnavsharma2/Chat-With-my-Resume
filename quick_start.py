#!/usr/bin/env python3
"""
Quick Start Script for Resume Chatbot
Simple launcher that avoids complex ML dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if basic requirements are met"""
    print("🔍 Checking basic requirements...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("   Creating .env file...")
        with open('.env', 'w') as f:
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
        print("   Please edit .env and add your OpenAI API key")
        return False
    
    # Check if resume.pdf exists
    if not os.path.exists('resume.pdf'):
        print("❌ resume.pdf not found!")
        print("   Please place your resume PDF in the project directory")
        return False
    
    # Check if basic requirements are installed
    try:
        import streamlit
        import openai
        import PyPDF2
        print("✅ All basic requirements satisfied!")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("   Installing basic requirements...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "simple_requirements.txt"], check=True)
            print("✅ Requirements installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements")
            return False

def main():
    """Main launcher function"""
    print("🤖 Resume Chatbot - Quick Start")
    print("=" * 40)
    
    if not check_requirements():
        print("\n❌ Setup incomplete. Please fix the issues above.")
        return
    
    print("\n🚀 Starting Resume Chatbot...")
    print("   The application will open in your default browser")
    print("   Press Ctrl+C to stop the application")
    print("-" * 40)
    
    try:
        # Run the simple app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "simple_resume_chatbot.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting application: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
