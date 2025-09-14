#!/usr/bin/env python3
"""
Setup script for Chat with Resume application
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_resume_file():
    """Check if resume.pdf exists"""
    if os.path.exists("resume.pdf"):
        print("✅ Resume file found!")
        return True
    else:
        print("❌ Resume file 'resume.pdf' not found!")
        print("Please place your resume PDF in the project directory and name it 'resume.pdf'")
        return False

def check_api_key():
    """Check if API key is set"""
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print("✅ Gemini API key found!")
        return True
    else:
        print("❌ Gemini API key not found!")
        print("Please set your GEMINI_API_KEY environment variable:")
        print("export GEMINI_API_KEY='your_api_key_here'")
        print("Or create a .env file with: GEMINI_API_KEY=your_api_key_here")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Chat with Resume application...\n")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check resume file
    if not check_resume_file():
        sys.exit(1)
    
    # Check API key
    if not check_api_key():
        sys.exit(1)
    
    print("\n🎉 Setup complete! You can now run the application with:")
    print("python main.py")

if __name__ == "__main__":
    main()
