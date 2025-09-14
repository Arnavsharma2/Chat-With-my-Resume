#!/usr/bin/env python3
"""
Terminal Resume Chatbot Launcher
Simple launcher for the command-line version
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if basic requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        return False
    
    # Check if resume.pdf exists
    if not os.path.exists('resume.pdf'):
        print("❌ resume.pdf not found!")
        print("   Please place your resume PDF in the project directory")
        return False
    
    # Check if basic requirements are installed
    try:
        import openai
        import PyPDF2
        print("✅ All requirements satisfied!")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("   Installing requirements...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "openai", "pypdf2", "python-dotenv"], check=True)
            print("✅ Requirements installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements")
            return False

def main():
    """Main launcher function"""
    print("🤖 Resume Chatbot - Terminal Version")
    print("=" * 40)
    
    if not check_requirements():
        print("\n❌ Setup incomplete. Please fix the issues above.")
        return
    
    print("\n🚀 Starting Resume Chatbot...")
    print("-" * 40)
    
    try:
        # Run the terminal chatbot
        subprocess.run([sys.executable, "terminal_chatbot.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting application: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
