"""
Simple test for Resume Chatbot system
Tests individual components without full initialization
"""

import os
import sys
from dotenv import load_dotenv

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import openai
        print("✅ OpenAI imported successfully")
    except ImportError as e:
        print(f"❌ OpenAI import failed: {e}")
        return False
    
    try:
        import PyPDF2
        print("✅ PyPDF2 imported successfully")
    except ImportError as e:
        print(f"❌ PyPDF2 import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✅ Sentence Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Sentence Transformers import failed: {e}")
        return False
    
    try:
        import faiss
        print("✅ FAISS imported successfully")
    except ImportError as e:
        print(f"❌ FAISS import failed: {e}")
        return False
    
    return True

def test_pdf_processing():
    """Test PDF processing without full system"""
    print("\n📄 Testing PDF processing...")
    
    try:
        from pdf_processor import PDFProcessor
        
        if not os.path.exists('resume.pdf'):
            print("❌ resume.pdf not found")
            return False
        
        processor = PDFProcessor('resume.pdf')
        result = processor.process_resume()
        
        print(f"✅ PDF processed successfully!")
        print(f"   - Sections found: {len(result['sections'])}")
        print(f"   - Chunks created: {result['total_chunks']}")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF processing failed: {str(e)}")
        return False

def test_environment():
    """Test environment setup"""
    print("\n🔧 Testing environment...")
    
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env file")
        print("   Please edit .env and add your OpenAI API key")
        return False
    
    if api_key == "your_openai_api_key_here":
        print("❌ Please replace the placeholder API key in .env file")
        return False
    
    print("✅ Environment configured correctly")
    return True

def main():
    """Run simple tests"""
    print("🧪 Resume Chatbot - Simple System Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please check your installation.")
        return
    
    # Test environment
    if not test_environment():
        print("\n❌ Environment test failed. Please configure your .env file.")
        return
    
    # Test PDF processing
    if not test_pdf_processing():
        print("\n❌ PDF processing test failed.")
        return
    
    print("\n🎉 All basic tests passed!")
    print("\nTo start the full application:")
    print("   streamlit run app.py")
    print("\nOr use the launcher:")
    print("   python run.py")

if __name__ == "__main__":
    main()
