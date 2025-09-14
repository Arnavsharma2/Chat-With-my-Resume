"""
Demo script showcasing the Resume Chatbot capabilities
"""

import os
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine


def run_demo():
    """Run a comprehensive demo of the resume chatbot"""
    
    print("🚀 Resume Chatbot Demo")
    print("=" * 60)
    print("This demo showcases the AI-powered resume chatbot capabilities")
    print("that will impress any recruiter with its intelligent responses.\n")
    
    # Load environment
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Please set your OPENAI_API_KEY in the .env file")
        print("   Copy env_example.txt to .env and add your API key")
        return
    
    try:
        # Initialize system
        print("📄 Processing resume...")
        processor = PDFProcessor('resume.pdf')
        resume_data = processor.process_resume()
        
        print("🧠 Building AI knowledge base...")
        vector_store = VectorStore()
        vector_store.build_index(resume_data['chunks'])
        
        print("🤖 Initializing RAG engine...")
        rag_engine = RAGEngine(api_key, vector_store)
        
        print("✅ System ready! Here are some impressive responses:\n")
        
        # Demo questions that will impress recruiters
        demo_questions = [
            "Give me a comprehensive overview of this candidate's technical expertise and professional achievements",
            "What makes this candidate stand out from other applicants?",
            "Describe their leadership experience and management capabilities",
            "What are their strongest programming languages and how have they applied them?",
            "Tell me about their most impressive projects and their business impact"
        ]
        
        for i, question in enumerate(demo_questions, 1):
            print(f"🎯 Demo Question {i}: {question}")
            print("-" * 80)
            
            response_data = rag_engine.process_query(question)
            print(f"🤖 AI Response:\n{response_data['response']}")
            print(f"\n📊 Context Sources: {response_data['context_sources']}")
            print("=" * 80)
            print()
        
        print("🎉 Demo completed! This chatbot will definitely impress recruiters!")
        print("\nTo start the full application, run:")
        print("   streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print("Make sure your resume.pdf is in the project directory")


if __name__ == "__main__":
    run_demo()
