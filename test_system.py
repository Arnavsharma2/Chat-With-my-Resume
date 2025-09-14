"""
Test script for Resume Chatbot system
"""

import os
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine


def test_pdf_processing():
    """Test PDF processing functionality"""
    print("Testing PDF processing...")
    
    try:
        processor = PDFProcessor('resume.pdf')
        result = processor.process_resume()
        
        print(f"✅ PDF processed successfully!")
        print(f"   - Sections found: {len(result['sections'])}")
        print(f"   - Chunks created: {result['total_chunks']}")
        
        # Show section titles
        print("   - Section titles:")
        for section in result['sections']:
            print(f"     * {section.title}")
        
        return result
        
    except Exception as e:
        print(f"❌ PDF processing failed: {str(e)}")
        return None


def test_vector_store(resume_data):
    """Test vector store functionality"""
    print("\nTesting vector store...")
    
    try:
        vector_store = VectorStore()
        vector_store.build_index(resume_data['chunks'])
        
        print("✅ Vector store built successfully!")
        
        # Test search
        test_query = "What are the technical skills?"
        results = vector_store.search(test_query, top_k=3)
        
        print(f"   - Search test query: '{test_query}'")
        print(f"   - Results found: {len(results)}")
        
        for i, result in enumerate(results, 1):
            print(f"     {i}. Score: {result.score:.3f} - Section: {result.section}")
        
        return vector_store
        
    except Exception as e:
        print(f"❌ Vector store test failed: {str(e)}")
        return None


def test_rag_engine(vector_store):
    """Test RAG engine functionality"""
    print("\nTesting RAG engine...")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        return None
    
    try:
        rag_engine = RAGEngine(api_key, vector_store)
        
        print("✅ RAG engine initialized successfully!")
        
        # Test query
        test_query = "Tell me about this candidate's experience and skills"
        print(f"   - Test query: '{test_query}'")
        
        response_data = rag_engine.process_query(test_query)
        
        print("✅ Query processed successfully!")
        print(f"   - Response length: {len(response_data['response'])} characters")
        print(f"   - Context sources: {response_data['context_sources']}")
        print(f"   - Response preview: {response_data['response'][:200]}...")
        
        return rag_engine
        
    except Exception as e:
        print(f"❌ RAG engine test failed: {str(e)}")
        return None


def main():
    """Run all tests"""
    print("🧪 Resume Chatbot System Test")
    print("=" * 50)
    
    # Test PDF processing
    resume_data = test_pdf_processing()
    if not resume_data:
        print("\n❌ System test failed at PDF processing stage")
        return
    
    # Test vector store
    vector_store = test_vector_store(resume_data)
    if not vector_store:
        print("\n❌ System test failed at vector store stage")
        return
    
    # Test RAG engine
    rag_engine = test_rag_engine(vector_store)
    if not rag_engine:
        print("\n❌ System test failed at RAG engine stage")
        return
    
    print("\n🎉 All tests passed! System is ready to use.")
    print("\nTo start the application, run:")
    print("   streamlit run app.py")


if __name__ == "__main__":
    main()
