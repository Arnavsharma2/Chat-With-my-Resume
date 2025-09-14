"""
RAG Engine for Resume Chatbot
Handles context retrieval and response generation using OpenAI
"""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from vector_store import VectorStore, SearchResult
from pdf_processor import PDFProcessor
import json


class RAGEngine:
    """Retrieval-Augmented Generation engine for resume chatbot"""
    
    def __init__(self, openai_api_key: str, vector_store: VectorStore):
        self.client = OpenAI(api_key=openai_api_key)
        self.vector_store = vector_store
        self.conversation_history = []
        
    def retrieve_context(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieve relevant context from the vector store"""
        try:
            results = self.vector_store.search(query, top_k)
            return results
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return []
    
    def format_context(self, search_results: List[SearchResult]) -> str:
        """Format search results into context string"""
        if not search_results:
            return "No relevant information found in the resume."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(
                f"Source {i} (Section: {result.section}):\n{result.text}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str, conversation_history: List[Dict] = None) -> str:
        """Generate response using OpenAI with retrieved context"""
        
        # System prompt designed to impress recruiters
        system_prompt = """You are an AI assistant representing a professional candidate's resume. Your role is to provide detailed, impressive, and contextually relevant responses about the candidate's background, skills, and experience.

Key guidelines:
1. Always be professional, confident, and articulate
2. Use specific details and quantifiable achievements when available
3. Highlight technical skills, leadership experience, and unique accomplishments
4. Structure responses clearly with bullet points or numbered lists when appropriate
5. Show enthusiasm for the candidate's work and potential
6. If asked about specific technologies or skills, provide detailed explanations of the candidate's experience
7. Always maintain a positive, professional tone that would impress recruiters
8. If information isn't available in the context, politely state that and offer to discuss related topics
9. Use industry-standard terminology and demonstrate understanding of professional contexts
10. End responses with engaging questions or offers to discuss specific aspects further

Remember: You are representing a highly qualified professional. Make them sound impressive!"""

        # Build conversation context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-6:]:  # Keep last 6 messages for context
                messages.append(msg)
        
        # Add current query with context
        user_message = f"""Based on the following resume information, please answer this question: "{query}"

Resume Context:
{context}

Please provide a comprehensive, professional response that would impress recruiters. Include specific details, achievements, and technical expertise where relevant."""

        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=800,
                temperature=0.7,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"I apologize, but I encountered an error generating a response: {str(e)}. Please try rephrasing your question."
    
    def process_query(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Process a user query and return response with metadata"""
        try:
            # Retrieve relevant context
            search_results = self.retrieve_context(query, top_k=5)
            
            # Format context
            context = self.format_context(search_results)
            
            # Generate response
            response = self.generate_response(query, context, conversation_history)
            
            # Update conversation history
            if conversation_history is not None:
                conversation_history.append({"role": "user", "content": query})
                conversation_history.append({"role": "assistant", "content": response})
            
            return {
                'response': response,
                'context_sources': len(search_results),
                'search_results': search_results,
                'context': context
            }
            
        except Exception as e:
            return {
                'response': f"I apologize, but I encountered an error processing your query: {str(e)}",
                'context_sources': 0,
                'search_results': [],
                'context': ""
            }
    
    def get_suggested_questions(self) -> List[str]:
        """Get a list of suggested questions for recruiters"""
        return [
            "Tell me about your technical skills and programming languages",
            "What are your most significant professional achievements?",
            "Describe your experience with machine learning and AI projects",
            "What leadership roles have you held?",
            "Tell me about your educational background and certifications",
            "What projects have you worked on that you're most proud of?",
            "How do you stay updated with the latest technology trends?",
            "Describe a challenging problem you solved in your career",
            "What are your career goals and aspirations?",
            "Tell me about your experience with data analysis and visualization"
        ]
    
    def get_resume_summary(self) -> str:
        """Get a high-level summary of the resume"""
        try:
            # Get general information about the candidate
            general_query = "Provide a comprehensive professional summary including key skills, experience, and achievements"
            result = self.process_query(general_query)
            return result['response']
        except Exception as e:
            return f"Unable to generate resume summary: {str(e)}"


def main():
    """Test the RAG engine"""
    # This would be used for testing
    pass


if __name__ == "__main__":
    main()
