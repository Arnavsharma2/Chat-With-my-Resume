"""
Vector Store Module for Resume Chatbot
Handles embedding generation and vector similarity search
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")


@dataclass
class SearchResult:
    """Represents a search result from vector store"""
    text: str
    score: float
    metadata: Dict[str, Any]
    section: str


class VectorStore:
    """Handles vector storage and similarity search for resume content"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model = None
        self.index = None
        self.chunks = []
        self.metadata = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
    def load_embedding_model(self):
        """Load the sentence transformer model"""
        try:
            # Set environment variables to avoid threading issues
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['OMP_NUM_THREADS'] = '1'
            
            self.embedding_model = SentenceTransformer(self.model_name)
            print(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            print(f"Warning: Error loading embedding model: {str(e)}")
            # Fallback to a simpler approach
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                print("Loaded fallback embedding model")
            except Exception as e2:
                raise Exception(f"Error loading embedding model: {str(e2)}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not self.embedding_model:
            self.load_embedding_model()
        
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build FAISS index from resume chunks"""
        if not chunks:
            raise ValueError("No chunks provided to build index")
        
        # Extract texts and metadata
        texts = [chunk['text'] for chunk in chunks]
        self.metadata = [chunk['metadata'] for chunk in chunks]
        self.chunks = chunks
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.generate_embeddings(texts)
        
        # Create FAISS index
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        print(f"Built FAISS index with {len(chunks)} chunks")
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for similar content using vector similarity"""
        if not self.index or not self.embedding_model:
            raise Exception("Index not built. Call build_index() first.")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search the index
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Create search results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    results.append(SearchResult(
                        text=chunk['text'],
                        score=float(score),
                        metadata=chunk['metadata'],
                        section=chunk['section']
                    ))
            
            return results
            
        except Exception as e:
            raise Exception(f"Error during search: {str(e)}")
    
    def save_index(self, filepath: str):
        """Save the vector index and metadata to disk"""
        if not self.index:
            raise Exception("No index to save")
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.index")
            
            # Save metadata and chunks
            data = {
                'chunks': self.chunks,
                'metadata': self.metadata,
                'model_name': self.model_name,
                'dimension': self.dimension
            }
            
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(data, f)
            
            print(f"Index saved to {filepath}")
            
        except Exception as e:
            raise Exception(f"Error saving index: {str(e)}")
    
    def load_index(self, filepath: str):
        """Load the vector index and metadata from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.index")
            
            # Load metadata and chunks
            with open(f"{filepath}.pkl", 'rb') as f:
                data = pickle.load(f)
            
            self.chunks = data['chunks']
            self.metadata = data['metadata']
            self.model_name = data['model_name']
            self.dimension = data['dimension']
            
            # Load embedding model
            self.load_embedding_model()
            
            print(f"Index loaded from {filepath}")
            
        except Exception as e:
            raise Exception(f"Error loading index: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'total_chunks': len(self.chunks),
            'dimension': self.dimension,
            'model_name': self.model_name,
            'index_built': self.index is not None
        }


def main():
    """Test the vector store"""
    # This would be used for testing
    pass


if __name__ == "__main__":
    main()
