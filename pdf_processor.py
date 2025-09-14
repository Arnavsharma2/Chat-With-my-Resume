"""
PDF Processing Module for Resume Chatbot
Extracts and processes resume content from PDF files
"""

import PyPDF2
import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ResumeSection:
    """Represents a section of the resume"""
    title: str
    content: str
    section_type: str


class PDFProcessor:
    """Handles PDF extraction and content processing"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.raw_text = ""
        self.sections = []
        
    def extract_text(self) -> str:
        """Extract text from PDF file"""
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                self.raw_text = text
                return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s@\.\-\+\/\(\)\[\]{}:;,\'\"!?]', '', text)
        
        return text.strip()
    
    def identify_sections(self, text: str) -> List[ResumeSection]:
        """Identify and extract different sections of the resume"""
        sections = []
        
        # Common resume section headers
        section_patterns = {
            'contact': r'(?i)(contact|personal information|personal details)',
            'summary': r'(?i)(summary|profile|objective|about)',
            'experience': r'(?i)(experience|work experience|employment|professional experience)',
            'education': r'(?i)(education|academic|qualifications)',
            'skills': r'(?i)(skills|technical skills|competencies)',
            'projects': r'(?i)(projects|portfolio|key projects)',
            'certifications': r'(?i)(certifications|certificates|licenses)',
            'achievements': r'(?i)(achievements|awards|honors|recognition)',
            'publications': r'(?i)(publications|papers|research)',
            'languages': r'(?i)(languages|language skills)'
        }
        
        # Split text into lines for processing
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line matches a section header
            section_found = False
            for section_type, pattern in section_patterns.items():
                if re.search(pattern, line):
                    # Save previous section if exists
                    if current_section and current_content:
                        sections.append(ResumeSection(
                            title=current_section,
                            content=' '.join(current_content),
                            section_type=current_section.lower()
                        ))
                    
                    # Start new section
                    current_section = line
                    current_content = []
                    section_found = True
                    break
            
            if not section_found and current_section:
                current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            sections.append(ResumeSection(
                title=current_section,
                content=' '.join(current_content),
                section_type=current_section.lower()
            ))
        
        # If no sections found, create a general section
        if not sections:
            sections.append(ResumeSection(
                title="Resume Content",
                content=text,
                section_type="general"
            ))
        
        self.sections = sections
        return sections
    
    def chunk_content(self, max_chunk_size: int = 500) -> List[Dict[str, Any]]:
        """Split content into manageable chunks for vector storage"""
        chunks = []
        
        for section in self.sections:
            content = section.content
            words = content.split()
            
            # Split into chunks
            for i in range(0, len(words), max_chunk_size):
                chunk_words = words[i:i + max_chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                if chunk_text.strip():
                    chunks.append({
                        'text': chunk_text,
                        'section': section.title,
                        'section_type': section.section_type,
                        'chunk_id': f"{section.section_type}_{i//max_chunk_size}",
                        'metadata': {
                            'source': 'resume',
                            'section': section.title,
                            'type': section.section_type
                        }
                    })
        
        return chunks
    
    def process_resume(self) -> Dict[str, Any]:
        """Main method to process the entire resume"""
        try:
            # Extract text
            raw_text = self.extract_text()
            
            # Clean text
            cleaned_text = self.clean_text(raw_text)
            
            # Identify sections
            sections = self.identify_sections(cleaned_text)
            
            # Create chunks
            chunks = self.chunk_content()
            
            return {
                'raw_text': raw_text,
                'cleaned_text': cleaned_text,
                'sections': sections,
                'chunks': chunks,
                'total_chunks': len(chunks)
            }
            
        except Exception as e:
            raise Exception(f"Error processing resume: {str(e)}")


def main():
    """Test the PDF processor"""
    processor = PDFProcessor('resume.pdf')
    result = processor.process_resume()
    
    print(f"Processed resume with {len(result['sections'])} sections")
    print(f"Created {result['total_chunks']} chunks")
    
    for section in result['sections']:
        print(f"\nSection: {section.title}")
        print(f"Content preview: {section.content[:100]}...")


if __name__ == "__main__":
    main()
