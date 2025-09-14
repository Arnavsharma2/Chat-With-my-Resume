# 🤖 Resume Chatbot - AI-Powered Professional Assistant

A simple and effective Resume Chatbot that allows recruiters to have natural conversations with Arnav Sharma about his professional background, skills, and experience through a terminal-based AI interface.

## ✨ Features

- **💬 Natural Conversations**: Arnav speaks directly to recruiters in first person
- **📄 PDF Processing**: Automatic extraction of resume content
- **🤖 AI-Powered**: Uses OpenAI GPT-4o-mini for fast, cost-effective responses
- **💻 Terminal Interface**: Clean command-line interface
- **🎯 Concise Responses**: 2-3 line responses that sound natural and engaging
- **💡 Suggested Questions**: Pre-built questions to help recruiters get started

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Resume PDF file

### Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**
   - Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Place your resume PDF**
   - Ensure your resume PDF is named `resume.pdf` and placed in the project root directory

4. **Run the chatbot**
   ```bash
   python terminal_chatbot.py
   ```

## 🏗️ Architecture

### Components

1. **PDF Processor** (`pdf_processor.py`)
   - Extracts text from PDF files
   - Identifies resume sections (experience, education, skills, etc.)
   - Chunks content for optimal vector storage

2. **Vector Store** (`vector_store.py`)
   - Generates embeddings using Sentence Transformers
   - Implements FAISS for fast similarity search
   - Handles vector storage and retrieval

3. **RAG Engine** (`rag_engine.py`)
   - Orchestrates context retrieval and response generation
   - Integrates with OpenAI GPT-4 for intelligent responses
   - Manages conversation history and context

4. **Streamlit App** (`app.py`)
   - Professional web interface
   - Real-time chat functionality
   - Interactive features and analytics

### Technology Stack

- **Backend**: Python 3.8+
- **AI/ML**: OpenAI GPT-4, Sentence Transformers, FAISS
- **PDF Processing**: PyPDF2
- **Web Framework**: Streamlit
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: all-MiniLM-L6-v2

## 💡 Usage Examples

### For Recruiters

**Technical Skills Questions:**
- "What programming languages does this candidate know?"
- "Tell me about their machine learning experience"
- "What frameworks and tools have they used?"

**Experience Questions:**
- "Describe their most significant professional achievements"
- "What leadership roles have they held?"
- "Tell me about their project experience"

**Background Questions:**
- "What is their educational background?"
- "What certifications do they have?"
- "Tell me about their career progression"

### Sample Interactions

**Q: "What are this candidate's strongest technical skills?"**

**A:** Based on the resume, this candidate demonstrates exceptional technical proficiency across multiple domains:

**Programming Languages:**
- Python (Advanced) - Extensive experience with data science, machine learning, and web development
- JavaScript/TypeScript - Full-stack development capabilities
- SQL - Database design and optimization expertise
- R - Statistical analysis and data visualization

**Machine Learning & AI:**
- Deep learning frameworks (TensorFlow, PyTorch)
- Natural Language Processing (NLP) and text analysis
- Computer Vision and image processing
- Predictive modeling and statistical analysis

**Data Engineering:**
- ETL pipeline development
- Data warehousing and architecture
- Cloud platforms (AWS, Azure)
- Big data technologies (Spark, Hadoop)

**Web Development:**
- Full-stack development (React, Node.js, Express)
- RESTful API design and implementation
- Database management (PostgreSQL, MongoDB)
- DevOps and deployment automation

This candidate's technical foundation is particularly strong in AI/ML applications, with demonstrated ability to build end-to-end solutions from data collection to model deployment.

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Customization Options

1. **Embedding Model**: Change the model in `vector_store.py`
   ```python
   vector_store = VectorStore(model_name="your-preferred-model")
   ```

2. **Chunk Size**: Adjust in `pdf_processor.py`
   ```python
   chunks = processor.chunk_content(max_chunk_size=500)
   ```

3. **Search Results**: Modify in `rag_engine.py`
   ```python
   search_results = self.retrieve_context(query, top_k=5)
   ```

## 📊 Performance Features

- **Intelligent Caching**: System components are cached for faster subsequent loads
- **Efficient Search**: FAISS provides sub-second similarity search
- **Context Awareness**: Maintains conversation history for better responses
- **Source Attribution**: Shows which parts of the resume informed each response

## 🛠️ Troubleshooting

### Common Issues

1. **"Please set your OPENAI_API_KEY"**
   - Ensure your `.env` file contains a valid OpenAI API key
   - Check that the `.env` file is in the project root directory

2. **"Error extracting text from PDF"**
   - Verify that `resume.pdf` exists in the project root
   - Ensure the PDF is not password-protected
   - Check that the PDF contains extractable text (not just images)

3. **"Error loading embedding model"**
   - Ensure you have an internet connection for the first run
   - The model will be downloaded automatically

4. **Slow performance**
   - The first run takes longer as it processes the resume and builds the vector index
   - Subsequent runs will be much faster due to caching

### Getting Help

If you encounter issues:
1. Check the console output for detailed error messages
2. Ensure all dependencies are installed correctly
3. Verify your OpenAI API key is valid and has sufficient credits
4. Make sure your resume PDF is accessible and readable

## 🎯 Best Practices

### For Optimal Results

1. **Resume Quality**: Ensure your PDF resume has clear, well-structured content
2. **Specific Questions**: Ask detailed questions for more comprehensive answers
3. **Follow-up Questions**: Use the conversation history to ask follow-up questions
4. **Source Verification**: Check the source attribution to understand response accuracy

### Question Tips

- **Be Specific**: "Tell me about their Python experience" vs "What do they know?"
- **Ask for Examples**: "Give me specific examples of their leadership experience"
- **Request Details**: "What technologies did they use in their projects?"
- **Compare Skills**: "How does their experience compare to typical candidates?"

## 🔮 Future Enhancements

- [ ] Multi-resume comparison capabilities
- [ ] Interview question generation
- [ ] Skills gap analysis
- [ ] Resume optimization suggestions
- [ ] Integration with ATS systems
- [ ] Voice interaction support
- [ ] Mobile-responsive design improvements

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

**Built with ❤️ using RAG technology to revolutionize resume interactions for recruiters and candidates alike.**
