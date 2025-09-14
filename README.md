# Chat-With-my-Resume 🤖

An intelligent resume chatbot built with **Retrieval-Augmented Generation (RAG)** technology using Google's Gemini API. This application allows users to have natural conversations about professional background, skills, and experience through an AI-powered interface that understands context and provides detailed responses.

## ✨ Features

- **RAG Technology**: Uses retrieval-augmented generation for accurate, context-aware responses
- **PDF Processing**: Automatically extracts and processes resume content from PDF files
- **Smart Chunking**: Intelligently splits resume content into overlapping chunks for better retrieval
- **Vector Embeddings**: Uses sentence transformers for semantic similarity search
- **Gemini Integration**: Powered by Google's Gemini Pro model for natural language generation
- **Terminal Interface**: Clean, colorful command-line interface for easy interaction
- **Caching**: Automatically caches processed resume data for faster subsequent runs

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key (get it from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone or download this repository**
   ```bash
   cd Chat-With-my-Resume
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Gemini API key**
   ```bash
   export GEMINI_API_KEY='your_gemini_api_key_here'
   ```
   
   Or create a `.env` file in the project directory:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Place your resume PDF**
   - Make sure your resume is named `resume.pdf` in the project directory
   - Or modify the `resume_path` variable in `main.py`

5. **Run the application**
   ```bash
   python main.py
   ```

## 💬 Usage

Once the application starts, you can ask questions like:

- "What are my technical skills?"
- "Tell me about my work experience"
- "What projects have I worked on?"
- "What is my educational background?"
- "What programming languages do I know?"
- "Describe my professional achievements"

Type `quit`, `exit`, or `bye` to end the conversation.

## 🛠️ Technical Details

### Architecture

- **PDF Processing**: Uses PyPDF2 for text extraction
- **Text Chunking**: Splits resume into overlapping chunks for better retrieval
- **Embeddings**: Uses sentence-transformers (all-MiniLM-L6-v2) for semantic embeddings
- **Similarity Search**: Cosine similarity for finding relevant content
- **LLM Integration**: Google Gemini Pro for generating responses
- **Caching**: Pickle-based caching for processed resume data

### Dependencies

- `google-generativeai`: Gemini API integration
- `PyPDF2`: PDF text extraction
- `sentence-transformers`: Text embeddings
- `scikit-learn`: Similarity calculations
- `python-dotenv`: Environment variable management
- `colorama`: Terminal colors
- `numpy`: Numerical operations

## 📁 Project Structure

```
Chat-With-my-Resume/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── resume.pdf          # Your resume (place your PDF here)
├── resume_cache.pkl    # Cached processed data (auto-generated)
└── README.md           # This file
```

## 🔧 Configuration

You can modify these settings in `main.py`:

- `chunk_size`: Size of text chunks (default: 1000 words)
- `chunk_overlap`: Overlap between chunks (default: 200 words)
- `top_k`: Number of relevant chunks to retrieve (default: 3)

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**
   - Make sure your Gemini API key is correctly set
   - Check that the API key has proper permissions

2. **PDF Not Found**
   - Ensure `resume.pdf` exists in the project directory
   - Check file permissions

3. **Import Errors**
   - Run `pip install -r requirements.txt` to install all dependencies
   - Make sure you're using Python 3.8+

4. **Memory Issues**
   - For very large resumes, consider reducing `chunk_size`
   - The application caches data, so subsequent runs will be faster

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this application!

---

**Happy Chatting! 🎉**