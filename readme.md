# 📚 AI-Powered Collaborative Knowledge Base

## Overview
This project is a **Streamlit-based AI Knowledge Base** that enables users to upload and store documents, visualize insights, and chat with an AI that leverages **Ollama, ChromaDB, and SentenceTransformer** to provide relevant responses based on stored knowledge.

## Features
- **📂 Document Upload**: Supports **PDF** and **TXT** files for text extraction and storage.
- **📜 View Documents**: Retrieve and display stored documents.
- **🤖 AI Chat**: Ask questions and receive AI-generated responses based on indexed knowledge.
- **📊 Visualize Insights**: Generate a **word cloud** from stored knowledge.
- **🗂️ Document Metadata**: Store and retrieve metadata using **SQLite**.

## Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Embedding Model**: [SentenceTransformers](https://www.sbert.net/)
- **Vector Database**: [ChromaDB](https://chromadb.com/)
- **LLM Model**: [Ollama](https://ollama.com/)
- **Database**: SQLite (for document metadata storage)
- **Visualization**: Matplotlib & WordCloud

## Installation
### Prerequisites
Ensure you have Python installed (Python 3.8+ recommended).

```bash
pip install streamlit sentence-transformers chromadb llama-index ollama PyPDF2 chardet matplotlib wordcloud sqlite3
```

## Usage
Run the Streamlit app:
```bash
streamlit run final.py
```

## How It Works
1. **Upload Documents**: Select and upload a **PDF** or **TXT** file.
2. **Text Extraction**: Content is extracted and stored in **ChromaDB**.
3. **Chat with AI**: Input queries, and the AI retrieves relevant knowledge to generate responses.
4. **Metadata Storage**: Document metadata (file type, size, and timestamp) is saved in **SQLite**.
5. **Data Visualization**: Generate a **word cloud** from stored knowledge.

## Future Enhancements
- 🔍 **Advanced Search**: Improve document retrieval with advanced query techniques.
- 📑 **More File Types**: Support for additional formats like DOCX, CSV.
- 🎨 **Enhanced UI**: Improve user experience with a more interactive design.


