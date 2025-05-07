# Multimodal RAG Application

This application implements a complete Multimodal Retrieval Augmented Generation (RAG) pipeline that processes both text and images without using OCR for image content extraction.

## Key Features

- **Text and Image Processing**: Handle both text documents and images seamlessly
- **Multimodal Embeddings**: Create embeddings for both text and images in a shared vector space
- **Semantic Retrieval**: Retrieve relevant documents based on semantic similarity
- **Multimodal LLM Integration**: Generate responses using Groq's LLM while considering both text and images
- **User-Friendly Interface**: Easy-to-use Streamlit interface for document ingestion and querying

## Architecture

The application follows this pipeline:

1. **Document Ingestion**: Users can add text documents or images with descriptions
2. **Embedding Generation**: The CLIP model encodes both text and images into the same embedding space
3. **Vector Storage**: Embeddings are stored in a FAISS index for efficient retrieval
4. **Query Processing**: User queries (text or text+image) are processed through the same embedding pipeline
5. **Document Retrieval**: Most semantically similar documents are retrieved
6. **Response Generation**: Retrieved documents are sent to Groq's LLM for final response generation

## Setup and Installation

### Prerequisites

- Python 3.8+
- Groq API key
- Hugging Face API token (for CLIP model)

### Installation

1. Clone this repository
   ```
   git clone https://github.com/Devarajan8/Multimodal-RAG.git
   cd multimodal-rag-app
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys
   ```
   GROQ_API_KEY=your_groq_api_key
   HUGGING_FACE_TOKEN=your_huggingface_token
   ```

4. Run the application
   ```
   streamlit run chat.py
   ```

## Usage Instructions

### Adding Documents

1. Use the sidebar to add documents:
   - Text tab: Enter text content and click "Add Text Document"
   - Image tab: Upload an image, optionally add a description, and click "Add Image Document"

2. All documents are automatically embedded and stored in the vector database

### Querying

1. Enter your query in the text field
2. Optionally upload an image to include with your query
3. Adjust the number of documents to retrieve using the slider
4. Click "Submit Query" to process your request
5. View the generated response and the retrieved documents

## Technical Details

- **Embedding Model**: CLIP ViT-B/32 for both text and image embeddings (512 dimensions)
- **Vector Database**: FAISS with inner product similarity (cosine similarity for normalized vectors)
- **LLM**: Groq's llama3-8b-8192 model for response generation
- **Document Storage**: Local file system with pickle serialization

## Future Improvements

- Integration with true multimodal LLMs for better image understanding
- Chunk long documents for more precise retrieval
- Add document deletion and editing capabilities
- Implement user management and multi-user support
- Add memory and conversation history for contextual queries

## Requirements

```
streamlit>=1.28.0
pillow>=9.0.0
numpy>=1.22.0
requests>=2.28.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
python-dotenv>=1.0.0
uuid>=1.30
```

## License

This project is licensed under the MIT License.
