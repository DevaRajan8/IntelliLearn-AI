import os
import io
import base64
import requests
import numpy as np
import streamlit as st
from typing import List, Dict, Optional, Tuple, Union
from PIL import Image
import pickle
import uuid
import time
from sentence_transformers import SentenceTransformer
import faiss

# Load environment variables

# Configuration
GROQ_API_KEY = "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"
HUGGING_FACE_TOKEN = "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"
DATA_DIR = "data"
VECTOR_DB_PATH = os.path.join(DATA_DIR, "vector_db.pkl")
DOCUMENTS_PATH = os.path.join(DATA_DIR, "documents.pkl")

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

class MultimodalEncoder:
    """Class for encoding text and images into a shared embedding space."""
    
    def __init__(self):
        # Initialize text encoder
        self.text_encoder = SentenceTransformer('clip-ViT-B-32')
        
        # Import CLIP directly for image encoding
        from PIL import Image
        import torch
        from transformers import CLIPProcessor, CLIPModel
        
        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = self.clip_model.to(self.device)
        
        self.embedding_dim = 512  # CLIP-ViT-B-32 dimension
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector."""
        return self.text_encoder.encode(text)
    
    def encode_image(self, image) -> np.ndarray:
        """Encode image to embedding vector."""
        import torch
        
        # Convert PIL image to format expected by the CLIP processor
        if isinstance(image, Image.Image):
            # Process image for CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            # Convert to numpy array and normalize
            image_embedding = image_features.cpu().numpy()[0]
            return image_embedding / np.linalg.norm(image_embedding)
        
        # Handle image paths
        if isinstance(image, str) and os.path.exists(image):
            img = Image.open(image)
            return self.encode_image(img)
        
        return None
    
    def encode_multimodal(self, text: str = "", image = None) -> np.ndarray:
        """Encode both text and image if available into a combined embedding."""
        embeddings = []
        
        if text:
            text_embedding = self.encode_text(text)
            embeddings.append(text_embedding)
        
        if image is not None:
            image_embedding = self.encode_image(image)
            if image_embedding is not None:
                embeddings.append(image_embedding)
        
        # If we have both text and image embeddings, average them
        if len(embeddings) > 1:
            return np.mean(embeddings, axis=0)
        elif len(embeddings) == 1:
            return embeddings[0]
        else:
            return np.zeros(self.embedding_dim)


class VectorDatabase:
    """FAISS-based vector database for efficient similarity search."""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine similarity for normalized vectors)
        self.documents = []  # Store document content (text, image paths, etc.)
    
    def add_document(self, doc_id: str, embedding: np.ndarray, content: Dict):
        """Add a document to the vector database."""
        if embedding is None or not isinstance(embedding, np.ndarray):
            return False
        
        # Normalize the embedding for cosine similarity
        embedding = embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(embedding)
        
        # Add to FAISS index
        self.index.add(embedding)
        
        # Store document content
        self.documents.append({
            "id": doc_id,
            "content": content
        })
        
        return True
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar documents based on embedding."""
        if self.index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx != -1:
                results.append({
                    **self.documents[idx],
                    "score": float(scores[0][i])
                })
        
        return results
    
    def save(self, filepath: str):
        """Save the vector database to disk."""
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save documents
        with open(f"{filepath}.docs", "wb") as f:
            pickle.dump(self.documents, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Load vector database from disk."""
        db = cls()
        
        # Load FAISS index if exists
        if os.path.exists(f"{filepath}.index"):
            db.index = faiss.read_index(f"{filepath}.index")
        
        # Load documents if exists
        if os.path.exists(f"{filepath}.docs"):
            with open(f"{filepath}.docs", "rb") as f:
                db.documents = pickle.load(f)
        
        return db


class GroqLLM:
    """Interface to the Groq LLM API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def call_groq_api(self, messages: List[Dict[str, str]], max_retries: int = 3) -> Optional[str]:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": "llama3-8b-8192", "messages": messages, "max_tokens": 1000, "temperature": 0.2}
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    st.error(f"Error calling Groq API: {str(e)}")
                    return None
                time.sleep(1)  # Wait before retrying
    
    def generate_response(self, query: str, retrieved_docs: List[Dict], image_base64: Optional[str] = None) -> str:
        """Generate a response based on the query and retrieved documents."""
        # Construct prompt with retrieved documents
        context = ""
        for i, doc in enumerate(retrieved_docs):
            doc_content = doc["content"]
            if "text" in doc_content:
                context += f"Document {i+1} (Text): {doc_content['text']}\n\n"
            if "image_desc" in doc_content:
                context += f"Document {i+1} (Image Description): {doc_content['image_desc']}\n\n"
        
        messages = [
            {"role": "system", "content": "You are a helpful multimodal AI assistant that answers queries based on retrieved information from documents and images. Provide informative, accurate, and concise responses."},
            {"role": "user", "content": f"Here is my query: {query}\n\nRetrieved information:\n{context}"}
        ]
        
        # If an image is provided with the query, include it
        if image_base64:
            # Note: Groq currently doesn't support image inputs directly
            # This is a placeholder for when Groq adds multimodal capabilities
            # For now, we'll just note that there was an image in the query
            messages[1]["content"] += "\n\nNote: The query included an image which I'll consider in my response."
        
        return self.call_groq_api(messages) or "I couldn't generate a response. Please try again."


class MultimodalRAGApp:
    """Main application class for the Multimodal RAG system."""
    
    def __init__(self):
        self.encoder = MultimodalEncoder()
        self.vector_db = VectorDatabase(embedding_dim=self.encoder.embedding_dim)
        self.llm = GroqLLM(api_key=GROQ_API_KEY)
        
        # Load existing database if available
        self.load_database()
    
    def load_database(self):
        """Load the vector database from disk."""
        try:
            if os.path.exists(f"{VECTOR_DB_PATH}.index"):
                self.vector_db = VectorDatabase.load(VECTOR_DB_PATH)
                st.sidebar.success(f"Loaded database with {self.vector_db.index.ntotal} documents")
            else:
                st.sidebar.info("No existing database found. Start by adding documents.")
        except Exception as e:
            st.sidebar.error(f"Error loading database: {str(e)}")
    
    def save_database(self):
        """Save the vector database to disk."""
        try:
            self.vector_db.save(VECTOR_DB_PATH)
            st.sidebar.success("Database saved successfully")
        except Exception as e:
            st.sidebar.error(f"Error saving database: {str(e)}")
    
    def add_document(self, text: str = "", image = None, image_desc: str = ""):
        """Add a document (text and/or image) to the database."""
        doc_id = str(uuid.uuid4())
        content = {}
        
        if text:
            content["text"] = text
        
        if image is not None:
            # Save image to disk
            image_path = os.path.join(DATA_DIR, f"{doc_id}.jpg")
            image.save(image_path)
            content["image_path"] = image_path
            
            if image_desc:
                content["image_desc"] = image_desc
        
        # Create embedding
        embedding = self.encoder.encode_multimodal(text, image)
        
        # Add to vector database
        if self.vector_db.add_document(doc_id, embedding, content):
            # Save the updated database
            self.save_database()
            return True
        
        return False
    
    def process_query(self, query: str, query_image = None, top_k: int = 3) -> Tuple[str, List[Dict]]:
        """Process a user query and return a response with retrieved documents."""
        # Encode the query
        query_embedding = self.encoder.encode_multimodal(query, query_image)
        
        # Search for relevant documents
        retrieved_docs = self.vector_db.search(query_embedding, top_k=top_k)
        
        # Prepare image for LLM if provided
        image_base64 = None
        if query_image:
            buffered = io.BytesIO()
            query_image.save(buffered, format="JPEG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Generate response
        response = self.llm.generate_response(query, retrieved_docs, image_base64)
        
        return response, retrieved_docs


def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def run_app():
    """Main Streamlit application."""
    st.set_page_config(page_title="Multimodal RAG System", layout="wide")
    
    st.title("Multimodal RAG System")
    
    # Initialize application
    if "app" not in st.session_state:
        st.session_state.app = MultimodalRAGApp()
    
    # Sidebar for adding documents
    st.sidebar.title("Add Documents")
    
    doc_tab1, doc_tab2 = st.sidebar.tabs(["Text", "Image"])
    
    with doc_tab1:
        doc_text = st.text_area("Document Text", height=150)
        if st.button("Add Text Document"):
            if doc_text:
                if st.session_state.app.add_document(text=doc_text):
                    st.success("Text document added successfully!")
                else:
                    st.error("Failed to add document.")
            else:
                st.warning("Please enter some text.")
    
    with doc_tab2:
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        image_desc = st.text_area("Image Description (optional)", height=100)
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Add Image Document"):
                if st.session_state.app.add_document(text=image_desc, image=image, image_desc=image_desc):
                    st.success("Image document added successfully!")
                else:
                    st.error("Failed to add document.")
    
    # Main query interface
    st.header("Query the Knowledge Base")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_input("Enter your query")
    
    with col2:
        query_image = st.file_uploader("Include an image with your query (optional)", type=["jpg", "jpeg", "png"])
    
    top_k = st.slider("Number of documents to retrieve", min_value=1, max_value=10, value=3)
    
    if query_image is not None:
        query_img = Image.open(query_image).convert("RGB")
        st.image(query_img, caption="Query Image", width=300)
    else:
        query_img = None
    
    if st.button("Submit Query"):
        if query or query_img:
            with st.spinner("Processing query..."):
                response, retrieved_docs = st.session_state.app.process_query(query, query_img, top_k)
                
                st.header("Response")
                st.write(response)
                
                st.header("Retrieved Documents")
                for i, doc in enumerate(retrieved_docs):
                    with st.expander(f"Document {i+1} (Score: {doc['score']:.4f})"):
                        if "text" in doc["content"]:
                            st.write("Text:", doc["content"]["text"])
                        
                        if "image_path" in doc["content"]:
                            try:
                                img = Image.open(doc["content"]["image_path"])
                                st.image(img, caption="Document Image", width=300)
                                
                                if "image_desc" in doc["content"]:
                                    st.write("Image Description:", doc["content"]["image_desc"])
                            except Exception as e:
                                st.error(f"Error loading image: {str(e)}")
        else:
            st.warning("Please enter a query or upload an image.")
    
    # Add database statistics
    st.sidebar.title("Database Statistics")
    st.sidebar.metric("Documents in Database", st.session_state.app.vector_db.index.ntotal)
    
    # Feedback mechanism
    st.sidebar.title("Feedback")
    feedback = st.sidebar.text_area("Provide feedback on the response quality")
    if st.sidebar.button("Submit Feedback"):
        if feedback:
            # In a real application, this would be stored and used for improvement
            st.sidebar.success("Thank you for your feedback!")
        else:
            st.sidebar.warning("Please enter feedback before submitting.")


# Main entry point
if __name__ == "__main__":
    run_app()