import os
import streamlit as st
import requests
import speech_recognition as sr
import numpy as np
import io
from PIL import Image
import base64

class TechSkillNavigator:
    def __init__(self):
        # Set up API credentials
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.huggingface_api_key = "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"
        
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        
        # Validate API keys
        if not self.groq_api_key:
            st.error("GROQ_API_KEY not found. Please set it in Streamlit secrets.")
        if not self.huggingface_api_key:
            st.warning("HUGGINGFACE_API_KEY not set. Some features may be limited.")
    
    def call_groq_api(self, messages: list, max_retries: int = 3) -> str:
        """Call Groq API with robust error handling and retry mechanism."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-8b-8192",
            "messages": messages,
            "max_tokens": 1500,
            "temperature": 0.3
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content']
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    st.error(f"API call failed after {max_retries} attempts: {e}")
                    return None
    
    def detect_objects_in_image(self, image):
        """Detect objects using Hugging Face Inference API."""
        try:
            # Convert image to base64
            pil_image = Image.open(image)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Call Hugging Face Inference API
            headers = {
                "Authorization": f"Bearer {self.huggingface_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": img_str,
                "model": "facebook/detr-resnet-50"
            }
            
            response = requests.post(
                "https://api-inference.huggingface.co/models/facebook/detr-resnet-50", 
                headers=headers, 
                json=payload
            )
            
            if response.status_code != 200:
                st.error(f"Object detection API error: {response.text}")
                return []
            
            results = response.json()
            
            # Process results
            detected_objects = []
            for result in results:
                detected_objects.append({
                    'label': result['label'],
                    'confidence': result['score']
                })
            
            return detected_objects
        
        except Exception as e:
            st.error(f"Error in object detection: {e}")
            return []
    
    def transcribe_audio(self):
        """Transcribe audio input using speech recognition."""
        try:
            with sr.Microphone() as source:
                st.info("Listening... Speak your tech query")
                audio = self.recognizer.listen(source, timeout=5)
                
                try:
                    text = self.recognizer.recognize_google(audio)
                    return text
                except sr.UnknownValueError:
                    st.warning("Could not understand audio")
                    return None
                except sr.RequestError:
                    st.error("Could not request results from speech recognition service")
                    return None
        except Exception as e:
            st.error(f"Error in audio transcription: {e}")
            return None
    
    def generate_tech_explanation(self, query: str, context: dict = None):
        """Generate detailed tech concept explanation."""
        messages = [
            {
                "role": "system",
                "content": "You are an expert tech mentor specializing in explaining complex technological concepts in an engaging, accessible manner. Adapt your explanation to the user's context and learning style."
            },
            {
                "role": "user",
                "content": f"""Provide a comprehensive explanation of the following tech concept:
                {query}
                
                {'Additional Context: ' + str(context) if context else ''}
                
                Explanation Guidelines:
                - Use simple, clear language
                - Provide real-world examples
                - Suggest practical learning resources
                - Highlight key takeaways"""
            }
        ]
        
        return self.call_groq_api(messages)
    
    def render_app(self):
        """Main Streamlit app rendering."""
        st.title("🚀 Tech Skill Navigator")
        st.markdown("""
        ### Multimodal Learning for Tech Enthusiasts
        Explore technology concepts through voice, text, and image interactions!
        """)
        
        # Sidebar for navigation
        mode = st.sidebar.radio("Choose Your Learning Mode", 
            ["Text Exploration", "Voice Query", "Image-Based Learning"]
        )
        
        if mode == "Text Exploration":
            self.text_exploration_section()
        elif mode == "Voice Query":
            self.voice_query_section()
        else:
            self.image_learning_section()
    
    def text_exploration_section(self):
        """Section for text-based tech concept exploration."""
        st.subheader("📝 Text-Driven Tech Learning")
        tech_query = st.text_input("Enter a tech concept or technology you want to learn about")
        
        # Optional context selection
        learning_context = st.selectbox("Select Your Learning Context", [
            "Beginner", 
            "Professional Development", 
            "Academic Research", 
            "Hobbyist Interest"
        ])
        
        if st.button("Explore Concept"):
            if tech_query:
                with st.spinner("Generating explanation..."):
                    explanation = self.generate_tech_explanation(
                        tech_query, 
                        context={"learning_level": learning_context}
                    )
                    
                    st.markdown("#### 🔍 Tech Concept Breakdown")
                    st.markdown(explanation or "Could not generate explanation.")
                    
                    # Additional learning resources
                    st.markdown("#### 📚 Recommended Resources")
                    resources = self.generate_learning_resources(tech_query)
                    st.markdown(resources or "No resources found.")
            else:
                st.warning("Please enter a tech concept to explore.")
    
    def voice_query_section(self):
        """Section for voice-based tech queries."""
        st.subheader("🎙️ Voice-Powered Tech Learning")
        
        if st.button("Start Voice Query"):
            # Transcribe audio input
            voice_query = self.transcribe_audio()
            
            if voice_query:
                st.write(f"Detected Query: {voice_query}")
                
                with st.spinner("Processing your voice query..."):
                    explanation = self.generate_tech_explanation(voice_query)
                    
                    st.markdown("#### 🔊 Voice Query Explanation")
                    st.markdown(explanation or "Could not generate explanation.")
            else:
                st.warning("No voice query detected.")
    
    def image_learning_section(self):
        """Section for image-based tech learning."""
        st.subheader("🖼️ Image-Driven Tech Insights")
        uploaded_image = st.file_uploader("Upload a tech-related image", 
                                          type=['png', 'jpg', 'jpeg'])
        
        if uploaded_image:
            # Display uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Image"):
                # Detect objects in the image
                with st.spinner("Detecting objects and generating insights..."):
                    detected_objects = self.detect_objects_in_image(uploaded_image)
                    
                    st.markdown("#### 🔬 Detected Objects")
                    for obj in detected_objects:
                        st.markdown(f"- {obj['label']} (Confidence: {obj['confidence']:.2f})")
                    
                    # Generate tech explanations based on detected objects
                    if detected_objects:
                        for obj in detected_objects:
                            explanation = self.generate_tech_explanation(obj['label'])
                            st.markdown(f"#### 💡 {obj['label']} Tech Insights")
                            st.markdown(explanation or "Could not generate explanation.")
    
    def generate_learning_resources(self, topic: str):
        """Generate learning resources for a given tech topic."""
        messages = [
            {
                "role": "system",
                "content": "You are an expert tech learning resource curator."
            },
            {
                "role": "user",
                "content": f"""Suggest 3-5 learning resources for the following tech topic:
                {topic}
                
                For each resource, provide:
                - Resource name
                - Type (Online Course, Tutorial, Book, YouTube Channel)
                - Brief description
                - Skill level (Beginner/Intermediate/Advanced)"""
            }
        ]
        
        return self.call_groq_api(messages)

def main():
    # Initialize and run the Tech Skill Navigator
    app = TechSkillNavigator()
    app.render_app()

if __name__ == "__main__":
    main()