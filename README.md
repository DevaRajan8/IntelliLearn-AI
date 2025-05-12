# 🚀 Tech Skill Navigator

## Overview

Tech Skill Navigator is a multimodal learning application that helps tech enthusiasts explore and understand technological concepts through various interaction modes: text, voice, and image-based learning.

## Features

- **Text Exploration**: Enter tech concepts and receive comprehensive explanations
- **Voice Query**: Use speech recognition to explore tech topics
- **Image-Based Learning**: Upload images to get object detection and tech insights

## Prerequisites

- Python 3.8+
- API Keys:
  - Groq API Key (Required)
  - Hugging Face API Key (Recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Devarajan8/IntelliLearn-AI.git
cd tech-skill-navigator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Create a `.env` file in the project root
- Add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
huggingface_api_key=your_hf_api_key
```

## Running the Application

```bash
streamlit run main.py
```

## Dependencies

- Streamlit
- Requests
- SpeechRecognition
- NumPy
- Pillow
- OpenCV
- PyAudio

## API Services Used

- Groq API (for language model interactions)
- Hugging Face Inference API (for object detection)
- Google Speech Recognition

## Modes of Learning

1. **Text Exploration**
   - Enter tech concepts
   - Get detailed explanations
   - Receive learning resources

2. **Voice Query**
   - Speak your tech query
   - Get instant explanations

3. **Image-Based Learning**
   - Upload tech-related images
   - Detect objects
   - Generate insights about detected objects

## Troubleshooting

- Ensure all dependencies are installed
- Check API key configurations
- Verify microphone permissions for voice queries

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/tech-skill-navigator](https://github.com/yourusername/tech-skill-navigator)
