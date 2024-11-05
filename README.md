# NoticiasAI 📚

Try the tool now: [https://huggingface.co/spaces/NoticIA-Col/Generador-Noticias](https://huggingface.co/spaces/NoticIA-Col/Generador-Noticias)

An AI-powered news article generator that processes multiple types of content using advanced language models and RAG (Retrieval-Augmented Generation) techniques.

## 🌟 Features

- **Multi-Source Content Processing**:
  - 🎥 Audio/Video transcription using OpenAI Whisper
  - 📱 Social media content extraction
  - 📄 Document parsing (PDF, DOCX, XLSX, CSV)
  - 🌐 Web content retrieval
  - 📊 Structured data processing

- **Advanced RAG Implementation**:
  - Knowledge base construction from multiple sources
  - Context-aware content generation
  - Semantic retrieval of relevant information
  - Source attribution and quote preservation

- **Journalistic Standards**:
  - Automated 5W's implementation (Who, What, When, Where, Why)
  - Direct and indirect quote management
  - Flexible tone adjustment
  - Customizable content length

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/NoticiasAI.git
cd NoticiasAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
apt-get install ffmpeg  # For audio processing
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

4. Run the application:
```bash
python app.py
```

## 🛠️ Technology Stack

- **Core Framework**: Gradio 4.31.5
- **Language Models**: OpenAI GPT-4
- **Audio Processing**: OpenAI Whisper
- **Document Processing**:
  - PyMuPDF (PDF)
  - python-docx (DOCX)
  - pandas (XLSX, CSV)
- **Media Processing**:
  - moviepy
  - yt-dlp
  - pydub
- **Web Scraping**:
  - BeautifulSoup4
  - requests

## 💡 How It Works

1. **Content Ingestion**:
   - Processes multiple input types simultaneously
   - Converts media to text using state-of-the-art models
   - Extracts relevant information from documents and web content

2. **Knowledge Base Construction**:
   - Organizes extracted information into a structured format
   - Maintains source attribution and context
   - Preserves quotes and testimonials

3. **Content Generation**:
   - Applies RAG techniques to combine retrieved information
   - Follows journalistic principles and formatting
   - Generates coherent, fact-based articles

## 🎯 Use Cases

- **Journalism**: Rapidly create news drafts from multiple sources
- **Content Creation**: Generate well-structured articles with proper attribution
- **Research**: Compile information from various sources into coherent narratives
- **Documentation**: Transform multiple content types into structured documents

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

[Camilo Vega](https://www.linkedin.com/in/camilo-vega-169084b1/) - AI Consultant

## 🙏 Acknowledgments

- OpenAI for GPT-4 and Whisper
- Hugging Face for hosting support
- Gradio team for the amazing UI framework
