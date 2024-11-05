import os
import openai
import whisper
import tempfile
import gradio as gr
from pydub import AudioSegment
import fitz  # PyMuPDF for handling PDFs
import docx  # For handling .docx files
import pandas as pd  # For handling .xlsx and .csv files
import requests
from bs4 import BeautifulSoup
from moviepy.editor import VideoFileClip
import yt_dlp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the highest quality Whisper model once
model = whisper.load_model("large")

def download_social_media_video(url):
    """Downloads a video from social media."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': '%(id)s.%(ext)s',
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            audio_file = f"{info_dict['id']}.mp3"
        logger.info(f"Video successfully downloaded: {audio_file}")
        return audio_file
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        raise

def convert_video_to_audio(video_file):
    """Converts a video file to audio."""
    try:
        video = VideoFileClip(video_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            video.audio.write_audiofile(temp_file.name)
            logger.info(f"Video converted to audio: {temp_file.name}")
            return temp_file.name
    except Exception as e:
        logger.error(f"Error converting video to audio: {str(e)}")
        raise

def preprocess_audio(audio_file):
    """Preprocesses the audio file to improve quality."""
    try:
        audio = AudioSegment.from_file(audio_file)
        audio = audio.apply_gain(-audio.dBFS + (-20))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            audio.export(temp_file.name, format="mp3")
            logger.info(f"Audio preprocessed: {temp_file.name}")
            return temp_file.name
    except Exception as e:
        logger.error(f"Error preprocessing audio file: {str(e)}")
        raise

def transcribe_audio(file):
    """Transcribes an audio or video file."""
    try:
        if isinstance(file, str) and file.startswith('http'):
            logger.info(f"Downloading social media video: {file}")
            file_path = download_social_media_video(file)
        elif isinstance(file, str) and file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            logger.info(f"Converting local video to audio: {file}")
            file_path = convert_video_to_audio(file)
        else:
            logger.info(f"Preprocessing audio file: {file}")
            file_path = preprocess_audio(file)

        logger.info(f"Transcribing audio: {file_path}")
        result = model.transcribe(file_path)
        transcription = result.get("text", "Error in transcription")
        logger.info(f"Transcription completed: {transcription[:50]}...")
        return transcription
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return f"Error processing file: {str(e)}"

def read_document(document_path):
    """Reads content from PDF, DOCX, XLSX or CSV documents."""
    try:
        if document_path.endswith(".pdf"):
            doc = fitz.open(document_path)
            return "\n".join([page.get_text() for page in doc])
        elif document_path.endswith(".docx"):
            doc = docx.Document(document_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        elif document_path.endswith(".xlsx"):
            return pd.read_excel(document_path).to_string()
        elif document_path.endswith(".csv"):
            return pd.read_csv(document_path).to_string()
        else:
            return "Unsupported file type. Please upload a PDF, DOCX, XLSX or CSV document."
    except Exception as e:
        return f"Error reading document: {str(e)}"

def read_url(url):
    """Reads content from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        return f"Error reading URL: {str(e)}"

def process_social_content(url):
    """Processes content from a social media URL, handling both text and video."""
    try:
        # First, try to read content as text
        text_content = read_url(url)

        # Then, try to process as video
        try:
            video_content = transcribe_audio(url)
        except Exception:
            video_content = None

        return {
            "text": text_content,
            "video": video_content
        }
    except Exception as e:
        logger.error(f"Error processing social content: {str(e)}")
        return None

def generate_news(instructions, facts, size, tone, *args):
    """Generates a news article from instructions, facts, URLs, documents, transcriptions, and social media content."""
    knowledge_base = {
        "instructions": instructions,
        "facts": facts,
        "document_content": [],
        "audio_data": [],
        "url_content": [],
        "social_content": []
    }
    num_audios = 5 * 3  # 5 audios/videos * 3 fields (file, name, position)
    num_social_urls = 3 * 3  # 3 social media URLs * 3 fields (URL, name, context)
    num_urls = 5  # 5 general URLs
    audios = args[:num_audios]
    social_urls = args[num_audios:num_audios+num_social_urls]
    urls = args[num_audios+num_social_urls:num_audios+num_social_urls+num_urls]
    documents = args[num_audios+num_social_urls+num_urls:]

    for url in urls:
        if url:
            knowledge_base["url_content"].append(read_url(url))

    for document in documents:
        if document is not None:
            knowledge_base["document_content"].append(read_document(document.name))

    for i in range(0, len(audios), 3):
        audio_file, name, position = audios[i:i+3]
        if audio_file is not None:
            knowledge_base["audio_data"].append({"audio": audio_file, "name": name, "position": position})

    for i in range(0, len(social_urls), 3):
        social_url, social_name, social_context = social_urls[i:i+3]
        if social_url:
            social_content = process_social_content(social_url)
            if social_content:
                knowledge_base["social_content"].append({
                    "url": social_url,
                    "name": social_name,
                    "context": social_context,
                    "text": social_content["text"],
                    "video": social_content["video"]
                })
                logger.info(f"Social media content processed: {social_url}")

    transcriptions_text, raw_transcriptions = "", ""

    for idx, data in enumerate(knowledge_base["audio_data"]):
        if data["audio"] is not None:
            transcription = transcribe_audio(data["audio"])
            transcription_text = f'"{transcription}" - {data["name"]}, {data["position"]}'
            raw_transcription = f'[Audio/Video {idx + 1}]: "{transcription}" - {data["name"]}, {data["position"]}'
            transcriptions_text += transcription_text + "\n"
            raw_transcriptions += raw_transcription + "\n\n"

    for data in knowledge_base["social_content"]:
        if data["text"]:
            transcription_text = f'[Social media text]: "{data["text"][:200]}..." - {data["name"]}, {data["context"]}'
            transcriptions_text += transcription_text + "\n"
            raw_transcriptions += transcription_text + "\n\n"
        if data["video"]:
            transcription_video = f'[Social media video]: "{data["video"]}" - {data["name"]}, {data["context"]}'
            transcriptions_text += transcription_video + "\n"
            raw_transcriptions += transcription_video + "\n\n"

    document_content = "\n\n".join(knowledge_base["document_content"])
    url_content = "\n\n".join(knowledge_base["url_content"])

    internal_prompt = """
    Instructions for the model:
    - Follow news article principles: answer the 5 Ws in the first paragraph (Who?, What?, When?, Where?, Why?).
    - Ensure at least 80% of quotes are direct and in quotation marks.
    - The remaining 20% can be indirect quotes.
    - Don't invent new information.
    - Be rigorous with provided facts.
    - When processing uploaded documents, extract and highlight important quotes and testimonials from sources.
    - When processing uploaded documents, extract and highlight key figures.
    - Avoid using the date at the beginning of the news body. Start directly with the 5Ws.
    - Include social media content relevantly, citing the source and providing proper context.
    - Make sure to relate the provided context for social media content with its corresponding transcription or text.
    """

    prompt = f"""
    {internal_prompt}
    Write a news article with the following information, including a title, a 15-word hook (additional information that complements the title), and the content body with {size} words. The tone should be {tone}.
    Instructions: {knowledge_base["instructions"]}
    Facts: {knowledge_base["facts"]}
    Additional content from documents: {document_content}
    Additional content from URLs: {url_content}
    Use the following transcriptions as direct and indirect quotes (without changing or inventing content):
    {transcriptions_text}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        news = response['choices'][0]['message']['content']
        return news, raw_transcriptions
    except Exception as e:
        logger.error(f"Error generating news article: {str(e)}")
        return f"Error generating news article: {str(e)}", ""

with gr.Blocks() as demo:
    gr.Markdown("## All-in-One News Generator")
    
    # Add tool description and attribution
    gr.Markdown("""
    ### About this tool
    
    This AI-powered news generator helps journalists and content creators produce news articles by processing multiple types of input:
    - Audio and video files with automatic transcription
    - Social media content
    - Documents (PDF, DOCX, XLSX, CSV)
    - Web URLs
    
    The tool uses advanced AI to generate well-structured news articles following journalistic principles and maintaining the integrity of source quotes.
    
    Created by [Camilo Vega](https://www.linkedin.com/in/camilo-vega-169084b1/), AI Consultant
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            instructions = gr.Textbox(label="News article instructions", lines=2)
            facts = gr.Textbox(label="Describe the news facts", lines=4)
            size = gr.Number(label="Content body size (in words)", value=100)
            tone = gr.Dropdown(label="News tone", choices=["serious", "neutral", "lighthearted"], value="neutral")
        with gr.Column(scale=3):
            inputs_list = [instructions, facts, size, tone]
            with gr.Tabs():
                for i in range(1, 6):
                    with gr.TabItem(f"Audio/Video {i}"):
                        file = gr.File(label=f"Audio/Video {i}", type="filepath", file_types=["audio", "video"])
                        name = gr.Textbox(label="Name", scale=1)
                        position = gr.Textbox(label="Position", scale=1)
                        inputs_list.extend([file, name, position])
                for i in range(1, 4):
                    with gr.TabItem(f"Social Media {i}"):
                        social_url = gr.Textbox(label=f"Social media URL {i}", lines=1)
                        social_name = gr.Textbox(label=f"Person/account name {i}", scale=1)
                        social_context = gr.Textbox(label=f"Content context {i}", lines=2)
                        inputs_list.extend([social_url, social_name, social_context])
                for i in range(1, 6):
                    with gr.TabItem(f"URL {i}"):
                        url = gr.Textbox(label=f"URL {i}", lines=1)
                        inputs_list.append(url)
                for i in range(1, 6):
                    with gr.TabItem(f"Document {i}"):
                        document = gr.File(label=f"Document {i}", type="filepath", file_count="single")
                        inputs_list.append(document)

    gr.Markdown("---")  # Visual separator

    with gr.Row():
        transcriptions_output = gr.Textbox(label="Transcriptions", lines=10)

    gr.Markdown("---")  # Visual separator

    with gr.Row():
        generate = gr.Button("Generate Draft")
    with gr.Row():
        news_output = gr.Textbox(label="Generated Draft", lines=20)

    generate.click(fn=generate_news, inputs=inputs_list, outputs=[news_output, transcriptions_output])

demo.launch(share=True)