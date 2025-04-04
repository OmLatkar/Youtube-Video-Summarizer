import streamlit as st
import yt_dlp
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import whisper
import tempfile
import os
import nltk
import torch
import asyncio
import sys
from io import StringIO

# Windows event loop fix
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Custom CSS for layout
st.set_page_config(
    page_title="Video Summarizer",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
st.markdown("""
    <style>
        /* Base font size increase */
        html, body, [class*="css"]  {
            font-size: 18px !important;
        }
        
        /* Main content area */
        .main .block-container {
            max-width: 95%;
            padding-top: 2.5rem;
            padding-bottom: 2.5rem;
        }
        
        /* Sidebar enhancements */
        [data-testid="stSidebar"] {
            width: 400px !important;
            min-width: 400px !important;
        }
        .sidebar .sidebar-content {
            padding: 2.5rem 2rem;
        }
        .sidebar-title {
            font-size: 1.8rem !important;
            font-weight: bold !important;
            margin-bottom: 2rem !important;
        }
        .sidebar-section {
            margin-bottom: 2.5rem;
        }
        .sidebar-instructions {
            font-size: 1.3rem !important;
        }
        
        /* Title and headers */
        h1 {
            font-size: 2.8rem !important;
        }
        h2 {
            font-size: 2.2rem !important;
        }
        h3 {
            font-size: 1.8rem !important;
        }
        
        /* Input fields */
        .stTextInput input, .stTextArea textarea {
            font-size: 1.3rem !important;
            padding: 0.8rem !important;
        }
        
        /* Slider enhancements */
        div[data-baseweb="slider"] {
            padding: 1.5rem 0 !important;
        }
        .stSlider [role="slider"] {
            width: 25px !important;
            height: 25px !important;
        }
        .stSlider [data-testid="stThumbValue"] {
            font-size: 1.3rem !important;
        }
        
        /* Text areas - especially important for summary */
        .stTextArea [data-baseweb=base-input] {
            min-height: 300px;
            font-size: 1.4rem !important;
            line-height: 1.6 !important;
            padding: 1.2rem !important;
        }
        /* Scrollbar styling */
        .stTextArea textarea::-webkit-scrollbar {
            width: 15px !important;
            height: 15px !important;
        }
        .stTextArea textarea::-webkit-scrollbar-thumb {
            background-color: #4a4a4a !important;
            border-radius: 10px !important;
        }
        .stTextArea textarea::-webkit-scrollbar-track {
            background-color: #f0f0f0 !important;
        }
        
        /* Buttons */
        .stButton>button {
            width: 100%;
            padding: 1rem 1.5rem;
            font-size: 1.4rem !important;
            height: auto !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 1rem 2rem;
            font-size: 1.4rem !important;
            height: auto !important;
        }
        
        /* Download button */
        .download-btn {
            margin-top: 2rem;
            font-size: 1.4rem !important;
            padding: 1rem !important;
        }
        
        /* Footer */
        .stCaption {
            font-size: 1.2rem !important;
        }
        
        /* Spacing adjustments */
        .stSpinner {
            margin: 2rem 0 !important;
        }
        .stMarkdown {
            margin: 1.5rem 0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Ensure NLTK punkt tokenizer is downloaded
nltk.download('punkt')

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("base", device=device)

model = load_whisper_model()

def summarize_text(text, sentences_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return ' '.join(str(sentence) for sentence in summary)

def download_youtube_audio(youtube_url):
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': tmp_path.replace('.mp3', ''),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            st.info("Downloading YouTube audio...")
            ydl.download([youtube_url])
        
        return tmp_path.replace('.mp3', '') + '.mp3'
    except Exception as e:
        st.error(f"YouTube download failed: {str(e)}")
        return None

def process_audio(audio_path, sentences_count):
    try:
        with st.spinner("Transcribing..."):
            result = model.transcribe(audio_path)
            text = result["text"]
        
        with st.spinner("Generating summary..."):
            summary = summarize_text(text, sentences_count)
        
        return summary
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None

# Sidebar - Settings
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Settings</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**Summary Length**", help="Adjust how many sentences the summary should contain")
        sentences_count = st.slider(
            " ",
            min_value=1,
            max_value=10,
            value=3,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-instructions">', unsafe_allow_html=True)
        st.markdown("**How to use:**")
        st.markdown("1. Enter YouTube URL or upload file")
        st.markdown("2. Click process button")
        st.markdown("3. View and download your summary")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Main Content
st.title("🎥 Video Summarizer")
st.markdown("Extract key insights from videos using AI")

# Create tabs
tab1, tab2 = st.tabs(["📺 YouTube Video", "📁 Local File"])

# Session state to store summary
if 'summary' not in st.session_state:
    st.session_state.summary = None

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        youtube_url = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed"
        )
    with col2:
        if st.button("Process ▶️", key="youtube_btn"):
            if not youtube_url:
                st.error("Please enter a YouTube URL")
            else:
                audio_path = download_youtube_audio(youtube_url)
                if audio_path:
                    st.session_state.summary = process_audio(audio_path, sentences_count)
                    os.unlink(audio_path)

with tab2:
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["mp3", "mp4", "wav", "m4a"],
        label_visibility="collapsed"
    )
    if uploaded_file and st.button("Process ▶️", key="file_btn"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        st.session_state.summary = process_audio(tmp_path, sentences_count)
        os.unlink(tmp_path)

# Display summary and download button
if st.session_state.summary:
    st.subheader("Summary")
    st.text_area(
        "Summary Output",
        value=st.session_state.summary,
        height=400,  # Increased from 200 to 400
        label_visibility="collapsed",
        key="summary_output"
    )
    
    # Download button
    st.download_button(
        label="📥 Download Summary",
        data=st.session_state.summary,
        file_name="video_summary.txt",
        mime="text/plain",
        key="download_btn",
        help="Click to download the summary as a text file"
    )

# Footer
st.markdown("---")
st.caption("Powered by OpenAI Whisper and Streamlit")