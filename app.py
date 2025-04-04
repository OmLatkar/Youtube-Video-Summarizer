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

# Modern, clean color scheme
st.markdown("""
    <style>
        /* Modern color palette */
        :root {
            --primary: #4361ee;
            --primary-light: #4895ef;
            --secondary: #3f37c9;
            --accent: #f72585;
            --dark: #1a1a2e;
            --light: #f8f9fa;
            --success: #4cc9f0;
            --warning: #f8961e;
            --danger: #ef233c;
            --gray: #6c757d;
        }
        
        /* Base styles */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: var(--dark);
            background-color: var(--light);
        }
        
        /* Main content area */
        .main .block-container {
            max-width: 95%;
            padding: 2rem;
            background-color: white;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            margin: 1rem;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
            width: 320px !important;
            border-right: none;
        }
        .sidebar-title {
            color: var(--primary);
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            margin-bottom: 1.5rem !important;
        }
        
        /* Input fields */
        .stTextInput input, .stTextArea textarea {
            border-radius: 12px !important;
            border: 2px solid #e9ecef !important;
            padding: 0.75rem 1rem !important;
            font-size: 1rem !important;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.15) !important;
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(67, 97, 238, 0.2) !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            margin-bottom: 1.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            padding: 0.75rem 1.5rem !important;
            border-radius: 12px !important;
            transition: all 0.3s ease !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(67, 97, 238, 0.1) !important;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(67, 97, 238, 0.1) !important;
            color: var(--primary) !important;
            font-weight: 600 !important;
        }
        
        /* Slider */
        .stSlider [role="slider"] {
            background-color: var(--primary) !important;
        }
        .stSlider [data-testid="stThumbValue"] {
            color: var(--primary) !important;
            font-weight: 500 !important;
        }
        
        /* Text areas */
        .stTextArea [data-baseweb=base-input] {
            min-height: 300px;
            border-radius: 12px !important;
            padding: 1.25rem !important;
            font-size: 1rem !important;
            line-height: 1.6 !important;
        }
        
        /* Download button */
        .download-btn {
            background: linear-gradient(135deg, var(--success) 0%, #3a86ff 100%) !important;
            margin-top: 1.5rem;
        }
        
        /* Status messages */
        .stAlert {
            border-radius: 12px !important;
            border-left: none !important;
        }
        .stError {
            background-color: rgba(239, 35, 60, 0.1) !important;
            color: var(--danger) !important;
        }
        .stInfo {
            background-color: rgba(67, 97, 238, 0.1) !important;
            color: var(--primary) !important;
        }
        
        /* Progress spinner */
        .stSpinner > div {
            border-color: var(--primary) transparent transparent transparent !important;
        }
        
        /* Step indicators */
        .step {
            display: flex;
            align-items: center;
            margin-bottom: 1.25rem;
        }
        .step-number {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: var(--primary);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 0.75rem;
            font-weight: 600;
            font-size: 0.875rem;
        }
        .step-text {
            font-size: 0.95rem;
            color: var(--dark);
        }
    </style>
""", unsafe_allow_html=True)

# Ensure NLTK punkt tokenizer is downloaded
nltk.download('punkt', download_dir='/tmp/nltk_data')

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
        # Check if FFmpeg is available, install if not
        try:
            import subprocess
            # Check if ffmpeg exists
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            st.warning("FFmpeg not found - installing...")
            try:
                # Try to install ffmpeg (Linux/Ubuntu)
                subprocess.run(["apt-get", "update"], check=True)
                subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True)
            except Exception as e:
                st.error(f"Failed to install FFmpeg: {str(e)}")
                return None

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': tmp_path.replace('.mp3', ''),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'extract_flat': True,
            'no_check_certificate': True,
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
        with st.spinner("Transcribing audio..."):
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
    
    st.markdown("**Summary Length**")
    sentences_count = st.slider(
        "Select number of sentences for summary",
        min_value=1,
        max_value=10,
        value=3,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### How to use")
    st.markdown("""
        <div class="step">
            <div class="step-number">1</div>
            <div class="step-text">Enter YouTube URL or upload file</div>
        </div>
        <div class="step">
            <div class="step-number">2</div>
            <div class="step-text">Click the Process button</div>
        </div>
        <div class="step">
            <div class="step-number">3</div>
            <div class="step-text">View and download your summary</div>
        </div>
    """, unsafe_allow_html=True)

# Main Content
st.title("🎥 Video Summarizer")
st.markdown("Extract key insights from videos using AI")

# Create tabs
tab1, tab2 = st.tabs(["📺 YouTube Video", "📁 Local File"])

# Session state to store summary
if 'summary' not in st.session_state:
    st.session_state.summary = None

with tab1:
    youtube_url = st.text_input(
        "Enter YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed"
    )
    if st.button("Process Video", key="youtube_btn"):
        if not youtube_url:
            st.error("Please enter a YouTube URL")
        else:
            audio_path = download_youtube_audio(youtube_url)
            if audio_path:
                st.session_state.summary = process_audio(audio_path, sentences_count)
                os.unlink(audio_path)

with tab2:
    uploaded_file = st.file_uploader(
        "Upload audio or video file",
        type=["mp3", "mp4", "wav", "m4a"],
        label_visibility="collapsed"
    )
    if uploaded_file and st.button("Process File", key="file_btn"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        st.session_state.summary = process_audio(tmp_path, sentences_count)
        os.unlink(tmp_path)

# Display summary and download button
if st.session_state.summary:
    st.subheader("📝 Summary")
    st.text_area(
        "Summary",
        value=st.session_state.summary,
        height=300,
        label_visibility="collapsed"
    )
    
    st.download_button(
        label="Download Summary",
        data=st.session_state.summary,
        file_name="video_summary.txt",
        mime="text/plain",
        key="download_btn",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.caption("✨ Powered by OpenAI Whisper and Streamlit")
