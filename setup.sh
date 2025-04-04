#!/bin/bash
# Install FFmpeg (required for audio extraction)
sudo apt-get update && sudo apt-get install -y ffmpeg

# Download NLTK punkt tokenizer
python -c "import nltk; nltk.download('punkt')"
