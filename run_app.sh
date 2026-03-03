#!/bin/zsh
# Word Counter App Launcher
set -e

# Navigate to app directory
cd "$(dirname "$0")"

# Check virtual environment exists
if [[ ! -d "venv" ]]; then
    echo "Error: Virtual environment not found. Run:"
    echo "  python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check required packages are installed
if ! python -c "import vosk, pyaudio, jellyfish" 2>/dev/null; then
    echo "Missing dependencies. Installing from requirements.txt..."
    pip install -r requirements.txt
fi

# Check Vosk model exists
if [[ ! -d "vosk-model-small-en-us-0.15" ]]; then
    echo "Vosk speech model not found. Downloading (~40 MB)..."
    curl -L -o vosk-model.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    unzip -q vosk-model.zip && rm vosk-model.zip
    echo "Model downloaded successfully."
fi

# Run the application
python word_counter.py
