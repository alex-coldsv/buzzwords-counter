# BuzzWords Counter

A macOS application that uses **Vosk** (offline, real-time speech recognition) and **phonetic matching** to listen to your microphone and count how many times you say a specific word. Automatically catches mis-transcriptions of abbreviations like "AI", "GPU", "ML" etc. No internet connection or API keys required.

## Features

- 🎤 Real-time streaming speech recognition (~250ms latency)
- 🔢 Live word counting with partial result tracking
- 🎯 Customizable target word (default: "AI")
- 🔊 **Automatic phonetic matching** — catches "ay", "a i", "ay eye" etc. for "AI"
- 📐 **Plural & possessive support** — "AIs", "AI's", "hellos" etc. matched automatically
- 🎙️ Microphone selector with refresh support
- 📝 Transcript display showing what was heard
- ▶️ Start/Stop controls
- 🔄 Reset counter
- 🛡️ Robust error handling with automatic retries (audio read, model loading)
- 🔒 Fully offline — no API keys, no internet, no data leaves your machine

## Requirements

- macOS
- Python 3.13+ (Homebrew recommended)
- Microphone access
- `python-tk@3.13` (for Tkinter GUI), installable via `brew install python-tk@3.13`

## Installation

1. **Clone or download** the project:
   ```zsh
   cd ~/word-counter-app
   ```

2. **Create a virtual environment and install dependencies:**
   ```zsh
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Download the Vosk speech model** (~40 MB):
   ```zsh
   curl -L -o vosk-model.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
   unzip -q vosk-model.zip && rm vosk-model.zip
   ```
   This creates the `vosk-model-small-en-us-0.15/` directory used by the app.

   > **Tip:** The `run_app.sh` launcher script automatically creates the venv, installs missing packages, and downloads the model if needed — so you can skip steps 2–3 and just run `./run_app.sh`.

4. **Grant microphone permissions:**
   - When you first run the app, macOS will ask for microphone permission
   - Go to System Settings > Privacy & Security > Microphone
   - Ensure Terminal (or your Python IDE) has microphone access

## Usage

1. **Run the application:**
   ```zsh
   cd ~/word-counter-app
   ./run_app.sh
   ```

   Or alternatively:
   ```zsh
   cd ~/word-counter-app
   source venv/bin/activate
   python word_counter.py
   ```

2. **Select your microphone** from the dropdown (refresh if needed)

3. **Enter your target word** in the input field (default is "AI")

4. **Click "Start Listening"** to begin speech recognition

   - The app first runs a short **3-second mic calibration** phase.
   - During this time, status shows `Calibrating mic...` and early audio is intentionally ignored.
   - After calibration, status switches to `Listening...` and counting begins.

5. **Speak naturally** — the app will:
   - Display what it heard in the transcript area
   - Count occurrences of your target word in real time
   - Show both partial (live) and final (committed) results

6. **Click "Stop"** to pause listening

7. **Click "Reset Count"** to clear the counter back to zero

## How It Works

The app uses:
- **Vosk** for offline, streaming speech-to-text (via `KaldiRecognizer`)
- **jellyfish** for phonetic matching (Metaphone, Soundex, Jaro-Winkler similarity)
- **PyAudio** for direct microphone audio capture (16 kHz, mono, 250ms chunks)
- **Tkinter** for the graphical user interface

### Dual Recognizer Architecture
The app runs **two Vosk recognizers** simultaneously on the same audio stream:

1. **Grammar-constrained recognizer** — configured with only the phonetic variants of the target word (+ `[unk]`). This biases Vosk's decoder heavily toward the target, dramatically improving detection of short words and abbreviations like "AI".
2. **Unconstrained recognizer** — runs freely for human-readable transcript display.

Each recognizer's output is processed independently: the grammar recognizer drives the count, the unconstrained recognizer drives the transcript.

### Confidence Filtering
The grammar recognizer returns per-word confidence scores. Tokens with confidence below the threshold (default: 0.5) are discarded to prevent false positives — e.g. when background noise or common words like "hey" are force-mapped to target variants.

### Peak Partial Tracking
Vosk emits partial results as speech is being recognized. Sometimes a partial correctly detects a target word, but a later partial revision or the final result loses it. The app tracks the **peak partial match count** per utterance so that once a match is detected, it is never lost to a later revision.

### Audio Pipeline
PyAudio captures 250ms frames → both recognizers process each frame → grammar results drive the counter with confidence filtering → unconstrained results drive the transcript display.

### Startup Calibration
To improve first-run accuracy, the app ignores the first **3.0 seconds** of microphone input after Start is pressed. This allows macOS audio routing, gain control, and noise suppression to stabilize before recognition/counting begins.

### Phonetic Matching
When you click "Start Listening", the app builds a `PhoneticMatcher` for your target word:

- **Abbreviations** (e.g. "AI", "GPU", "ML"): auto-generates all letter-by-letter phonetic variants using a built-in pronunciation table. For "AI" this produces variants including "ay eye", "a i", "ay", "eh eye", etc. Phonetic neighbours are intentionally skipped for abbreviations to avoid false positives.
- **Regular words**: uses Metaphone and Soundex phonetic algorithms to find similar-sounding words.
- **Plurals & possessives**: all variants also match their plural (`AIs`, `ais`) and possessive (`AI's`) forms automatically via an optional regex suffix.
- All variants are compiled into a single regex for fast real-time matching.

## Troubleshooting

### Microphone not working
- Check System Settings > Privacy & Security > Microphone permissions
- Ensure your microphone is working in other applications
- Try selecting a different microphone from the dropdown
- Try restarting the application

### PyAudio installation errors
If `pip install pyaudio` fails, install PortAudio first:
```zsh
brew install portaudio
pip install pyaudio
```

### Vosk model not found
Make sure the `vosk-model-small-en-us-0.15/` directory exists in the project root. Re-download if needed:
```zsh
curl -L -o vosk-model.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip -q vosk-model.zip && rm vosk-model.zip
```

### Poor recognition accuracy
- Wait for calibration to finish (`Calibrating mic...` → `Listening...`) before speaking target words
- Speak clearly and at a moderate pace
- Reduce background noise
- Move your microphone closer
- For better accuracy, consider downloading a larger Vosk model (e.g., `vosk-model-en-us-0.22`, ~1.8 GB)

## Running Tests

```zsh
source venv/bin/activate
python -m unittest test_word_counter
```

All 135 tests should pass.

## Notes

- Fully offline — no internet connection or API keys required
- Word matching is case-insensitive
- **Abbreviations are auto-expanded** into all phonetic variants (no manual configuration needed)
- **Plurals and possessives** are matched automatically (e.g. "AIs", "AI's")
- The counter increments for each occurrence of the word, even if it appears multiple times in one phrase
- The app uses peak-partial tracking and confidence filtering so words are not double-counted or lost during streaming
- Audio stream retries up to 5 consecutive read errors before stopping
- Vosk model loading retries up to 3 times with 1-second delays
- Stream cleanup is thread-safe (guarded by a dedicated lock)

## License

Free to use and modify for personal and commercial purposes.
# buzzwords-counter
# buzzwords-counter
