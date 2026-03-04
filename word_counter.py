#!/usr/bin/env python3
"""
BuzzWords Counter
Real-time speech recognition using Vosk (offline, streaming).
Counts occurrences of a specific word from microphone input.

Uses phonetic matching (jellyfish) to automatically catch
mis-transcriptions of the target word — no API key needed.
For abbreviations like "AI", it auto-generates all plausible
letter-by-letter phonetic variants.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pyaudio
import threading
import re
import json
import os
import time
import logging
from itertools import product

import jellyfish

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Audio settings for Vosk
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 4000  # ~250ms of audio at 16kHz
FORMAT = pyaudio.paInt16

# Minimum per-word confidence (0.0–1.0) from the grammar recognizer
# for a token to be counted.  Raising this reduces false positives
# (e.g. "hey" → "ay") at the cost of occasionally missing quiet speech.
GRAMMAR_CONFIDENCE_THRESHOLD = 0.6

# Maximum number of transcript lines kept in memory
MAX_TRANSCRIPT_LINES = 20

# Ignore initial mic audio to let device gain/noise suppression stabilize
STARTUP_CALIBRATION_SECONDS = 3.0

# Path to Vosk model (relative to this script)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vosk-model-small-en-us-0.15")

# ---------------------------------------------------------------------------
# Phonetic letter table: how each letter sounds when spoken aloud.
# Used to auto-generate all ways a speech model could transcribe an
# abbreviation like AI, ML, GPU, etc.
# ---------------------------------------------------------------------------
LETTER_PHONETICS = {
    'a': ['ay', 'a', 'eh'],
    'b': ['bee', 'be'],
    'c': ['see', 'sea', 'si'],
    'd': ['dee', 'de'],
    'e': ['ee', 'e'],
    'f': ['ef', 'eff'],
    'g': ['gee', 'ji'],
    'h': ['aitch', 'age', 'ach'],
    'i': ['eye', 'i', 'ai'],
    'j': ['jay', 'je'],
    'k': ['kay', 'ca'],
    'l': ['el', 'elle'],
    'm': ['em'],
    'n': ['en'],
    'o': ['oh', 'o'],
    'p': ['pee', 'pe'],
    'q': ['cue', 'kyu'],
    'r': ['are', 'ar'],
    's': ['es', 'ess'],
    't': ['tee', 'te'],
    'u': ['you', 'yu'],
    'v': ['vee', 've'],
    'w': ['double you', 'double u'],
    'x': ['ex', 'eks'],
    'y': ['why', 'wi'],
    'z': ['zee', 'zed'],
}

# Minimum length for a standalone (single-token) variant to avoid
# false positives on common short words like "a" and "i".
MIN_STANDALONE_VARIANT_LEN = 2


# ---------------------------------------------------------------------------
# PhoneticMatcher
# ---------------------------------------------------------------------------

class PhoneticMatcher:
    """
    Automatically generates and matches phonetic variants of a target word.

    For abbreviations (e.g. "AI", "GPU"):
      – Builds all letter-by-letter phonetic combos: "ay eye", "a i", etc.
      – Both space-separated and joined forms are stored.
      – Phonetic neighbours are *skipped* to avoid false positives.

    For regular words:
      – Adds Metaphone / Soundex similar short words.

    In both cases:
      – Builds a single compiled regex for fast runtime matching.
      – Also provides a live phonetic similarity check as a fallback.
    """

    # Small word list used for phonetic scanning (common Vosk outputs)
    _COMMON_SOUNDS = [
        'a', 'i', 'ai', 'ay', 'aye', 'eye', 'hey', 'hi', 'he', 'oh',
        'eh', 'ah', 'uh', 'er', 'are', 'or', 'an', 'in', 'on', 'it',
        'at', 'is', 'as', 'us', 'am', 'be', 'by', 'do', 'go', 'if',
        'me', 'my', 'no', 'so', 'to', 'up', 'we', 'el', 'em', 'en',
        'see', 'sea', 'say', 'may', 'way', 'day', 'pay', 'lay', 'ray',
        'bee', 'dee', 'fee', 'gee', 'jay', 'kay', 'pee', 'tee', 'vee',
        'cue', 'you', 'why', 'ex', 'age', 'aye aye',
    ]

    def __init__(self, target_word):
        self.original = target_word.strip()
        self.target = self.original.lower()
        self.is_abbreviation = self._detect_abbreviation(self.original)
        self.variants = set()
        self.variant_regex = None
        # Compute phonetic codes — guard against empty / non-alpha input
        clean_target = self.target.replace(' ', '')
        if clean_target and clean_target.isalpha():
            try:
                self._target_metaphone = jellyfish.metaphone(clean_target)
                self._target_soundex = jellyfish.soundex(clean_target)
            except Exception:
                self._target_metaphone = ''
                self._target_soundex = ''
        else:
            self._target_metaphone = ''
            self._target_soundex = ''
        self._generate_variants()
        self._filter_and_compile_regex()

    # ---- detection -------------------------------------------------------

    @staticmethod
    def _detect_abbreviation(word):
        """Return True if *word* looks like an abbreviation."""
        clean = word.strip().replace('.', '')
        if not clean:
            return False
        # All uppercase, 1-6 letters (AI, GPU, LLM, ...)
        if clean.isupper() and clean.isalpha() and len(clean) <= 6:
            return True
        # Dot-separated letters: A.I., U.S.A.
        if re.match(r'^[A-Za-z](\.[A-Za-z])+\.?$', word.strip()):
            return True
        return False

    # ---- variant generation -----------------------------------------------

    def _generate_variants(self):
        """Populate self.variants with every plausible transcription."""
        # Always include the target itself
        self.variants.add(self.target)

        if self.is_abbreviation:
            self._generate_abbreviation_variants()
            # Skip phonetic neighbours for abbreviations — the letter-by-
            # letter combos already cover all valid transcriptions.
            # Phonetic neighbours add false positives like "ah", "uh" for
            # short abbreviations.
        else:
            self._generate_phonetic_neighbours()

    def _generate_abbreviation_variants(self):
        """Build letter-by-letter phonetic combos for abbreviations."""
        letters = [c for c in self.target if c.isalpha()]
        if not letters:
            return

        letter_options = []
        for letter in letters:
            options = LETTER_PHONETICS.get(letter, [letter])
            letter_options.append(options)

        for combo in product(*letter_options):
            # Space-separated: "ay eye"
            spaced = ' '.join(combo)
            self.variants.add(spaced)
            # Joined: "ayeye"
            joined = ''.join(combo)
            self.variants.add(joined)

        # Also add each individual letter sound as a standalone variant,
        # but only for short abbreviations (<=2 letters).
        #
        # Rationale: for longer abbreviations like "SAS", standalone
        # sounds such as "es" become overly broad and create many
        # false positives in natural speech.
        if len(letters) <= 2:
            for options in letter_options:
                for opt in options:
                    if len(opt) >= MIN_STANDALONE_VARIANT_LEN:
                        self.variants.add(opt)

        # Common merged forms the model produces
        # e.g. for "AI" Vosk often says just "aye"
        if len(letters) >= 2:
            for combo in product(*letter_options):
                # Also try merging last two parts
                for k in range(len(combo) - 1):
                    merged = list(combo)
                    merged[k] = merged[k] + merged[k + 1]
                    del merged[k + 1]
                    self.variants.add(' '.join(merged))
                    self.variants.add(''.join(merged))

    def _generate_phonetic_neighbours(self):
        """Add words that share Metaphone / Soundex codes with the target."""
        for word in self._COMMON_SOUNDS:
            clean = word.replace(' ', '')
            try:
                if jellyfish.metaphone(clean) == self._target_metaphone:
                    self.variants.add(word)
                elif jellyfish.soundex(clean) == self._target_soundex:
                    self.variants.add(word)
            except Exception:
                continue

    # ---- regex compilation ------------------------------------------------

    def _filter_and_compile_regex(self):
        """Filter out noisy short variants, then build a regex.

        Removes single-character standalone variants (e.g. "a", "i")
        that would cause too many false positives.  Multi-word variants
        containing those characters ("a i") are kept.

        Mutates ``self.variants`` to the filtered set so that
        ``describe_variants()`` and ``is_phonetic_match()`` also
        operate on the cleaned variant list.
        """
        if not self.variants:
            return

        filtered = set()
        for v in self.variants:
            if ' ' in v:
                # Multi-word variant — always keep
                filtered.add(v)
            elif len(v) >= MIN_STANDALONE_VARIANT_LEN:
                filtered.add(v)
            # Single-char variants are dropped (too noisy: "a", "i", etc.)

        if not filtered:
            filtered.add(self.target)

        self.variants = filtered

        # Sort longest-first for greedy matching
        sorted_variants = sorted(filtered, key=len, reverse=True)
        escaped = [re.escape(v) for v in sorted_variants]
        # Allow optional plural / possessive suffixes: 's, s, es
        # e.g. "ai" also matches "ais", "ai's", "aies"
        pattern = r"(?<!\w)(?:" + '|'.join(escaped) + r")(?:'?e?s)?(?!\w)"
        try:
            self.variant_regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            logger.error(f"Regex compilation error: {e}")
            self.variant_regex = None

    # ---- matching ---------------------------------------------------------

    def count_matches(self, text):
        """Count how many times any variant appears in *text*."""
        if not text:
            return 0
        lowered = text.lower()
        if self.variant_regex:
            return len(self.variant_regex.findall(lowered))
        # Fallback: exact word match
        return len(re.findall(r'\b' + re.escape(self.target) + r'\b', lowered))

    def is_phonetic_match(self, word):
        """Check if a single word sounds like the target (live fallback)."""
        if not word:
            return False
        w = word.lower().strip()
        if w in self.variants:
            return True
        try:
            if len(w) >= 2:
                if jellyfish.metaphone(w) == self._target_metaphone:
                    return True
                if jellyfish.soundex(w) == self._target_soundex:
                    return True
                if jellyfish.jaro_winkler_similarity(w, self.target) >= 0.85:
                    return True
        except Exception:
            pass
        return False

    def describe_variants(self):
        """Return a human-readable summary of active variants."""
        return sorted(self.variants)

    def build_vosk_grammar(self):
        """Return a JSON-encoded grammar list for Vosk's KaldiRecognizer.

        Includes every variant (individual words) plus the special
        ``[unk]`` token so Vosk can absorb non-matching speech.
        """
        # Collect every individual word that appears in any variant.
        # Vosk grammar expects single tokens, so multi-word variants
        # are split into their component words.
        words = set()
        for v in self.variants:
            for w in v.split():
                words.add(w.lower())
        # Also add the raw target itself
        words.add(self.target.lower())
        # [unk] is the catch-all for non-matching audio
        word_list = sorted(words) + ["[unk]"]
        return json.dumps(word_list)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class WordCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BuzzWords Counter")
        self.root.geometry("420x460")
        self.root.resizable(True, True)

        # Application state
        self.is_listening = False
        self.target_word = ""
        self.count = 0            # confirmed count (from final results only)
        self._partial_count = 0   # speculative count from current partial
        self._peak_partial_count = 0  # highest partial count in current utterance
        self._lock = threading.Lock()
        self._stream_lock = threading.Lock()  # guards _close_stream
        self._mic_devices = []
        self._pyaudio = None
        self._stream = None
        self._listen_thread = None
        self._vosk_model = None
        self._full_transcript = []
        self._previous_partial = ""
        self._listening_started_at = 0.0
        self._calibration_done = False

        # Phonetic matcher (created when listening starts)
        self._matcher = None

        # Setup UI
        self.setup_ui()

        # Populate microphone list
        self._refresh_microphones()

        # Load Vosk model in background
        self._model_loaded = threading.Event()
        threading.Thread(target=self._load_model, daemon=True).start()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def setup_ui(self):
        """Create the user interface."""
        main_frame = ttk.Frame(self.root, padding="12")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # Title
        ttk.Label(
            main_frame, text="\U0001f41d BuzzWords Counter",
            font=("Helvetica", 16, "bold")
        ).grid(row=0, column=0, pady=(0, 8))

        # Microphone selector
        mic_frame = ttk.LabelFrame(main_frame, text="Microphone", padding="6")
        mic_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 6))
        ttk.Label(mic_frame, text="Device:", font=("Helvetica", 9)).grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        self.mic_var = tk.StringVar()
        self.mic_combo = ttk.Combobox(
            mic_frame, textvariable=self.mic_var,
            state="readonly", width=25, font=("Helvetica", 9)
        )
        self.mic_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 4))
        self.refresh_btn = ttk.Button(
            mic_frame, text="\u21bb Refresh",
            command=self._refresh_microphones, width=8
        )
        self.refresh_btn.grid(row=0, column=2, sticky=tk.E)
        mic_frame.columnconfigure(1, weight=1)

        # Word input
        word_frame = ttk.LabelFrame(main_frame, text="Target Word", padding="6")
        word_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 6))
        ttk.Label(word_frame, text="Word to count:", font=("Helvetica", 9)).grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        self.word_entry = ttk.Entry(word_frame, width=25, font=("Helvetica", 10))
        self.word_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))
        self.word_entry.insert(0, "AI")
        word_frame.columnconfigure(1, weight=1)

        # Counter
        counter_frame = ttk.LabelFrame(main_frame, text="Count", padding="6")
        counter_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 6))
        self.count_label = ttk.Label(
            counter_frame, text="0",
            font=("Helvetica", 28, "bold"), foreground="#2196F3"
        )
        self.count_label.pack()

        # Status
        self.status_label = ttk.Label(
            main_frame, text="Status: Loading model...",
            font=("Helvetica", 9), foreground="#2196F3"
        )
        self.status_label.grid(row=5, column=0, pady=(0, 4))

        # Live transcript
        ttk.Label(main_frame, text="Live transcript:", font=("Helvetica", 8)).grid(
            row=6, column=0, sticky=tk.W
        )
        self.transcript_text = tk.Text(
            main_frame, height=4, width=45, wrap=tk.WORD, font=("Helvetica", 9)
        )
        self.transcript_text.grid(row=7, column=0, pady=(3, 6), sticky=(tk.W, tk.E))
        self.transcript_text.config(state=tk.DISABLED)

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=8, column=0)
        self.start_button = ttk.Button(
            btn_frame, text="\u25b6 Start",
            command=self.start_listening, width=10
        )
        self.start_button.pack(side=tk.LEFT, padx=3)
        self.stop_button = ttk.Button(
            btn_frame, text="\u23f8 Stop",
            command=self.stop_listening, state=tk.DISABLED, width=10
        )
        self.stop_button.pack(side=tk.LEFT, padx=3)
        self.reset_button = ttk.Button(
            btn_frame, text="\U0001f504 Reset",
            command=self.reset_count, width=10
        )
        self.reset_button.pack(side=tk.LEFT, padx=3)

    # --------------------------------------------------------- Model loading
    def _load_model(self):
        """Load Vosk model in background thread with automatic retry."""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                from vosk import Model, SetLogLevel
                SetLogLevel(-1)
                if not os.path.isdir(MODEL_DIR):
                    logger.error(f"Vosk model not found at {MODEL_DIR}")
                    self.update_status(
                        "Error: speech model not found \u2014 see README for download instructions",
                        "#F44336"
                    )
                    return
                self._vosk_model = Model(MODEL_DIR)
                self._model_loaded.set()
                self.update_status("Ready to start", "#666")
                logger.info("Vosk model loaded successfully")
                return
            except ImportError:
                logger.error("Vosk package not installed")
                self.update_status("Error: vosk not installed \u2014 run pip install vosk", "#F44336")
                return
            except Exception as e:
                logger.error(f"Model load attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    self.update_status(f"Model load failed, retrying ({attempt}/{max_retries})...", "#FF9800")
                    time.sleep(1)
                else:
                    self.update_status(f"Model error after {max_retries} attempts: {e}", "#F44336")

    # ------------------------------------------------------- Microphone mgmt
    def _get_input_devices(self):
        """Return list of (pyaudio_index, device_name) for input devices."""
        devices = []
        try:
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                try:
                    info = p.get_device_info_by_index(i)
                    if info.get("maxInputChannels", 0) > 0:
                        devices.append((i, info["name"]))
                except Exception:
                    continue
            p.terminate()
        except Exception as e:
            logger.error(f"Error enumerating audio devices: {e}")
        return devices

    def _refresh_microphones(self):
        """Refresh the microphone dropdown."""
        self._mic_devices = self._get_input_devices()
        names = [name for _, name in self._mic_devices]
        self.mic_combo["values"] = names
        if names:
            self.mic_combo.current(0)
        else:
            self.mic_var.set("-- No microphones found --")
            self.update_status("No microphone detected", "#F44336")
            logger.warning("No input devices found")

    def _get_selected_device_index(self):
        """Return the pyaudio device index for the selected mic, or None."""
        idx = self.mic_combo.current()
        if 0 <= idx < len(self._mic_devices):
            return self._mic_devices[idx][0]
        return None

    # ------------------------------------------------------------ Controls
    def start_listening(self):
        """Start real-time listening."""
        # Guard against double-start (button may not disable fast enough)
        if self.is_listening:
            return

        if not self._model_loaded.is_set():
            messagebox.showinfo("Please Wait", "Speech model is still loading. Try again in a moment.")
            return

        device_index = self._get_selected_device_index()
        if device_index is None:
            messagebox.showerror(
                "Microphone Error",
                "Could not open the selected microphone.\n"
                "Try clicking \u21bb Refresh and selecting a different device."
            )
            return

        raw_word = self.word_entry.get().strip()
        if not raw_word:
            messagebox.showwarning("Input Required", "Please enter a word to count.")
            return

        self.target_word = raw_word.lower()

        # Build phonetic matcher
        self._matcher = PhoneticMatcher(raw_word)
        variants = self._matcher.describe_variants()
        abbr_tag = " [abbreviation]" if self._matcher.is_abbreviation else ""
        logger.info(f"Matching variants{abbr_tag}: {variants}")

        self.is_listening = True
        self._listening_started_at = time.monotonic()
        self._calibration_done = False
        self._previous_partial = ""
        self._full_transcript = []
        with self._lock:
            self._partial_count = 0
            self._peak_partial_count = 0
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.word_entry.config(state=tk.DISABLED)
        self.mic_combo.config(state=tk.DISABLED)
        self.refresh_btn.config(state=tk.DISABLED)

        self.update_status("Calibrating mic...", "#FF9800")

        self._listen_thread = threading.Thread(
            target=self._stream_loop, args=(device_index,), daemon=True
        )
        self._listen_thread.start()

    def stop_listening(self):
        """Stop listening."""
        self.is_listening = False

        def _update_ui():
            try:
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                self.word_entry.config(state=tk.NORMAL)
                self.mic_combo.config(state="readonly")
                self.refresh_btn.config(state=tk.NORMAL)
                self.status_label.config(text="Status: Stopped", foreground="#FF9800")
            except tk.TclError:
                pass
        self.root.after(0, _update_ui)

    def reset_count(self):
        """Reset counter and transcript."""
        with self._lock:
            self.count = 0
            self._partial_count = 0
            self._peak_partial_count = 0
            self._full_transcript = []
            self._previous_partial = ""
        self.update_count()
        self.update_transcript("")

    # ------------------------------------------------ Real-time stream loop
    def _stream_loop(self, device_index):
        """
        Core streaming loop using **dual recognizers**:

        1. A *grammar-constrained* recognizer that only outputs our
           phonetic variants (+ ``[unk]``).  This makes Vosk's decoder
           strongly biased toward the target word, dramatically
           improving detection of abbreviations like "AI".
        2. An *unconstrained* recognizer for human-readable transcript
           display.

        Both receive the same audio frames on every iteration.
        """
        from vosk import KaldiRecognizer

        # --- Unconstrained recognizer (transcript) -----------------------
        rec_transcript = KaldiRecognizer(self._vosk_model, SAMPLE_RATE)
        rec_transcript.SetWords(True)

        # --- Grammar-constrained recognizer (counting) -------------------
        if self._matcher:
            grammar = self._matcher.build_vosk_grammar()
            logger.info(f"Vosk grammar ({len(grammar)} chars): {grammar[:200]}")
            rec_grammar = KaldiRecognizer(self._vosk_model, SAMPLE_RATE, grammar)
            rec_grammar.SetWords(True)
        else:
            rec_grammar = None

        try:
            self._pyaudio = pyaudio.PyAudio()
            self._stream = self._pyaudio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK_SIZE,
            )
            logger.info(f"Audio stream opened (device {device_index}, {SAMPLE_RATE}Hz)")
        except Exception as e:
            logger.error(f"Failed to open audio stream: {e}")
            self.update_status("Mic error - check permissions", "#F44336")
            self.stop_listening()
            return

        audio_error_count = 0
        max_audio_errors = 5  # consecutive read errors before giving up

        try:
            while self.is_listening:
                # --- Read audio frame with retry ---
                data = None
                try:
                    data = self._stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    audio_error_count = 0  # reset on success
                except OSError as e:
                    audio_error_count += 1
                    logger.warning(f"Audio read error ({audio_error_count}/{max_audio_errors}): {e}")
                    if audio_error_count >= max_audio_errors:
                        if self.is_listening:
                            logger.error("Too many consecutive audio errors, stopping")
                            self.update_status("Audio error — mic disconnected?", "#F44336")
                        break
                    continue  # retry next frame

                if data is None:
                    continue

                # Warm-up window: ignore unstable initial audio frames
                if not self._calibration_done:
                    elapsed = time.monotonic() - self._listening_started_at
                    if elapsed < STARTUP_CALIBRATION_SECONDS:
                        continue
                    self._calibration_done = True
                    self.update_status("Listening...", "#4CAF50")

                # --- Feed both recognizers and process results ---
                try:
                    # -- Grammar recognizer (counting) --
                    if rec_grammar is not None:
                        if rec_grammar.AcceptWaveform(data):
                            gres = json.loads(rec_grammar.Result())
                            self._handle_grammar_final(gres)
                        else:
                            gpart = json.loads(rec_grammar.PartialResult())
                            gtext = gpart.get("partial", "")
                            if gtext:
                                self._handle_grammar_partial(gtext)

                    # -- Unconstrained recognizer (transcript) --
                    if rec_transcript.AcceptWaveform(data):
                        result = json.loads(rec_transcript.Result())
                        text = result.get("text", "")
                        if text:
                            self._handle_transcript_final(text)
                    else:
                        partial = json.loads(rec_transcript.PartialResult())
                        partial_text = partial.get("partial", "")
                        if partial_text:
                            self._handle_transcript_partial(partial_text)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping malformed recognizer output: {e}")
                    continue

            # Flush remaining audio
            try:
                if rec_grammar is not None:
                    gfinal = json.loads(rec_grammar.FinalResult())
                    self._handle_grammar_final(gfinal)

                tfinal = json.loads(rec_transcript.FinalResult())
                ttext = tfinal.get("text", "")
                if ttext:
                    self._handle_transcript_final(ttext)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error flushing final result: {e}")

        except Exception as e:
            if self.is_listening:
                logger.error(f"Stream loop error: {e}")
                self.update_status(f"Error: {e}", "#F44336")
        finally:
            self._close_stream()

    def _close_stream(self):
        """Safely close audio stream and PyAudio.

        Thread-safe: may be called from both the stream thread (finally)
        and the main thread (_on_close after join timeout).
        """
        with self._stream_lock:
            try:
                if self._stream:
                    self._stream.stop_stream()
                    self._stream.close()
                    self._stream = None
            except Exception:
                pass
            try:
                if self._pyaudio:
                    self._pyaudio.terminate()
                    self._pyaudio = None
            except Exception:
                pass

    # ------------------------------------------------- Result processing
    # Shared helpers --------------------------------------------------------

    def _commit_utterance(self, final_matches):
        """Commit the count for a completed utterance.

        Takes ``max(final_matches, _peak_partial_count)`` so that if
        the partial caught a match that the final lost (or the
        confidence filter removed), the match is still committed.
        Resets both ``_partial_count`` and ``_peak_partial_count``.
        """
        with self._lock:
            committed = max(final_matches, self._peak_partial_count)
            self.count += committed
            self._partial_count = 0
            self._peak_partial_count = 0
        self.update_count()

    def _update_speculative_count(self, matches):
        """Set the speculative partial count and update the peak high-water mark."""
        with self._lock:
            self._partial_count = matches
            if matches > self._peak_partial_count:
                self._peak_partial_count = matches
        self.update_count()

    def _append_transcript(self, text):
        """Append text to the transcript history, enforcing the max-lines limit.

        Protected by ``_lock`` because ``reset_count`` may clear
        ``_full_transcript`` from the main thread while the stream
        thread is appending.
        """
        with self._lock:
            self._full_transcript.append(text)
            if len(self._full_transcript) > MAX_TRANSCRIPT_LINES:
                self._full_transcript = self._full_transcript[-MAX_TRANSCRIPT_LINES:]

    # Grammar recognizer handlers (counting) --------------------------------

    def _handle_grammar_final(self, result_dict):
        """Count target-word matches from the grammar-constrained recognizer.

        Uses per-word confidence scores to filter out false positives
        (e.g. "hey" forced to "ay" with low confidence).
        ``result_dict`` is the full JSON dict from Vosk, which may
        contain a ``result`` array with word-level confidence.

        Uses ``_peak_partial_count`` (the highest partial-match count
        seen during this utterance) so that a late partial revision to
        [unk] does not discard a match the partial had already caught.
        """
        clean = self._extract_confident_text(result_dict)
        if not clean:
            self._commit_utterance(0)
            return
        logger.info(f"Grammar final: {clean}")
        final_matches = self._count_matches_in(clean)
        self._commit_utterance(final_matches)

    def _handle_grammar_partial(self, text):
        """Speculatively count from grammar partial (shown immediately).

        Partials don't carry confidence scores, so we strip [unk] only.
        Also maintains ``_peak_partial_count`` — the high-water mark for
        partial matches in the current utterance — so that a later
        revision to [unk] cannot lose a match already detected.
        """
        clean = self._strip_unk(text)
        partial_matches = self._count_matches_in(clean) if clean else 0
        self._update_speculative_count(partial_matches)

    @staticmethod
    def _extract_confident_text(result_dict):
        """Return text built only from high-confidence non-[unk] words.

        If the result dict contains a ``result`` array (word-level detail),
        each word is kept only when its confidence >= GRAMMAR_CONFIDENCE_THRESHOLD.
        Falls back to plain ``text`` with [unk] stripping when no
        word-level data is available.
        """
        word_details = result_dict.get("result", [])
        if word_details:
            confident_words = [
                w["word"]
                for w in word_details
                if w.get("word", "") != "[unk]"
                and w.get("conf", 0.0) >= GRAMMAR_CONFIDENCE_THRESHOLD
            ]
            return ' '.join(confident_words).strip()
        # Fallback: no word-level data (shouldn't happen with SetWords(True))
        return WordCounterApp._strip_unk(result_dict.get("text", ""))

    @staticmethod
    def _strip_unk(text):
        """Remove ``[unk]`` tokens from Vosk grammar output."""
        return ' '.join(w for w in text.split() if w != '[unk]').strip()

    # Transcript recognizer handlers ----------------------------------------

    def _handle_transcript_final(self, text):
        """Append finalised text to the readable transcript."""
        self._append_transcript(text)
        with self._lock:
            self._previous_partial = ""
        self._update_transcript_display()

    def _handle_transcript_partial(self, text):
        """Update the live partial line in the transcript."""
        with self._lock:
            changed = text != self._previous_partial
            if changed:
                self._previous_partial = text
        if changed:
            self._update_transcript_display()

    # Legacy handlers — used only by the ``process_speech()`` simplified
    # public API, NOT by the dual-recognizer stream loop.  Kept for
    # backward compatibility and testing of the partial→final anti-
    # double-count logic in isolation.

    def _handle_final_result(self, text):
        """Commit a finalized phrase (legacy path, not used by stream loop)."""
        logger.info(f"Final: {text}")
        final_matches = self._count_matches_in(text)
        self._commit_utterance(final_matches)
        self._append_transcript(text)
        with self._lock:
            self._previous_partial = ""
        self._update_transcript_display()

    def _handle_partial_result(self, text):
        """Process a partial phrase (legacy path, not used by stream loop)."""
        with self._lock:
            changed = text != self._previous_partial
            if changed:
                self._previous_partial = text
        if changed:
            partial_matches = self._count_matches_in(text)
            self._update_speculative_count(partial_matches)
            self._update_transcript_display()

    def _count_matches_in(self, text):
        """Return the number of target-word matches in *text* (pure helper, no side effects)."""
        if not text:
            return 0
        if self._matcher:
            return self._matcher.count_matches(text)
        # Fallback: exact word match
        words = re.findall(r'\b\w+\b', text.lower())
        return words.count(self.target_word)

    def _count_word(self, text):
        """Count occurrences of target word and add to confirmed count."""
        if not text:
            return
        occurrences = self._count_matches_in(text)
        if occurrences > 0:
            with self._lock:
                self.count += occurrences
            self.update_count()

    def process_speech(self, text):
        """Process recognized speech and count target word (public API)."""
        if not text or not isinstance(text, str):
            return
        self._count_word(text)
        self.update_transcript(text)

    # ---------------------------------------------------------- UI updates
    def _update_transcript_display(self):
        """Update transcript with full history + current partial."""
        with self._lock:
            lines = list(self._full_transcript)
            partial = self._previous_partial
        if partial:
            lines.append(f"... {partial}")
        self.update_transcript("\n".join(lines))

    def update_count(self):
        """Update the count display (thread-safe).

        Shows confirmed count + speculative partial count.
        """
        with self._lock:
            display = self.count + self._partial_count
        try:
            self.root.after(0, lambda: self.count_label.config(text=str(display)))
        except tk.TclError:
            pass

    def update_status(self, message, color="#666"):
        """Update status message (thread-safe)."""
        try:
            self.root.after(0, lambda: self.status_label.config(
                text=f"Status: {message}", foreground=color
            ))
        except tk.TclError:
            pass

    def update_transcript(self, text):
        """Update the transcript display (thread-safe)."""
        def _update():
            try:
                self.transcript_text.config(state=tk.NORMAL)
                self.transcript_text.delete(1.0, tk.END)
                self.transcript_text.insert(1.0, text)
                self.transcript_text.see(tk.END)
                self.transcript_text.config(state=tk.DISABLED)
            except tk.TclError:
                pass
        self.root.after(0, _update)

    def _on_close(self):
        """Handle window close gracefully."""
        self.is_listening = False
        # Wait for listen thread to finish (with timeout to avoid hanging)
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=2.0)
        self._close_stream()
        self.root.destroy()


def main():
    """Main application entry point."""
    root = tk.Tk()
    app = WordCounterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
