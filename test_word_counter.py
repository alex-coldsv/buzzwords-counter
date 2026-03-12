#!/usr/bin/env python3
"""
Tests for Word Counter Application (Vosk + phonetic matching)
"""

import unittest
from unittest.mock import patch, MagicMock
import tkinter as tk
import threading
import re
import json

from word_counter import (
    WordCounterApp, PhoneticMatcher, LETTER_PHONETICS,
    MAX_TRANSCRIPT_LINES,
    STARTUP_CALIBRATION_SECONDS,
)


def _make_app(root):
    """Create a WordCounterApp with mocked pyaudio and deferred model loading."""
    with patch('word_counter.pyaudio') as mock_pa:
        mock_instance = MagicMock()
        mock_instance.get_device_count.return_value = 1
        mock_instance.get_device_info_by_index.return_value = {
            'name': 'Test Mic', 'maxInputChannels': 1
        }
        mock_pa.PyAudio.return_value = mock_instance
        app = WordCounterApp(root)
    return app


# =========================================================================
# PhoneticMatcher tests
# =========================================================================

class TestPhoneticMatcherAbbreviation(unittest.TestCase):
    """Test PhoneticMatcher with abbreviations like AI, GPU, ML."""

    def test_ai_is_detected_as_abbreviation(self):
        m = PhoneticMatcher("AI")
        self.assertTrue(m.is_abbreviation)

    def test_gpu_is_detected_as_abbreviation(self):
        m = PhoneticMatcher("GPU")
        self.assertTrue(m.is_abbreviation)

    def test_hello_is_not_abbreviation(self):
        m = PhoneticMatcher("hello")
        self.assertFalse(m.is_abbreviation)

    def test_dotted_abbreviation(self):
        m = PhoneticMatcher("A.I.")
        self.assertTrue(m.is_abbreviation)

    def test_ai_variants_include_target(self):
        m = PhoneticMatcher("AI")
        self.assertIn("ai", m.variants)

    def test_ai_variants_include_ay_eye(self):
        m = PhoneticMatcher("AI")
        self.assertIn("ay eye", m.variants)

    def test_ai_variants_include_a_i(self):
        m = PhoneticMatcher("AI")
        # "a i" has "a" which is 1 char — it appears as a multi-word variant
        self.assertIn("a i", m.variants)

    def test_ai_variants_include_ay(self):
        m = PhoneticMatcher("AI")
        self.assertIn("ay", m.variants)

    def test_ai_variants_exclude_phonetic_neighbours(self):
        m = PhoneticMatcher("AI")
        # Abbreviations skip phonetic neighbours to avoid false positives
        # like "ah", "aye" which are too generic
        self.assertNotIn("ah", m.variants)
        self.assertNotIn("aye", m.variants)

    def test_ai_matches_ay_in_text(self):
        m = PhoneticMatcher("AI")
        self.assertEqual(m.count_matches("I think ay is important"), 1)

    def test_ai_matches_a_i_in_text(self):
        m = PhoneticMatcher("AI")
        self.assertEqual(m.count_matches("talking about a i today"), 1)

    def test_ai_matches_ay_eye_in_text(self):
        m = PhoneticMatcher("AI")
        self.assertEqual(m.count_matches("we discussed ay eye models"), 1)

    def test_ai_matches_exact_in_text(self):
        m = PhoneticMatcher("AI")
        self.assertEqual(m.count_matches("ai is transforming the world"), 1)

    def test_ai_multiple_matches(self):
        m = PhoneticMatcher("AI")
        self.assertEqual(m.count_matches("ai is great and ay is everywhere"), 2)

    def test_ai_no_false_positive_on_aim(self):
        m = PhoneticMatcher("AI")
        # "aim" should not match because the regex uses word boundaries
        self.assertEqual(m.count_matches("my aim is good"), 0)

    def test_gpu_variants_include_expected(self):
        m = PhoneticMatcher("GPU")
        self.assertIn("gpu", m.variants)
        # "gee pee you" — letter-by-letter
        self.assertIn("gee pee you", m.variants)

    def test_ml_variants(self):
        m = PhoneticMatcher("ML")
        self.assertIn("ml", m.variants)
        self.assertIn("em el", m.variants)

    def test_sas_excludes_standalone_letter_sounds(self):
        m = PhoneticMatcher("SAS")
        self.assertNotIn("es", m.variants)
        self.assertNotIn("ess", m.variants)
        self.assertNotIn("ay", m.variants)

    def test_sas_still_matches_exact_word(self):
        m = PhoneticMatcher("SAS")
        self.assertEqual(m.count_matches("sas is widely used"), 1)

    def test_sas_reduces_false_positive_on_common_words(self):
        m = PhoneticMatcher("SAS")
        self.assertEqual(m.count_matches("this is as simple as it gets"), 0)

    def test_sas_grammar_excludes_single_char_a_token(self):
        m = PhoneticMatcher("SAS")
        grammar_words = json.loads(m.build_vosk_grammar())
        self.assertNotIn("a", grammar_words)

    def test_sas_variants_do_not_include_joined_noise_forms(self):
        m = PhoneticMatcher("SAS")
        self.assertNotIn("esayess", m.variants)


class TestPluralAndPossessive(unittest.TestCase):
    """Test that plural and possessive forms are matched."""

    def test_ai_plural_ais(self):
        m = PhoneticMatcher("AI")
        self.assertEqual(m.count_matches("multiple ais are available"), 1)

    def test_ai_possessive(self):
        m = PhoneticMatcher("AI")
        self.assertEqual(m.count_matches("ai's capabilities are growing"), 1)

    def test_ai_plural_uppercase(self):
        m = PhoneticMatcher("AI")
        self.assertEqual(m.count_matches("AIs are everywhere"), 1)

    def test_hello_plural(self):
        m = PhoneticMatcher("hello")
        self.assertEqual(m.count_matches("many hellos were exchanged"), 1)

    def test_hello_possessive(self):
        m = PhoneticMatcher("hello")
        self.assertEqual(m.count_matches("hello's origin is interesting"), 1)

    def test_gpu_plural(self):
        m = PhoneticMatcher("GPU")
        self.assertEqual(m.count_matches("we bought new gpus"), 1)

    def test_plural_and_base_both_count(self):
        m = PhoneticMatcher("AI")
        self.assertEqual(m.count_matches("ai and ais are both here"), 2)

    def test_variant_plural_ay_s(self):
        m = PhoneticMatcher("AI")
        self.assertEqual(m.count_matches("the ays of the world"), 1)

    def test_no_false_positive_on_aims(self):
        m = PhoneticMatcher("AI")
        self.assertEqual(m.count_matches("she aims high"), 0)


class TestPhoneticMatcherRegularWord(unittest.TestCase):
    """Test PhoneticMatcher with regular (non-abbreviation) words."""

    def test_hello_exact_match(self):
        m = PhoneticMatcher("hello")
        self.assertEqual(m.count_matches("hello world"), 1)

    def test_hello_no_match(self):
        m = PhoneticMatcher("hello")
        self.assertEqual(m.count_matches("goodbye world"), 0)

    def test_hello_case_insensitive(self):
        m = PhoneticMatcher("hello")
        self.assertEqual(m.count_matches("Hello HELLO hello"), 3)

    def test_hello_not_abbreviation(self):
        m = PhoneticMatcher("hello")
        self.assertFalse(m.is_abbreviation)

    def test_empty_text(self):
        m = PhoneticMatcher("test")
        self.assertEqual(m.count_matches(""), 0)

    def test_none_text(self):
        m = PhoneticMatcher("test")
        self.assertEqual(m.count_matches(None), 0)


class TestPhoneticMatcherIsMatch(unittest.TestCase):
    """Test the is_phonetic_match method for live fallback."""

    def test_exact_match(self):
        m = PhoneticMatcher("AI")
        self.assertTrue(m.is_phonetic_match("ai"))

    def test_variant_match(self):
        m = PhoneticMatcher("AI")
        self.assertTrue(m.is_phonetic_match("ay"))

    def test_no_match(self):
        m = PhoneticMatcher("AI")
        self.assertFalse(m.is_phonetic_match("banana"))

    def test_empty_string(self):
        m = PhoneticMatcher("AI")
        self.assertFalse(m.is_phonetic_match(""))

    def test_none(self):
        m = PhoneticMatcher("AI")
        self.assertFalse(m.is_phonetic_match(None))


class TestDetectAbbreviation(unittest.TestCase):
    """Test abbreviation detection separately."""

    def test_uppercase_short(self):
        self.assertTrue(PhoneticMatcher._detect_abbreviation("AI"))

    def test_uppercase_three(self):
        self.assertTrue(PhoneticMatcher._detect_abbreviation("GPU"))

    def test_lowercase_short(self):
        self.assertFalse(PhoneticMatcher._detect_abbreviation("ai"))

    def test_long_uppercase(self):
        # 7+ chars: not treated as abbreviation
        self.assertFalse(PhoneticMatcher._detect_abbreviation("VERYLONGWORD"))

    def test_mixed_case(self):
        self.assertFalse(PhoneticMatcher._detect_abbreviation("Hello"))

    def test_dotted(self):
        self.assertTrue(PhoneticMatcher._detect_abbreviation("A.I."))

    def test_empty(self):
        self.assertFalse(PhoneticMatcher._detect_abbreviation(""))

    def test_numbers(self):
        self.assertFalse(PhoneticMatcher._detect_abbreviation("123"))


class TestVariantRegex(unittest.TestCase):
    """Test that the compiled regex works correctly."""

    def test_regex_is_compiled(self):
        m = PhoneticMatcher("AI")
        self.assertIsNotNone(m.variant_regex)

    def test_regex_word_boundary(self):
        m = PhoneticMatcher("AI")
        # "aisle" should not match "ai" because of word boundary
        self.assertEqual(m.count_matches("walking down the aisle"), 0)

    def test_regex_multiple_variants_in_one_text(self):
        m = PhoneticMatcher("AI")
        # "ai" and "ay" both in text
        count = m.count_matches("ai and ay are both here")
        self.assertEqual(count, 2)

    def test_regex_case_insensitive(self):
        m = PhoneticMatcher("AI")
        self.assertEqual(m.count_matches("AI is great"), 1)


# =========================================================================
# WordCounterApp UI tests
# =========================================================================

class TestWordCounterUI(unittest.TestCase):
    """Tests for UI initialization."""

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = _make_app(self.root)

    def tearDown(self):
        self.app.is_listening = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def test_initial_state(self):
        self.assertFalse(self.app.is_listening)
        self.assertEqual(self.app.target_word, "")
        self.assertEqual(self.app.count, 0)

    def test_window_title(self):
        self.assertEqual(self.root.title(), "BuzzWords Counter")

    def test_default_word_entry(self):
        self.assertEqual(self.app.word_entry.get(), "AI")

    def test_count_label_initial_value(self):
        self.assertEqual(self.app.count_label.cget("text"), "0")

    def test_stop_button_initially_disabled(self):
        self.assertEqual(str(self.app.stop_button.cget("state")), "disabled")

    def test_mic_combo_populated(self):
        values = self.app.mic_combo['values']
        self.assertTrue(len(values) > 0)
        self.assertEqual(values[0], 'Test Mic')

    def test_matcher_initially_none(self):
        self.assertIsNone(self.app._matcher)


# =========================================================================
# Speech processing tests
# =========================================================================

class TestProcessSpeech(unittest.TestCase):
    """Tests for process_speech with phonetic matching active."""

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = _make_app(self.root)
        self.app.target_word = "ai"
        self.app._matcher = PhoneticMatcher("AI")

    def tearDown(self):
        self.app.is_listening = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def test_ai_exact(self):
        self.app.process_speech("we use ai every day")
        self.assertEqual(self.app.count, 1)

    def test_ai_as_ay(self):
        self.app.process_speech("we use ay every day")
        self.assertEqual(self.app.count, 1)

    def test_ai_as_a_i(self):
        self.app.process_speech("talking about a i today")
        self.assertEqual(self.app.count, 1)

    def test_ai_multiple(self):
        self.app.process_speech("ai is great and ai will grow")
        self.assertEqual(self.app.count, 2)

    def test_ai_mixed_variants(self):
        self.app.process_speech("ai and ay are the same")
        self.assertEqual(self.app.count, 2)

    def test_no_match(self):
        self.app.process_speech("the weather is nice today")
        self.assertEqual(self.app.count, 0)

    def test_empty_text(self):
        self.app.process_speech("")
        self.assertEqual(self.app.count, 0)

    def test_none_text(self):
        self.app.process_speech(None)
        self.assertEqual(self.app.count, 0)

    def test_non_string(self):
        self.app.process_speech(12345)
        self.assertEqual(self.app.count, 0)

    def test_accumulates(self):
        self.app.process_speech("ai here")
        self.app.process_speech("ai there")
        self.assertEqual(self.app.count, 2)

    def test_case_insensitive(self):
        self.app.process_speech("AI Ai ai")
        self.assertEqual(self.app.count, 3)

    def test_no_partial_word_match(self):
        self.app.process_speech("aim ail air aide")
        self.assertEqual(self.app.count, 0)


class TestProcessSpeechRegularWord(unittest.TestCase):
    """Test process_speech with a regular (non-abbreviation) word."""

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = _make_app(self.root)
        self.app.target_word = "hello"
        self.app._matcher = PhoneticMatcher("hello")

    def tearDown(self):
        self.app.is_listening = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def test_exact_match(self):
        self.app.process_speech("hello world")
        self.assertEqual(self.app.count, 1)

    def test_multiple(self):
        self.app.process_speech("hello there hello friend hello")
        self.assertEqual(self.app.count, 3)

    def test_no_match(self):
        self.app.process_speech("goodbye world")
        self.assertEqual(self.app.count, 0)

    def test_very_long_text(self):
        text = " ".join(["hello"] * 500)
        self.app.process_speech(text)
        self.assertEqual(self.app.count, 500)


# =========================================================================
# _count_word tests
# =========================================================================

class TestCountWord(unittest.TestCase):
    """Tests for the _count_word internal method."""

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = _make_app(self.root)

    def tearDown(self):
        self.app.is_listening = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def test_count_with_matcher(self):
        self.app.target_word = "ai"
        self.app._matcher = PhoneticMatcher("AI")
        self.app._count_word("ai is here")
        self.assertEqual(self.app.count, 1)

    def test_count_without_matcher_fallback(self):
        self.app.target_word = "test"
        self.app._matcher = None
        self.app._count_word("this is a test")
        self.assertEqual(self.app.count, 1)

    def test_count_empty(self):
        self.app.target_word = "test"
        self.app._matcher = PhoneticMatcher("test")
        self.app._count_word("")
        self.assertEqual(self.app.count, 0)

    def test_count_none(self):
        self.app.target_word = "test"
        self.app._count_word(None)
        self.assertEqual(self.app.count, 0)

    def test_count_thread_safe(self):
        self.app.target_word = "yes"
        self.app._matcher = PhoneticMatcher("yes")
        threads = []
        for _ in range(100):
            t = threading.Thread(target=self.app._count_word, args=("yes",))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(self.app.count, 100)



# =========================================================================
# Result handler tests (legacy path)
# =========================================================================

class TestHandleResults(unittest.TestCase):
    """Tests for final and partial result handlers."""

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = _make_app(self.root)
        self.app.target_word = "ai"
        self.app._matcher = PhoneticMatcher("AI")

    def tearDown(self):
        self.app.is_listening = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def test_handle_final_result_counts(self):
        self.app._handle_final_result("we use ai every day")
        self.assertEqual(self.app.count, 1)

    def test_handle_final_result_appends_transcript(self):
        self.app._handle_final_result("first sentence")
        self.app._handle_final_result("second sentence")
        self.assertEqual(len(self.app._full_transcript), 2)

    def test_handle_final_keeps_max_lines(self):
        for i in range(MAX_TRANSCRIPT_LINES + 5):
            self.app._handle_final_result(f"line {i}")
        self.assertLessEqual(len(self.app._full_transcript), MAX_TRANSCRIPT_LINES)

    def test_handle_partial_counts_new_words_only(self):
        self.app._previous_partial = ""
        self.app._handle_partial_result("talking about ai")
        # Partial matches go to _partial_count, not confirmed count
        self.assertEqual(self.app._partial_count, 1)
        self.assertEqual(self.app.count, 0)
        # Extend partial — "ai" still appears once, so _partial_count stays 1
        self.app._handle_partial_result("talking about ai and its future")
        self.assertEqual(self.app._partial_count, 1)
        self.assertEqual(self.app.count, 0)

    def test_handle_partial_counts_new_target_word(self):
        self.app._previous_partial = "talking about"
        self.app._handle_partial_result("talking about ai")
        self.assertEqual(self.app._partial_count, 1)
        self.assertEqual(self.app.count, 0)

    def test_handle_final_clears_partial(self):
        self.app._previous_partial = "some partial text"
        self.app._handle_final_result("final text")
        self.assertEqual(self.app._previous_partial, "")
        # Partial count should be reset when final arrives
        self.assertEqual(self.app._partial_count, 0)

    def test_handle_final_catches_variant_ay(self):
        self.app._handle_final_result("we use ay every day")
        self.assertEqual(self.app.count, 1)

    def test_partial_then_final_no_double_count(self):
        """Core anti-double-count test: partial sees 'ai', then final
        confirms 'ai' — total confirmed count must be 1, not 2."""
        self.app._handle_partial_result("talking about ai")
        self.assertEqual(self.app._partial_count, 1)
        self.assertEqual(self.app.count, 0)
        # Final arrives for the same utterance
        self.app._handle_final_result("talking about ai")
        self.assertEqual(self.app.count, 1)
        self.assertEqual(self.app._partial_count, 0)

    def test_partial_caught_but_final_dropped(self):
        """If partial caught a match but Vosk's final drops it, keep the
        partial's count so we don't lose correctly-heard words.
        Uses a regular word (not abbreviation) — peak partial fallback
        is only active for non-abbreviation targets."""
        self.app.target_word = "hello"
        self.app._matcher = PhoneticMatcher("hello")
        self.app._handle_partial_result("talking about hello")
        self.assertEqual(self.app._partial_count, 1)
        # Vosk post-processing revises the text and drops 'hello'
        self.app._handle_final_result("talking about")
        # Should still commit 1 (from peak partial) not 0
        self.assertEqual(self.app.count, 1)
        self.assertEqual(self.app._partial_count, 0)

    def test_multiple_partials_then_final(self):
        """Successive partials should not accumulate into confirmed count."""
        self.app._handle_partial_result("the ai")
        self.assertEqual(self.app._partial_count, 1)
        self.app._handle_partial_result("the ai is great")
        self.assertEqual(self.app._partial_count, 1)
        self.app._handle_partial_result("the ai is great ai")
        self.assertEqual(self.app._partial_count, 2)
        # Final confirms both
        self.app._handle_final_result("the ai is great ai")
        self.assertEqual(self.app.count, 2)
        self.assertEqual(self.app._partial_count, 0)

    def test_final_sees_more_than_partial(self):
        """If final has more matches than partial, use the final count."""
        self.app._handle_partial_result("the ay")
        self.assertEqual(self.app._partial_count, 1)
        # Final resolves to two matches
        self.app._handle_final_result("the ai and ai")
        self.assertEqual(self.app.count, 2)
        self.assertEqual(self.app._partial_count, 0)


# =========================================================================
# Grammar recognizer handler tests
# =========================================================================

class TestGrammarHandlers(unittest.TestCase):
    """Tests for the grammar-constrained recognizer handlers."""

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = _make_app(self.root)
        self.app.target_word = "ai"
        self.app._matcher = PhoneticMatcher("AI")

    def tearDown(self):
        self.app.is_listening = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _make_result(self, words_with_conf):
        """Helper: build a Vosk-style result dict.

        *words_with_conf* is a list of (word, confidence) tuples.
        """
        return {
            "text": " ".join(w for w, _ in words_with_conf),
            "result": [
                {"word": w, "conf": c, "start": 0.0, "end": 0.1}
                for w, c in words_with_conf
            ],
        }

    def test_grammar_final_counts_high_confidence(self):
        res = self._make_result([("ay", 0.9), ("[unk]", 1.0), ("ay", 0.85)])
        self.app._handle_grammar_final(res)
        self.assertEqual(self.app.count, 2)

    def test_grammar_final_filters_low_confidence(self):
        """A forced mapping (e.g. 'hey' → 'ay') with low confidence
        should NOT be counted."""
        res = self._make_result([("ay", 0.3), ("[unk]", 1.0)])
        self.app._handle_grammar_final(res)
        self.assertEqual(self.app.count, 0)

    def test_grammar_final_mixed_confidence(self):
        """Only high-confidence tokens count."""
        res = self._make_result([("ay", 0.9), ("[unk]", 1.0), ("ay", 0.4)])
        self.app._handle_grammar_final(res)
        self.assertEqual(self.app.count, 1)

    def test_grammar_final_only_unk(self):
        """All [unk] with abbreviation: peak partial is NOT committed
        because grammar partials lack confidence scores."""
        self.app._partial_count = 1
        self.app._peak_partial_count = 1
        res = self._make_result([("[unk]", 1.0), ("[unk]", 1.0)])
        self.app._handle_grammar_final(res)
        self.assertEqual(self.app.count, 0)  # abbreviation — no peak fallback
        self.assertEqual(self.app._partial_count, 0)
        self.assertEqual(self.app._peak_partial_count, 0)

    def test_grammar_final_empty_result(self):
        res = {"text": "", "result": []}
        self.app._handle_grammar_final(res)
        self.assertEqual(self.app.count, 0)

    def test_grammar_final_fallback_no_result_array(self):
        """If no 'result' array, falls back to stripping [unk] from text."""
        res = {"text": "ay [unk] ay"}
        self.app._handle_grammar_final(res)
        self.assertEqual(self.app.count, 2)

    def test_grammar_partial_sets_speculative_count(self):
        self.app._handle_grammar_partial("ay [unk]")
        self.assertEqual(self.app._partial_count, 1)
        self.assertEqual(self.app.count, 0)

    def test_grammar_partial_ignores_pure_unk(self):
        self.app._handle_grammar_partial("[unk] [unk]")
        self.assertEqual(self.app._partial_count, 0)

    def test_grammar_partial_revises_to_unk_no_peak_commit_for_abbreviation(self):
        """For abbreviations, if grammar partial initially hears a variant
        then revises to all [unk], _peak_partial_count is tracked but NOT
        committed — confidence-filtered finals are the only counting path."""
        self.app._handle_grammar_partial("ay [unk]")
        self.assertEqual(self.app._partial_count, 1)
        self.assertEqual(self.app._peak_partial_count, 1)
        # Recognizer revises — now pure [unk]
        self.app._handle_grammar_partial("[unk] [unk]")
        self.assertEqual(self.app._partial_count, 0)
        self.assertEqual(self.app._peak_partial_count, 1)  # peak preserved!
        # Final confirms nothing — peak is NOT committed for abbreviations
        res = self._make_result([("[unk]", 1.0), ("[unk]", 1.0)])
        self.app._handle_grammar_final(res)
        self.assertEqual(self.app.count, 0)  # no peak fallback
        self.assertEqual(self.app._peak_partial_count, 0)

    def test_grammar_partial_then_final_no_double(self):
        self.app._handle_grammar_partial("ay [unk]")
        self.assertEqual(self.app._partial_count, 1)
        res = self._make_result([("ay", 0.95), ("[unk]", 1.0)])
        self.app._handle_grammar_final(res)
        self.assertEqual(self.app.count, 1)
        self.assertEqual(self.app._partial_count, 0)
        self.assertEqual(self.app._peak_partial_count, 0)

    def test_grammar_partial_updates_peak(self):
        """_peak_partial_count must track the highest partial seen."""
        self.app._handle_grammar_partial("ay [unk]")
        self.assertEqual(self.app._peak_partial_count, 1)
        # Second partial adds another match
        self.app._handle_grammar_partial("ay [unk] ay")
        self.assertEqual(self.app._peak_partial_count, 2)
        # Partial loses one — peak stays at 2
        self.app._handle_grammar_partial("ay [unk]")
        self.assertEqual(self.app._partial_count, 1)
        self.assertEqual(self.app._peak_partial_count, 2)

    def test_grammar_peak_across_two_utterances(self):
        """Peak must reset after final, not carry into next utterance."""
        self.app._handle_grammar_partial("ay [unk]")
        self.assertEqual(self.app._peak_partial_count, 1)
        res = self._make_result([("ay", 0.9)])
        self.app._handle_grammar_final(res)
        self.assertEqual(self.app.count, 1)
        self.assertEqual(self.app._peak_partial_count, 0)
        # New utterance — peak starts fresh
        self.app._handle_grammar_partial("ay [unk]")
        self.assertEqual(self.app._peak_partial_count, 1)

    def test_long_abbreviation_partial_peak_not_committed(self):
        """For all abbreviations, speculative partial peaks should not
        be committed if final has no confirmed match."""
        self.app.target_word = "sas"
        self.app._matcher = PhoneticMatcher("SAS")
        self.app._handle_grammar_partial("es ay es")
        self.assertEqual(self.app._partial_count, 1)
        self.assertEqual(self.app._peak_partial_count, 1)
        res = self._make_result([("[unk]", 1.0), ("[unk]", 1.0)])
        self.app._handle_grammar_final(res)
        self.assertEqual(self.app.count, 0)
        self.assertEqual(self.app._partial_count, 0)
        self.assertEqual(self.app._peak_partial_count, 0)

    def test_long_abbreviation_final_still_counts(self):
        """Long abbreviations should still count when final is confirmed."""
        self.app.target_word = "sas"
        self.app._matcher = PhoneticMatcher("SAS")
        res = self._make_result([("sas", 0.95)])
        self.app._handle_grammar_final(res)
        self.assertEqual(self.app.count, 1)

    def test_strip_unk(self):
        self.assertEqual(self.app._strip_unk("[unk] ay [unk] eye [unk]"), "ay eye")
        self.assertEqual(self.app._strip_unk("[unk]"), "")
        self.assertEqual(self.app._strip_unk("ay"), "ay")

    def test_extract_confident_text(self):
        """_extract_confident_text filters by confidence threshold."""
        res = {
            "text": "ay eye",
            "result": [
                {"word": "ay", "conf": 0.9, "start": 0.0, "end": 0.1},
                {"word": "eye", "conf": 0.3, "start": 0.1, "end": 0.2},
            ],
        }
        self.assertEqual(self.app._extract_confident_text(res), "ay")

    def test_extract_confident_text_all_below(self):
        res = {
            "text": "ay",
            "result": [{"word": "ay", "conf": 0.2, "start": 0.0, "end": 0.1}],
        }
        self.assertEqual(self.app._extract_confident_text(res), "")

    def test_extract_confident_text_fallback(self):
        """Without a result array, falls back to text with [unk] stripped."""
        res = {"text": "[unk] ay [unk]"}
        self.assertEqual(self.app._extract_confident_text(res), "ay")

    def test_transcript_final_appends(self):
        self.app._handle_transcript_final("hello world")
        self.assertEqual(self.app._full_transcript, ["hello world"])

    def test_transcript_partial_updates(self):
        self.app._handle_transcript_partial("hello")
        self.assertEqual(self.app._previous_partial, "hello")
        self.app._handle_transcript_partial("hello world")
        self.assertEqual(self.app._previous_partial, "hello world")

    def test_transcript_final_with_count_true(self):
        """When count=True, transcript final should also commit matches."""
        self.app.target_word = "ai"
        self.app._matcher = PhoneticMatcher("AI")
        self.app._handle_transcript_final("we discussed ai today", count=True)
        self.assertEqual(self.app.count, 1)
        self.assertEqual(self.app._full_transcript, ["we discussed ai today"])

    def test_transcript_final_with_count_false_no_counting(self):
        """Default count=False should not affect the counter."""
        self.app.target_word = "ai"
        self.app._matcher = PhoneticMatcher("AI")
        self.app._handle_transcript_final("we discussed ai today")
        self.assertEqual(self.app.count, 0)

    def test_transcript_partial_with_count_true(self):
        """When count=True, transcript partial should update speculative count."""
        self.app.target_word = "ai"
        self.app._matcher = PhoneticMatcher("AI")
        self.app._handle_transcript_partial("talking about ai", count=True)
        self.assertEqual(self.app._partial_count, 1)
        self.assertEqual(self.app.count, 0)

    def test_transcript_partial_with_count_false_no_counting(self):
        """Default count=False should not update speculative count."""
        self.app.target_word = "ai"
        self.app._matcher = PhoneticMatcher("AI")
        self.app._handle_transcript_partial("talking about ai")
        self.assertEqual(self.app._partial_count, 0)


# =========================================================================
# _use_grammar_recognizer tests
# =========================================================================

class TestUseGrammarRecognizer(unittest.TestCase):
    """Test the abbreviation-only grammar recognizer gate."""

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = _make_app(self.root)

    def tearDown(self):
        self.app.is_listening = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def test_abbreviation_uses_grammar(self):
        self.app._matcher = PhoneticMatcher("AI")
        self.assertTrue(self.app._use_grammar_recognizer())

    def test_long_abbreviation_uses_grammar(self):
        self.app._matcher = PhoneticMatcher("SAS")
        self.assertTrue(self.app._use_grammar_recognizer())

    def test_regular_word_skips_grammar(self):
        self.app._matcher = PhoneticMatcher("long")
        self.assertFalse(self.app._use_grammar_recognizer())

    def test_regular_word_hello_skips_grammar(self):
        self.app._matcher = PhoneticMatcher("hello")
        self.assertFalse(self.app._use_grammar_recognizer())

    def test_no_matcher_skips_grammar(self):
        self.app._matcher = None
        self.assertFalse(self.app._use_grammar_recognizer())


# =========================================================================
# build_vosk_grammar tests
# =========================================================================

class TestBuildVoskGrammar(unittest.TestCase):
    """Tests for PhoneticMatcher.build_vosk_grammar()."""

    def test_grammar_is_valid_json(self):
        m = PhoneticMatcher("AI")
        gram = m.build_vosk_grammar()
        parsed = json.loads(gram)
        self.assertIsInstance(parsed, list)

    def test_grammar_contains_unk(self):
        m = PhoneticMatcher("AI")
        parsed = json.loads(m.build_vosk_grammar())
        self.assertIn("[unk]", parsed)

    def test_grammar_contains_target(self):
        m = PhoneticMatcher("AI")
        parsed = json.loads(m.build_vosk_grammar())
        self.assertIn("ai", parsed)

    def test_grammar_contains_variant_words(self):
        m = PhoneticMatcher("AI")
        parsed = json.loads(m.build_vosk_grammar())
        # "ay" and "eye" should be individual grammar tokens
        self.assertIn("ay", parsed)
        self.assertIn("eye", parsed)

    def test_grammar_regular_word(self):
        m = PhoneticMatcher("hello")
        parsed = json.loads(m.build_vosk_grammar())
        self.assertIn("hello", parsed)
        self.assertIn("[unk]", parsed)


# =========================================================================
# Controls tests
# =========================================================================

class TestControls(unittest.TestCase):
    """Tests for start/stop/reset controls."""

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = _make_app(self.root)

    def tearDown(self):
        self.app.is_listening = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def test_reset_count(self):
        self.app.count = 10
        self.app._partial_count = 3
        self.app.reset_count()
        self.assertEqual(self.app.count, 0)
        self.assertEqual(self.app._partial_count, 0)

    def test_reset_clears_transcript(self):
        self.app._full_transcript = ["line1", "line2"]
        self.app._previous_partial = "partial"
        self.app.reset_count()
        self.assertEqual(self.app._full_transcript, [])
        self.assertEqual(self.app._previous_partial, "")

    def test_stop_listening_sets_flag(self):
        self.app.is_listening = True
        self.app.stop_listening()
        self.assertFalse(self.app.is_listening)

    def test_on_close_stops_listening(self):
        self.app.is_listening = True
        self.app._on_close()
        self.assertFalse(self.app.is_listening)

    @patch('word_counter.messagebox')
    def test_start_empty_word(self, mock_msgbox):
        self.app._model_loaded.set()
        self.app.word_entry.delete(0, tk.END)
        self.app.word_entry.insert(0, "")
        self.app.start_listening()
        self.assertFalse(self.app.is_listening)
        mock_msgbox.showwarning.assert_called_once()

    @patch('word_counter.messagebox')
    def test_start_no_microphone(self, mock_msgbox):
        self.app._model_loaded.set()
        self.app._mic_devices = []
        self.app.word_entry.delete(0, tk.END)
        self.app.word_entry.insert(0, "hello")
        self.app.start_listening()
        self.assertFalse(self.app.is_listening)
        mock_msgbox.showerror.assert_called_once()

    @patch('word_counter.messagebox')
    def test_start_model_not_loaded(self, mock_msgbox):
        self.app._model_loaded.clear()
        self.app.start_listening()
        self.assertFalse(self.app.is_listening)
        mock_msgbox.showinfo.assert_called_once()

    def test_no_devices_found(self):
        root = tk.Tk()
        root.withdraw()
        with patch('word_counter.pyaudio') as mock_pa:
            mock_instance = MagicMock()
            mock_instance.get_device_count.return_value = 0
            mock_pa.PyAudio.return_value = mock_instance
            app = WordCounterApp(root)
            self.assertEqual(len(app._mic_devices), 0)
        try:
            root.destroy()
        except tk.TclError:
            pass

    def test_double_start_guard(self):
        """Calling start_listening while already listening should be a no-op."""
        self.app._model_loaded.set()
        self.app.is_listening = True
        # Should return immediately without changing state
        self.app.start_listening()
        self.assertTrue(self.app.is_listening)

    def test_start_resets_partial_count(self):
        """Starting a new session must clear stale _partial_count and _peak_partial_count."""
        self.app._model_loaded.set()
        self.app._partial_count = 5  # leftover from previous session
        self.app._peak_partial_count = 5
        self.app.word_entry.delete(0, tk.END)
        self.app.word_entry.insert(0, "hello")
        # Simulate start (will fail at stream but sets state first)
        with patch.object(self.app, '_stream_loop'):
            with patch('threading.Thread') as mock_thread:
                mock_thread.return_value = MagicMock()
                self.app.start_listening()
                self.assertEqual(self.app._partial_count, 0)
                self.assertEqual(self.app._peak_partial_count, 0)

    def test_start_sets_calibration_status_and_state(self):
        """Start should enter calibration mode before active listening."""
        self.app._model_loaded.set()
        self.app.word_entry.delete(0, tk.END)
        self.app.word_entry.insert(0, "hello")
        with patch.object(self.app, '_stream_loop'):
            with patch('threading.Thread') as mock_thread:
                mock_thread.return_value = MagicMock()
                with patch.object(self.app, 'update_status') as mock_status:
                    self.app.start_listening()
                    self.assertFalse(self.app._calibration_done)
                    self.assertGreater(self.app._listening_started_at, 0.0)
                    mock_status.assert_called_with("Calibrating mic...", "#FF9800")

    def test_startup_calibration_seconds_constant(self):
        self.assertEqual(STARTUP_CALIBRATION_SECONDS, 3.0)

    def test_on_close_joins_thread(self):
        """_on_close should attempt to join the listen thread."""
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        self.app._listen_thread = mock_thread
        self.app.is_listening = True
        self.app._on_close()
        mock_thread.join.assert_called_once_with(timeout=2.0)
        self.assertFalse(self.app.is_listening)

    def test_on_close_no_thread(self):
        """_on_close should work even when no listen thread exists."""
        self.app._listen_thread = None
        self.app.is_listening = False
        # Should not raise
        self.app._on_close()
        self.assertFalse(self.app.is_listening)


# =========================================================================
# Error resilience tests
# =========================================================================

class TestErrorResilience(unittest.TestCase):
    """Test error handling and edge cases."""

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = _make_app(self.root)

    def tearDown(self):
        self.app.is_listening = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def test_phonetic_matcher_empty_string(self):
        """PhoneticMatcher should handle empty string without crashing."""
        m = PhoneticMatcher("")
        self.assertIsNotNone(m)
        self.assertEqual(m.count_matches("hello"), 0)

    def test_phonetic_matcher_whitespace_only(self):
        """PhoneticMatcher should handle whitespace input without crashing."""
        m = PhoneticMatcher("   ")
        self.assertIsNotNone(m)

    def test_phonetic_matcher_numeric_input(self):
        """PhoneticMatcher should handle numeric strings."""
        m = PhoneticMatcher("123")
        self.assertIsNotNone(m)
        self.assertEqual(m.count_matches("123 times"), 1)

    def test_phonetic_matcher_special_chars(self):
        """PhoneticMatcher should handle special characters safely."""
        m = PhoneticMatcher("C++")
        self.assertIsNotNone(m)
        # regex special chars should be escaped
        self.assertEqual(m.count_matches("I love C++"), 1)

    def test_count_word_with_malformed_text(self):
        """_count_word should survive unusual text."""
        self.app.target_word = "ai"
        self.app._matcher = PhoneticMatcher("AI")
        # Very long repeated string
        self.app._count_word("ai " * 10000)
        self.assertEqual(self.app.count, 10000)

    def test_handle_final_result_transcript_limit(self):
        """Transcript should not grow beyond MAX_TRANSCRIPT_LINES."""
        self.app.target_word = "x"
        self.app._matcher = PhoneticMatcher("x")
        for i in range(MAX_TRANSCRIPT_LINES + 30):
            self.app._handle_final_result(f"sentence {i}")
        self.assertEqual(len(self.app._full_transcript), MAX_TRANSCRIPT_LINES)
        # Most recent should be the last one
        self.assertEqual(self.app._full_transcript[-1], f"sentence {MAX_TRANSCRIPT_LINES + 29}")

    def test_reset_while_listening(self):
        """Reset should work even while listening."""
        self.app.is_listening = True
        self.app.count = 42
        self.app._full_transcript = ["a", "b"]
        self.app.reset_count()
        self.assertEqual(self.app.count, 0)
        self.assertEqual(self.app._full_transcript, [])


# =========================================================================
# Letter phonetics table integrity
# =========================================================================

class TestLetterPhoneticsTable(unittest.TestCase):
    """Verify the LETTER_PHONETICS table is complete and well-formed."""

    def test_all_26_letters(self):
        for c in 'abcdefghijklmnopqrstuvwxyz':
            self.assertIn(c, LETTER_PHONETICS, f"Missing letter: {c}")

    def test_values_are_non_empty_lists(self):
        for letter, sounds in LETTER_PHONETICS.items():
            self.assertIsInstance(sounds, list)
            self.assertTrue(len(sounds) > 0, f"No sounds for letter: {letter}")

    def test_values_are_lowercase_strings(self):
        for letter, sounds in LETTER_PHONETICS.items():
            for s in sounds:
                self.assertIsInstance(s, str)
                self.assertEqual(s, s.lower(), f"Non-lowercase sound '{s}' for letter '{letter}'")


if __name__ == "__main__":
    unittest.main()
