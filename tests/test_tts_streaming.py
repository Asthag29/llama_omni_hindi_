"""Streaming speech: sentences are split as text arrives and spoken in growing passes."""

import queue
import struct
import threading
import unittest

import numpy as np

from omni_speech.tts import indicf5, streaming


class PopSentencesTests(unittest.TestCase):
    def test_only_finished_sentences_are_returned(self):
        text = "मुझे मदद करने में खुशी होगी! आप आज कैसा महसूस कर रहे हैं? मैं ठीक"
        sentences, rest = streaming.pop_sentences(text, 0)
        self.assertEqual(sentences, ["मुझे मदद करने में खुशी होगी!", "आप आज कैसा महसूस कर रहे हैं?"])
        self.assertEqual(text[rest:].strip(), "मैं ठीक")

    def test_a_very_short_sentence_waits_for_the_next_one(self):
        # "ज़रूर!" alone is too little audio for a browser to start playing.
        sentences, _ = streaming.pop_sentences("ज़रूर! मुझे मदद करने में खुशी होगी। आगे", 0)
        self.assertEqual(sentences, ["ज़रूर! मुझे मदद करने में खुशी होगी।"])

    def test_a_mark_at_the_very_end_is_not_final_yet(self):
        # More marks may still follow ("?!"), so wait for the next sentence to start.
        self.assertEqual(streaming.pop_sentences("यह भारत की सबसे लंबी नदी है।", 0), ([], 0))

    def test_list_numbers_and_decimals_do_not_end_a_sentence(self):
        text = "भारत में कई पवित्र नदियाँ हैं। 2. यमुना 3.5 किमी लंबी है। अगला"
        sentences, _ = streaming.pop_sentences(text, 0)
        self.assertEqual(sentences, ["भारत में कई पवित्र नदियाँ हैं।", "2. यमुना 3.5 किमी लंबी है।"])

    def test_streaming_piece_by_piece_matches_splitting_at_once(self):
        text = ("ज़रूर, मुझे मदद करने में खुशी होगी! यह एक ऐतिहासिक मस्जिद है। "
                "क्या आप इसके बारे में जानते हैं? यह अकबर ने बनाई थी और प्रसिद्ध है। अंत")
        streamed, start = [], 0
        for end in range(1, len(text) + 1):
            sentences, start = streaming.pop_sentences(text[:end].strip(), start)
            streamed += sentences
        self.assertEqual(streamed, streaming.pop_sentences(text, 0)[0])
        self.assertEqual(len(streamed), 4)


class SpeakSentencesTests(unittest.TestCase):
    def speak(self, items, max_bytes=60, cancelled=None):
        sentences, audio, spoken = queue.Queue(), queue.Queue(), []
        for item in items:
            sentences.put(item)

        def synthesize(text):
            spoken.append(text)
            return 24000, np.zeros(4, dtype=np.float32)

        streaming.speak_sentences(sentences, audio, synthesize, max_bytes, cancelled or threading.Event())
        results = []
        while not audio.empty():
            results.append(audio.get())
        return spoken, results

    def test_first_pass_is_one_sentence_and_later_passes_grow(self):
        sentences = [f"s{i:02d}" for i in range(12)]  # 3 bytes each
        spoken, results = self.speak([*sentences, None], max_bytes=15)
        # Each pass may be twice the one before, up to max_bytes.
        self.assertEqual(spoken, ["s00", "s01 s02", "s03 s04 s05 s06", "s07 s08 s09 s10", "s11"])
        self.assertEqual(len(results), len(spoken) + 1)
        self.assertIs(results[-1], streaming.DONE)

    def test_sentences_that_arrive_later_are_spoken_in_order(self):
        sentences, audio, spoken = queue.Queue(), queue.Queue(), []
        gate = threading.Event()

        def synthesize(text):
            spoken.append(text)
            gate.set()
            return None

        thread = threading.Thread(
            target=streaming.speak_sentences, args=(sentences, audio, synthesize, 60, threading.Event())
        )
        thread.start()
        sentences.put("पहला।")
        self.assertTrue(gate.wait(timeout=5))  # spoken before the rest of the answer exists
        sentences.put("दूसरा।")
        sentences.put(None)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(spoken, ["पहला।", "दूसरा।"])

    def test_a_sentence_longer_than_the_budget_is_still_spoken(self):
        spoken, results = self.speak(["छोटा।", "x" * 500, None])
        self.assertEqual(spoken, ["छोटा।", "x" * 500])
        self.assertIs(results[-1], streaming.DONE)

    def test_cancelling_stops_before_the_next_pass(self):
        cancelled = threading.Event()
        cancelled.set()
        spoken, results = self.speak(["एक।", "दो।", None], cancelled=cancelled)
        self.assertEqual(spoken, [])
        self.assertEqual(results, [streaming.DONE])


class AudioFormatTests(unittest.TestCase):
    def test_default_reference_leaves_most_of_the_window_for_new_speech(self):
        budget = streaming.max_pass_bytes(indicf5.DEFAULT_REFERENCE_AUDIO, indicf5.DEFAULT_REFERENCE_TEXT)
        self.assertGreater(budget, 2 * len(indicf5.DEFAULT_REFERENCE_TEXT.encode("utf-8")))

    def test_stream_header_describes_16_bit_audio_of_unknown_length(self):
        header = streaming.wav_stream_header()
        fields = struct.unpack("<4sI4s4sIHHIIHH4sI", header)
        self.assertEqual(len(header), 44)
        self.assertEqual(fields[:3], (b"RIFF", 0xFFFFFFFF, b"WAVE"))
        self.assertEqual(fields[5:11], (1, 2, 48000, 192000, 4, 16))
        self.assertEqual(fields[11:], (b"data", 0xFFFFFFFF))

    def test_stream_bytes_match_the_header_format(self):
        one_second = 0.5 * np.sin(np.linspace(0, 2 * np.pi * 220, 24000, dtype=np.float32))
        frames = np.frombuffer(streaming.stream_bytes(one_second, 24000), dtype="<i2").reshape(-1, 2)
        self.assertEqual(len(frames), 48000)
        self.assertTrue(np.array_equal(frames[:, 0], frames[:, 1]))
        self.assertAlmostEqual(np.abs(frames).max() / 32767, 0.5, delta=0.02)

    def test_pcm16_clips_and_scales(self):
        samples = streaming.pcm16(np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32))
        self.assertEqual(samples.dtype, np.dtype("<i2"))
        self.assertEqual(samples.tolist(), [0, 32767, -32767, 32767])


if __name__ == "__main__":
    unittest.main()
