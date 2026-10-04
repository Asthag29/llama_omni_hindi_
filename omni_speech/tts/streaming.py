"""Speak an answer while it is still being written: sentence splitting and chunked synthesis."""

from __future__ import annotations

import queue
import re
import struct

import numpy as np
import soundfile as sf
import torch
import torchaudio

# IndicF5 fits the reference clip and the new speech into one window of this many seconds.
INDICF5_WINDOW_SECONDS = 25

# Put on the audio queue once every sentence has been spoken.
DONE = object()

# A browser starts playing a stream only once about 330 KB of it has arrived. 48 kHz stereo
# reaches that after under 2 s of speech; IndicF5's own 24 kHz mono would need 7 s.
STREAM_SAMPLE_RATE = 48000
STREAM_CHANNELS = 2
# Shorter sentences wait for the next one: on their own they are too little audio to start playback.
MIN_SENTENCE_CHARS = 20

# A short first sentence is over long before the second pass is ready, which leaves a silent gap
# in the middle of the answer. Playback therefore starts only once the speech in hand plus the
# time already waited add up to this many seconds, or as soon as the second pass is ready.
START_COVER_SECONDS = 15.0

# । ? ! always end a sentence; a full stop only after a non-digit ("2." numbers a list).
# The whitespace lookahead means the next sentence has started, so the mark is final.
_SENTENCE_END = re.compile(r"(?:[।॥?!]+|(?<![0-9०-९])\.+)(?=\s)")


def pop_sentences(text: str, start: int) -> tuple[list[str], int]:
    """Finished sentences in text[start:], and the index where the unfinished rest begins."""
    sentences = []
    for match in _SENTENCE_END.finditer(text, start):
        sentence = text[start:match.end()].strip()
        if len(sentence) < MIN_SENTENCE_CHARS:
            continue
        sentences.append(sentence)
        start = match.end()
    return sentences, start


def max_pass_bytes(ref_audio_path, ref_text: str) -> int:
    """About the longest text IndicF5 speaks in one pass with this reference (its own rule)."""
    seconds = sf.info(str(ref_audio_path)).duration
    return int(len(ref_text.encode("utf-8")) / seconds * (INDICF5_WINDOW_SECONDS - seconds))


def _size(text: str) -> int:
    return len(text.encode("utf-8"))


def speak_sentences(sentences: queue.Queue, audio: queue.Queue, synthesize, max_bytes: int, cancelled) -> None:
    """Thread body: speak the queued sentences in order, one IndicF5 pass at a time.

    sentences yields strings and then None. The first pass takes a single sentence so speech
    starts early; each later pass may be twice as long as the one before, up to max_bytes,
    which keeps synthesis ahead of playback. audio receives what synthesize(text) returns
    for every pass, then DONE.
    """
    budget = 0
    waiting = None
    finished = False
    try:
        while not finished and not cancelled.is_set():
            batch = waiting if waiting is not None else sentences.get()
            waiting = None
            if batch is None:
                break
            # Add the sentences that arrived in the meantime, as far as the budget allows.
            while True:
                try:
                    sentence = sentences.get_nowait()
                except queue.Empty:
                    break
                if sentence is None:
                    finished = True
                    break
                if _size(batch) + _size(sentence) > budget:
                    waiting = sentence
                    break
                batch = f"{batch} {sentence}"
            if cancelled.is_set():
                break
            audio.put(synthesize(batch))
            budget = min(max_bytes, 2 * _size(batch))
    finally:
        audio.put(DONE)


def start_playback(passes_ready: int, seconds_ready: float, waited: float, finished: bool) -> bool:
    """Whether the live player may start: waiting now is better than a gap after the first pass."""
    if passes_ready == 0:
        return False
    return finished or passes_ready > 1 or seconds_ready + waited >= START_COVER_SECONDS


def pcm16(samples) -> np.ndarray:
    """Float samples in [-1, 1] as 16-bit PCM."""
    return (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2")


def stream_bytes(samples, sample_rate: int) -> bytes:
    """Mono float samples as the 16-bit PCM the live player streams (see STREAM_SAMPLE_RATE)."""
    samples = torch.from_numpy(np.asarray(samples, dtype=np.float32))
    samples = torchaudio.functional.resample(samples, sample_rate, STREAM_SAMPLE_RATE).numpy()
    return np.repeat(pcm16(samples)[:, None], STREAM_CHANNELS, axis=1).tobytes()


def wav_stream_header() -> bytes:
    """WAV header for a 16-bit stream (see STREAM_SAMPLE_RATE) whose length is not known yet."""
    unknown = 0xFFFFFFFF
    frame_bytes = 2 * STREAM_CHANNELS
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", unknown, b"WAVE", b"fmt ", 16, 1, STREAM_CHANNELS, STREAM_SAMPLE_RATE,
        STREAM_SAMPLE_RATE * frame_bytes, frame_bytes, 16, b"data", unknown,
    )
