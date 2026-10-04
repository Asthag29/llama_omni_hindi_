"""inference.py: full answers are decoded, audio is prepared one way, and both modes work."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from omni_speech.constants import DEFAULT_MAX_NEW_TOKENS
from omni_speech.infer import inference
from omni_speech.train_utils import load_audio_16k
from omni_speech.tts import indicf5

N_MELS = 80
ANSWER_TOKENS = 300


class StubTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text):
        return SimpleNamespace(input_ids=[self.bos_token_id] + [5] * len(text.split()))

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return [" ".join(str(int(token)) for token in row) for row in token_ids]


class StubModel(nn.Module):
    """Like the real generate(): runs on inputs_embeds, so it returns only the new tokens."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.calls = []

    def generate(self, input_ids, speech=None, speech_lengths=None, **kwargs):
        self.calls.append({"speech": speech, "speech_lengths": speech_lengths, **kwargs})
        return torch.arange(10, 10 + ANSWER_TOKENS).unsqueeze(0)


def write_wav(path, seconds=0.5, sample_rate=48000):
    samples = np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, int(seconds * sample_rate)))
    sf.write(str(path), (0.3 * samples).astype(np.float32), sample_rate, subtype="PCM_16")
    return path


class GenerateFromWavTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.audio = write_wav(Path(self._tmp.name) / "question.wav")
        self.module = SimpleNamespace(model=StubModel(), tokenizer=StubTokenizer())
        self.cfg = OmegaConf.create({"data": {"mel_size": N_MELS}})

    def generate(self, **kwargs):
        return inference.generate_from_wav(
            audio_path=self.audio, prompt=inference.DEFAULT_PROMPT, cfg=self.cfg,
            module=self.module, device=torch.device("cpu"), **kwargs,
        )

    def test_answer_longer_than_the_prompt_keeps_its_first_tokens(self):
        # The answer has more tokens than the prompt; none of them is prompt.
        expected = " ".join(str(token) for token in range(10, 10 + ANSWER_TOKENS))
        self.assertEqual(self.generate(), expected)

    def test_default_length_limit_matches_the_demo(self):
        self.generate()
        self.assertEqual(self.module.model.calls[-1]["max_new_tokens"], DEFAULT_MAX_NEW_TOKENS)
        self.assertEqual(inference.MAX_NEW_TOKENS, 512)


class SpeechInputsTests(unittest.TestCase):
    def test_file_path_and_demo_samples_give_identical_features(self):
        # The demo sends load_audio_16k(...) samples as a JSON list to the worker, which
        # calls speech_inputs; inference.py goes through prepare_speech.
        model = StubModel()
        cfg = OmegaConf.create({"data": {"mel_size": N_MELS}})
        with tempfile.TemporaryDirectory() as tmp:
            audio = write_wav(Path(tmp) / "question.wav")
            from_path, path_lengths = inference.prepare_speech(
                audio, cfg, SimpleNamespace(model=model), torch.device("cpu")
            )
            samples = load_audio_16k(audio).tolist()
        from_samples, sample_lengths = inference.speech_inputs(samples, N_MELS, model, "cpu")
        self.assertTrue(torch.equal(from_path, from_samples))
        self.assertTrue(torch.equal(path_lengths, sample_lengths))
        self.assertEqual(tuple(from_path.shape), (1, 3000, N_MELS))


class ModeTests(unittest.TestCase):
    def parse(self, *argv):
        with mock.patch.object(sys, "argv", ["inference.py", *argv]):
            return inference.parse_args()

    def test_text_is_the_default_mode(self):
        self.assertEqual(self.parse().mode, inference.AUDIO_TO_TEXT)
        self.assertEqual(self.parse("--mode", "audio-to-audio").mode, inference.AUDIO_TO_AUDIO)

    def test_synthesize_answer_writes_the_spoken_answer(self):
        generator = mock.Mock()
        generator.synthesize.return_value = (24000, np.zeros(2400, dtype=np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            output = inference.synthesize_answer("नमस्ते", Path(tmp) / "nested" / "answer.wav", generator)
            info = sf.info(str(output))
        generator.synthesize.assert_called_once_with(
            "नमस्ते", indicf5.DEFAULT_REFERENCE_AUDIO, indicf5.DEFAULT_REFERENCE_TEXT
        )
        self.assertEqual((info.samplerate, info.frames), (24000, 2400))

    def test_own_voice_needs_the_clip_and_its_transcript_together(self):
        self.assertEqual(indicf5.resolve_reference(),
                         (indicf5.DEFAULT_REFERENCE_AUDIO, indicf5.DEFAULT_REFERENCE_TEXT))
        with tempfile.TemporaryDirectory() as tmp:
            clip = write_wav(Path(tmp) / "my_voice.wav")
            self.assertEqual(indicf5.resolve_reference(clip, " मेरी आवाज़ "), (clip.resolve(), "मेरी आवाज़"))
            with self.assertRaises(ValueError):
                indicf5.resolve_reference(clip, None)
            with self.assertRaises(ValueError):
                indicf5.resolve_reference(None, "मेरी आवाज़")
            with self.assertRaises(FileNotFoundError):
                indicf5.resolve_reference(Path(tmp) / "missing.wav", "मेरी आवाज़")

    def test_synthesize_answer_uses_the_given_voice(self):
        generator = mock.Mock()
        generator.synthesize.return_value = (24000, np.zeros(2400, dtype=np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "my_voice.wav"
            inference.synthesize_answer("नमस्ते", Path(tmp) / "answer.wav", generator, clip, "मेरी आवाज़")
        generator.synthesize.assert_called_once_with("नमस्ते", clip, "मेरी आवाज़")
        args = self.parse("--reference-audio", "my_voice.wav", "--reference-text", "मेरी आवाज़")
        self.assertEqual((args.reference_audio, args.reference_text), (Path("my_voice.wav"), "मेरी आवाज़"))

    def test_reference_voice_is_a_short_tracked_clip(self):
        # IndicF5 re-reads the reference on every chunk, so a long clip slows every answer.
        self.assertLess(sf.info(str(indicf5.DEFAULT_REFERENCE_AUDIO)).duration, 6.0)
        self.assertTrue(indicf5.DEFAULT_REFERENCE_TEXT.strip())


if __name__ == "__main__":
    unittest.main()
