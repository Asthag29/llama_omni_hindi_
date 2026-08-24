import unittest
from types import SimpleNamespace

import torch

from omni_speech.constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX
from omni_speech.datasets.preprocess import preprocess, tokenizer_speech_token


class DummyTokenizer:
    bos_token_id = 1
    pad_token_id = 0
    model_max_length = 128

    def _encode(self, text):
        tokens = [2 + (len(word) % 20) for word in text.split()]
        return [self.bos_token_id, *tokens]

    def __call__(self, text, return_tensors=None, **kwargs):
        if isinstance(text, list):
            encoded = [self._encode(item) for item in text]
            max_length = max(len(item) for item in encoded)
            encoded = [
                item + [self.pad_token_id] * (max_length - len(item))
                for item in encoded
            ]
            return SimpleNamespace(input_ids=torch.tensor(encoded, dtype=torch.long))
        return SimpleNamespace(input_ids=self._encode(text))


class PreprocessTests(unittest.TestCase):
    def test_llama3_preprocess_masks_instruction_tokens(self):
        source = [[
            {"from": "human", "value": "नमस्ते"},
            {"from": "gpt", "value": "कैसे हो?"},
        ]]

        result = preprocess(source, DummyTokenizer())

        self.assertEqual(tuple(result["input_ids"].shape), tuple(result["labels"].shape))
        self.assertIn(IGNORE_INDEX, result["labels"].tolist()[0])

    def test_speech_placeholder_becomes_speech_token(self):
        token_ids = tokenizer_speech_token(
            "<speech>\nप्रश्न",
            DummyTokenizer(),
        )

        self.assertIn(SPEECH_TOKEN_INDEX, token_ids)


if __name__ == "__main__":
    unittest.main()
