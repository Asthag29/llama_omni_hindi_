"""Trainable-parameter precision: fp32 LoRA/projector on a half-precision frozen base.

Uses a tiny OmniSpeech Llama (hidden size 64, 2 layers) with a stub speech encoder
built from whisper's dtype-adapting Conv1d, a real PEFT LoRA and the real speech
projector. The real training modules are exercised through subclasses that only
replace ``_load_model_and_tokenizer``.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from safetensors.torch import load_file

from omni_speech.constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX
from omni_speech.infer import inference
from omni_speech.model import omni_speech_arch
from omni_speech.model.language_model.omni_speech_llama import (
    OmniSpeechConfig,
    OmniSpeechLlamaForCausalLM,
)
from omni_speech.train_utils import (
    load_omni_speech_checkpoint,
    model_dtype,
    save_omni_speech_checkpoint,
)
from omni_speech.training.combined import OmniSpeechTrainingModule
from omni_speech.training.stage1 import BackboneTrainingModule

REPO_ROOT = Path(__file__).resolve().parents[1]
N_MELS = 8
ENCODER_DIM = 16
VOCAB = 128


class StubSpeechEncoder(nn.Module):
    """Whisper-like encoder: (B, n_mels, T) -> (B, T/2, D), adapts to the input dtype."""

    def __init__(self):
        super().__init__()
        from whisper.model import Conv1d

        self.conv = Conv1d(N_MELS, ENCODER_DIM, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return nn.functional.gelu(self.conv(x)).permute(0, 2, 1)


def tiny_config():
    config = OmniSpeechConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        speech_encoder="stub",
        speech_encoder_type="whisper",
        speech_projector_type="linear",
        speech_encoder_ds_rate=5,
        speech_encoder_hidden_size=ENCODER_DIM,
    )
    config.tokenizer_model_max_length = 128
    config.tokenizer_padding_side = "right"
    config._attn_implementation = "eager"
    return config


def build_tiny_model(dtype):
    torch.manual_seed(0)
    with mock.patch.object(omni_speech_arch, "build_speech_encoder", lambda cfg: StubSpeechEncoder()):
        model = OmniSpeechLlamaForCausalLM(tiny_config())
    # from_pretrained(torch_dtype=...) gives every parameter the configured dtype.
    return model.to(dtype)


class _TinyModelMixin:
    def _load_model_and_tokenizer(self):
        return None, build_tiny_model(model_dtype(self.cfg.training.precision))


class TinyBackboneModule(_TinyModelMixin, BackboneTrainingModule):
    pass


class TinyOmniSpeechModule(_TinyModelMixin, OmniSpeechTrainingModule):
    pass


def make_cfg(stage=2, precision="bf16-mixed", use_lora=True, tune_llm=True,
             gradient_checkpointing=False, init_checkpoint=None, lr=1e-5):
    return OmegaConf.create({
        "model": {"init_checkpoint": init_checkpoint},
        "training": {
            "precision": precision,
            "tune_llm_backbone": tune_llm,
            "use_lora": use_lora,
            "tune_speech_projector": stage == 2,
            "tune_speech_encoder": False,
            "lora_r": 8,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "gradient_checkpointing": gradient_checkpointing,
            "learning_rate": lr,
            "weight_decay": 0.01,
            "lr_scheduler_type": "constant",
        },
    })


def make_module(stage=2, **kwargs):
    cls = TinyOmniSpeechModule if stage == 2 else TinyBackboneModule
    return cls(make_cfg(stage=stage, **kwargs))


def make_batch(with_speech=True):
    torch.manual_seed(1)
    ids = [3, 5, 7, SPEECH_TOKEN_INDEX, 9, 11, 13, 15] if with_speech else [3, 5, 7, 9, 11, 13, 15, 17]
    input_ids = torch.tensor([ids, ids])
    labels = input_ids.clone()
    labels[:, :4] = IGNORE_INDEX
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "speech": torch.randn(2, 20, N_MELS),  # 20 frames -> 10 encoder frames -> 2 speech tokens
        "speech_lengths": torch.tensor([20, 20]),
    }


def forward_loss(module, batch):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        return module(batch).loss


def randomize_lora_b(module, std=0.02):
    # LoRA B starts at zero (so lora_A has no gradient); emulate a trained adapter.
    torch.manual_seed(2)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn_like(param.float()) * std)


def lora_a_params(module):
    return {n: p for n, p in module.named_parameters() if "lora_A" in n}


def trainable_state(module):
    return {n: p.detach().clone() for n, p in module.named_parameters() if p.requires_grad}


def fraction_lora_a_unchanged(module, steps=5, lr=1e-5):
    """Run AdamW steps with real autocast gradients; fraction of lora_A elements unchanged."""
    module.train()
    optimizer = torch.optim.AdamW(
        [p for p in module.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    before = {n: p.detach().clone() for n, p in lora_a_params(module).items()}
    batch = make_batch()
    for _ in range(steps):
        optimizer.zero_grad()
        forward_loss(module, batch).backward()
        optimizer.step()
    unchanged = sum(int((p.detach() == before[n]).sum()) for n, p in lora_a_params(module).items())
    total = sum(p.numel() for p in before.values())
    return unchanged / total


def demote_trainables_to_bf16(module):
    """The old behaviour: trainable params left in bf16 under bf16-mixed."""
    for param in module.parameters():
        if param.requires_grad:
            param.data = param.data.bfloat16()


def adam_second_moment_after_zero_grad_steps(dtype, steps=200):
    param = nn.Parameter(torch.ones(4, dtype=dtype))
    optimizer = torch.optim.AdamW([param], lr=1e-5, weight_decay=0.0)
    param.grad = torch.ones_like(param)
    optimizer.step()
    first = optimizer.state[param]["exp_avg_sq"].clone()
    for _ in range(steps):
        param.grad = torch.zeros_like(param)
        optimizer.step()
    return first, optimizer.state[param]["exp_avg_sq"]


class PromotionTests(unittest.TestCase):
    def assert_split_dtypes(self, module, trainable_dtype, frozen_dtype):
        for name, param in module.named_parameters():
            expected = trainable_dtype if param.requires_grad else frozen_dtype
            self.assertEqual(param.dtype, expected, name)

    def test_lora_trainables_fp32_frozen_bf16_and_adam_state_fp32(self):
        for stage in (1, 2):
            with self.subTest(stage=stage):
                module = make_module(stage=stage)
                self.assertTrue(any(p.requires_grad for p in module.parameters()))
                self.assert_split_dtypes(module, torch.float32, torch.bfloat16)
                projector_trainable = all(
                    p.requires_grad for p in module._get_inner_speech_model().speech_projector.parameters()
                )
                self.assertEqual(projector_trainable, stage == 2)

                optimizer = module.configure_optimizers()
                forward_loss(module, make_batch(with_speech=stage == 2)).backward()
                optimizer.step()
                self.assertTrue(optimizer.state)
                for param, state in optimizer.state.items():
                    self.assertEqual(param.dtype, torch.float32)
                    self.assertEqual(state["exp_avg"].dtype, torch.float32)
                    self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)

    def test_full_finetune_without_lora_is_not_promoted(self):
        module = make_module(stage=2, use_lora=False)
        self.assertGreater(sum(p.numel() for p in module.parameters() if p.requires_grad), 0)
        for name, param in module.named_parameters():
            self.assertEqual(param.dtype, torch.bfloat16, name)

    def test_small_updates_survive_in_fp32_but_not_bf16(self):
        fixed = make_module(stage=2)
        randomize_lora_b(fixed)
        old = make_module(stage=2)
        randomize_lora_b(old)
        demote_trainables_to_bf16(old)

        self.assertLess(fraction_lora_a_unchanged(fixed), 0.01)
        self.assertGreater(fraction_lora_a_unchanged(old), 0.5)

    def test_adam_second_moment_decays_in_fp32_not_bf16(self):
        first32, last32 = adam_second_moment_after_zero_grad_steps(torch.float32)
        self.assertTrue(torch.all(last32 < 0.9 * first32))
        first16, last16 = adam_second_moment_after_zero_grad_steps(torch.bfloat16)
        self.assertTrue(torch.equal(first16, last16))

    def test_gradients_reach_first_layer_lora_with_gradient_checkpointing(self):
        module = make_module(stage=2, gradient_checkpointing=True)
        randomize_lora_b(module)
        module.train()
        self.assertTrue(module.model.is_gradient_checkpointing)
        forward_loss(module, make_batch()).backward()
        first_layer = {
            n: p for n, p in module.named_parameters()
            if ".layers.0." in n and "lora_" in n
        }
        self.assertTrue(first_layer)
        for name, param in first_layer.items():
            self.assertIsNotNone(param.grad, name)
            self.assertEqual(param.grad.dtype, torch.float32, name)
            self.assertGreater(float(param.grad.abs().sum()), 0.0, name)
        for name, param in module._get_inner_speech_model().speech_projector.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertGreater(float(param.grad.abs().sum()), 0.0, name)

    def test_speech_forward_concatenates_in_embedding_dtype(self):
        module = make_module(stage=2)
        batch = make_batch()
        inner = module._get_inner_speech_model()
        self.assertEqual(inner.speech_projector.linear1.weight.dtype, torch.float32)
        self.assertEqual(inner.embed_tokens.weight.dtype, torch.bfloat16)
        speech = batch["speech"].to(module.speech_dtype)

        # Lightning's bf16-mixed wraps training AND validation forwards in autocast.
        for training in (True, False):
            module.train(training)
            with self.subTest(training=training), torch.set_grad_enabled(training), \
                    torch.autocast("cpu", dtype=torch.bfloat16):
                features = module.model.encode_speech(speech, batch["speech_lengths"])
                self.assertEqual({f.dtype for f in features}, {torch.bfloat16})
                _, _, _, _, embeds, _ = module.model.prepare_inputs_labels_for_speech_and_text(
                    batch["input_ids"], None, batch["attention_mask"], None, batch["labels"],
                    speech, batch["speech_lengths"],
                )
                self.assertEqual(embeds.dtype, inner.embed_tokens.weight.dtype)
                self.assertEqual(embeds.shape[1], batch["input_ids"].shape[1] - 1 + 2)
                loss = module(batch).loss
                self.assertTrue(torch.isfinite(loss))
                if training:
                    loss.backward()
                    self.assertIsNotNone(inner.speech_projector.linear1.weight.grad)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def randomize_trainables(module):
        torch.manual_seed(3)
        with torch.no_grad():
            for param in module.parameters():
                if param.requires_grad:
                    param.copy_(torch.randn_like(param) * 0.05)

    def saved_dtypes(self, path):
        dtypes = set()
        for name in ("adapter_model.safetensors", "speech_projector.safetensors"):
            dtypes |= {t.dtype for t in load_file(str(path / name)).values()}
        return dtypes

    def test_fp32_round_trip_is_bit_exact(self):
        module = make_module(stage=2)
        self.randomize_trainables(module)
        expected = trainable_state(module)
        save_omni_speech_checkpoint(module, str(self.tmp / "ckpt"))
        self.assertEqual(self.saved_dtypes(self.tmp / "ckpt"), {torch.float32})

        fresh = make_module(stage=2)
        load_omni_speech_checkpoint(fresh, str(self.tmp / "ckpt"), adapter_trainable=True)
        loaded = trainable_state(fresh)
        self.assertEqual(loaded.keys(), expected.keys())
        for name, value in expected.items():
            self.assertEqual(loaded[name].dtype, torch.float32, name)
            self.assertTrue(torch.equal(loaded[name], value), name)

    def test_stage2_init_from_fp32_stage1_checkpoint_is_bit_exact(self):
        stage1 = make_module(stage=1)
        self.randomize_trainables(stage1)
        expected = {n: p for n, p in trainable_state(stage1).items() if "lora_" in n}
        save_omni_speech_checkpoint(stage1, str(self.tmp / "stage1"))
        self.assertIn(torch.float32, self.saved_dtypes(self.tmp / "stage1"))

        stage2 = make_module(stage=2, init_checkpoint=str(self.tmp / "stage1"))
        loaded = trainable_state(stage2)
        for name, value in expected.items():
            self.assertEqual(loaded[name].dtype, torch.float32, name)
            self.assertTrue(torch.equal(loaded[name], value), name)

    def test_bf16_checkpoint_loads_as_upcast_values(self):
        module = make_module(stage=2)
        self.randomize_trainables(module)
        demote_trainables_to_bf16(module)  # what the old training saved
        expected = trainable_state(module)
        save_omni_speech_checkpoint(module, str(self.tmp / "bf16"))
        self.assertEqual(self.saved_dtypes(self.tmp / "bf16"), {torch.bfloat16})

        fresh = make_module(stage=2, init_checkpoint=str(self.tmp / "bf16"))
        loaded = trainable_state(fresh)
        for name, value in expected.items():
            self.assertEqual(loaded[name].dtype, torch.float32, name)
            self.assertTrue(torch.equal(loaded[name], value.float()), name)


class InferenceLoadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        module = make_module(stage=2)
        CheckpointTests.randomize_trainables(module)
        save_omni_speech_checkpoint(module, str(tmp / "fp32"))
        demote_trainables_to_bf16(module)
        save_omni_speech_checkpoint(module, str(tmp / "bf16"))
        cls.checkpoints = {"fp32": tmp / "fp32", "bf16": tmp / "bf16"}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def load(self, checkpoint, precision):
        cfg = make_cfg(stage=2, precision=precision)
        with mock.patch.object(inference, "OmniSpeechTrainingModule", TinyOmniSpeechModule):
            module, device = inference.load_module_from_checkpoint(checkpoint, cfg, requested_device="cpu")
        return module, device

    def generate(self, module, dtype):
        batch = make_batch()
        speech = batch["speech"][:1].to(dtype=inference.speech_input_dtype(module.model))
        input_ids = batch["input_ids"][:1]
        with torch.inference_mode():
            return module.model.generate(
                input_ids, speech=speech, speech_lengths=batch["speech_lengths"][:1],
                max_new_tokens=3, do_sample=False, num_beams=1, use_cache=True,
                pad_token_id=0, eos_token_id=VOCAB + 1,
            )

    def test_all_parameters_share_the_inference_dtype(self):
        for precision, expected in (("bf16-mixed", torch.bfloat16), ("32-true", torch.float32)):
            for kind, path in self.checkpoints.items():
                with self.subTest(precision=precision, checkpoint=kind):
                    module, _ = self.load(path, precision)
                    dtypes = {p.dtype for p in module.parameters()}
                    self.assertEqual(dtypes, {expected})
                    self.assertEqual(inference.speech_input_dtype(module.model), expected)
                    output = self.generate(module, expected)
                    self.assertEqual(output.shape[0], 1)
                    self.assertGreaterEqual(output.shape[1], 1)

    def test_loaded_values_match_checkpoint(self):
        module, _ = self.load(self.checkpoints["fp32"], "32-true")
        saved = load_file(str(self.checkpoints["fp32"] / "speech_projector.safetensors"))
        projector = module._get_inner_speech_model().speech_projector.state_dict()
        for name, value in saved.items():
            self.assertTrue(torch.equal(projector[name], value), name)


class SpeechDtypeHelperTests(unittest.TestCase):
    def test_helper_returns_model_dtype(self):
        for dtype in (torch.bfloat16, torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                model = build_tiny_model(dtype)
                self.assertEqual(inference.speech_input_dtype(model), dtype)

    def test_helper_on_peft_wrapped_inference_module(self):
        module = make_module(stage=2)
        for param in module.parameters():
            param.data = param.data.bfloat16()
        self.assertEqual(inference.speech_input_dtype(module.model), torch.bfloat16)

    def test_model_worker_uses_helper(self):
        # Importing the worker redirects stdout/stderr and writes a log file in the
        # cwd, so check it in a subprocess running in a temporary directory.
        code = (
            "import omni_speech.serve.model_worker as w, omni_speech.infer.inference as i, sys;"
            "sys.exit(0 if w.speech_input_dtype is i.speech_input_dtype else 3)"
        )
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
            result = subprocess.run(
                [sys.executable, "-c", code], cwd=tmp, env=env,
                capture_output=True, text=True, timeout=300,
            )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        source = (REPO_ROOT / "omni_speech" / "serve" / "model_worker.py").read_text(encoding="utf-8")
        self.assertIn("speech_input_dtype(model)", source)
        self.assertNotIn("torch.float16 if", source)


if __name__ == "__main__":
    unittest.main()
