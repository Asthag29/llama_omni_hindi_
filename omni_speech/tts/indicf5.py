from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

# Fixed reference voice shared by inference.py and the Gradio demo.
DEFAULT_REFERENCE_AUDIO = Path(__file__).resolve().parents[2] / "data" / "inference.wav"
DEFAULT_REFERENCE_TEXT = (
    "तितली रानी तितली रानी, तितली रानी, इतने सुंदर पंख कहां से लाई हो। "
    "क्या तुम कोई हो शहजादी, या परी लोक से आई हो। "
    "फूल तुम्हें भी अच्छे लगते, फूल हमें भी भाते है।"
)


class IndicF5SpeechGenerator:
    """Lazy IndicF5 TTS wrapper for cloning the user's reference voice."""

    sample_rate = 24000

    def __init__(
        self,
        model_path: str | Path = "models/indicf5",
        repo_id: str = "ai4bharat/IndicF5",
        device: str | None = None,
    ):
        self.model_path = Path(model_path)
        self.repo_id = repo_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not self.model_path.exists():
                # First run: fetch the snapshot into model_path so the local path always exists
                snapshot_download(self.repo_id, local_dir=str(self.model_path))
            model_source = str(self.model_path.resolve())
            config = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
            # IndicF5's remote code calls hf_hub_download(config.name_or_path, "checkpoints/vocab.txt"),
            # so this must stay the Hub repo id, not the local folder.
            config.name_or_path = self.repo_id
            model_cls = get_class_from_dynamic_module(config.auto_map["AutoModel"], model_source)
            self._model = model_cls(config)
            state = load_file(str(self.model_path / "model.safetensors"), device="cpu")
            # Deliberately not from_pretrained: Transformers 4.43.4 renames gamma/beta keys
            # to weight/bias while loading and silently drops IndicF5's 16 GRN / layer-scale
            # tensors. strict=True guarantees every checkpoint tensor lands.
            self._model.load_state_dict(state, strict=True)
            if hasattr(self._model, "to"):
                self._model = self._model.to(self.device)
            if hasattr(self._model, "eval"):
                self._model.eval()
        return self._model

    @torch.inference_mode()
    def synthesize(self, text: str, ref_audio_path: str | Path, ref_text: str) -> tuple[int, np.ndarray]:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text with IndicF5.")
        if not ref_text.strip():
            raise ValueError("IndicF5 requires the transcript of the reference audio.")

        audio = self.model(
            text.strip(),
            ref_audio_path=str(ref_audio_path),
            ref_text=ref_text.strip(),
        )
        audio = np.asarray(audio)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32)
        return self.sample_rate, audio
