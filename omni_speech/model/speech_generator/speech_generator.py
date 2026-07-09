from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from huggingface_hub.errors import HFValidationError


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
            try:
                from transformers import AutoConfig, AutoModel
            except ImportError as exc:
                raise RuntimeError("IndicF5 requires transformers to be installed.") from exc

            if self.model_path.exists():
                model_source = str(self.model_path.resolve())
                config = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
                # IndicF5's remote code uses config.name_or_path with hf_hub_download
                # for vocab lookup, so keep that value as the Hub repo id even when
                # loading the weights from a local snapshot.
                config.name_or_path = self.repo_id
                try:
                    self._model = AutoModel.from_pretrained(model_source, config=config, trust_remote_code=True)
                except HFValidationError:
                    # The upstream IndicF5 modeling code currently calls
                    # hf_hub_download(config.name_or_path, ...). Transformers
                    # rewrites name_or_path to the local folder during local
                    # loading, so fall back to the Hub repo while using the
                    # already-populated cache/downloaded snapshot.
                    self._model = AutoModel.from_pretrained(self.repo_id, trust_remote_code=True)
            else:
                self._model = AutoModel.from_pretrained(self.repo_id, trust_remote_code=True)
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
