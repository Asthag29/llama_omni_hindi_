from transformers import AutoModel
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import torch
import numpy as np
import soundfile as sf

# Load INF5 from Hugging Face
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)


def restore_grn_weights(model, repo_id):
    # transformers (<4.48) blindly renames any param containing "gamma"/"beta"
    # to "weight"/"bias" during load. IndicF5's ConvNeXt-v2 GRN layers are
    # literally named gamma/beta, so those tensors get dropped and the model
    # params are left randomly initialized (corrupts/skips audio). Reload them
    # directly from the checkpoint. Scoped to IndicF5 only; does not touch the
    # transformers library or affect any other model (Whisper/LLaMA) loading.
    ckpt = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    sd = load_file(ckpt)
    targets = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    restored = 0
    with torch.no_grad():
        for k, v in sd.items():
            if (k.endswith(".gamma") or k.endswith(".beta")) and k in targets:
                t = targets[k]
                assert t.shape == v.shape, f"shape mismatch for {k}"
                t.copy_(v.to(t.dtype).to(t.device))
                restored += 1
    print(f"[IndicF5] Restored {restored} GRN gamma/beta tensors from checkpoint.")


restore_grn_weights(model, repo_id)

# Generate speech
audio = model(
    "नमस्ते! मेरा नाम आस्था गुप्ता है ",
    ref_audio_path="astha.wav",
    ref_text="मेरा नाम आस्था गुप्ता है मेरी माता का नाम बिभा गुप्ता है मेरे पिता का नाम संजय कुमार है")

# Normalize and save output
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
sf.write("astha_generated.wav", np.array(audio, dtype=np.float32), samplerate=24000)