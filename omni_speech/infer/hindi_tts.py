import torch
import scipy
import numpy as np
from transformers import VitsModel, VitsConfig, AutoTokenizer

class HindiTTSBridge:
    def __init__(self, device='cuda'):
        print('Loading MMS-TTS Hindi...')
        from huggingface_hub import hf_hub_download

        # Load and remap weights (weight_g/weight_v → parametrizations)
        path = hf_hub_download("facebook/mms-tts-hin", "pytorch_model.bin")
        sd = torch.load(path, map_location="cpu")
#! remapping the weights to mak it adaptable for the new tranasformer version 
        new_sd = {}
        for k, v in sd.items():
            if k.endswith(".weight_g"):
                new_sd[k.replace(".weight_g", ".parametrizations.weight.original0")] = v
            elif k.endswith(".weight_v"):
                new_sd[k.replace(".weight_v", ".parametrizations.weight.original1")] = v
            else:
                new_sd[k] = v

        config = VitsConfig.from_pretrained("facebook/mms-tts-hin")
        self.model = VitsModel(config)
        self.model.load_state_dict(new_sd, strict=True)
        self.model.to(device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained('facebook/mms-tts-hin')
        self.device = device
        self.sample_rate = self.model.config.sampling_rate

    def synthesize(self, hindi_text: str) -> np.ndarray:
        text = hindi_text.replace('<s>', '').replace('</s>', '').strip()
        if not text:
            return np.zeros(self.sample_rate // 4)
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            output = self.model(**inputs).waveform
        return output.squeeze().cpu().numpy()

    def save(self, waveform: np.ndarray, path: str):
        scipy.io.wavfile.write(path, self.sample_rate, waveform)