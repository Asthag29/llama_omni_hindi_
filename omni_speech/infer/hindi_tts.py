import torch
import scipy
import numpy as np
from transformers import VitsModel, AutoTokenizer

class HindiTTSBridge:
    def __init__(self, device='cuda'):
        print('Loading MMS-TTS Hindi...')
        self.model = VitsModel.from_pretrained('facebook/mms-tts-hin').to(device).eval()
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