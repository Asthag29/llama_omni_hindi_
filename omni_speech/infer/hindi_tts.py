import torch
import scipy
import numpy as np
from transformers import VitsModel, AutoTokenizer

class HindiTTSBridge:
    """Replaces the HiFi-GAN unit vocoder with MMS-TTS Hindi."""

    def __init__(self, device='cuda'):
            print('Loading MMS-TTS Hindi...')
            self.model = VitsModel.from_pretrained('facebook/mms-tts-hin').to(device)   # Load the MMS-TTS Hindi model
            self.tokenizer = AutoTokenizer.from_pretrained('facebook/mms-tts-hin')  # Load the tokenizer for MMS-TTS Hindi
            self.model.eval()  # Set the model to evaluation mode   
            self.device = device  # Set the device to the device specified  
            self.sample_rate = self.model.config.sampling_rate  # Set the sample rate to the sample rate of the model 

    def synthesize(self, hindi_text: str) -> np.ndarray:
            """Convert Hindi text to audio waveform."""
            text = hindi_text.replace('<s>', '').replace('</s>', '').strip()  # Remove the special tokens start and end of the text
            if not text:
                return np.zeros(self.sample_rate // 4) # 0.25s silence for empty text
            inputs = self.tokenizer(text, return_tensors='pt').to(self.device) #return output as a tensor
            with torch.no_grad():
                output = self.model(**inputs).waveform #generate waveform from the model
            return output.squeeze().cpu().numpy() #return the wavefo rm as a numpy array
            
    def save(self, waveform: np.ndarray, path: str):
        scipy.io.wavfile.write(path, self.sample_rate, waveform) #save  encoded waveform to a file