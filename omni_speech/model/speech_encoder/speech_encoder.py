# Adopted from https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/encoder.py

import types
import torch
import torch.nn as nn
import torch.nn.functional as F
#! we have only used encoder of the whisper model , which still produces audio tokens

class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_config):

        def replace_layer_norm(module):
            from whisper.model import LayerNorm
            for name, child in module.named_children():
                if isinstance(child, LayerNorm):
                    old_params = child.state_dict()
                    new_layer_norm = nn.LayerNorm(child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine)
                    new_layer_norm.load_state_dict(old_params)
                    setattr(module, name, new_layer_norm)
                else:
                    replace_layer_norm(child)

        import whisper
        encoder = whisper.load_model(name=model_config.speech_encoder, device='cpu').encoder
        replace_layer_norm(encoder)
        return encoder


if __name__ == "__main__":
    from types import SimpleNamespace

    model_config = SimpleNamespace(
        speech_encoder="large-v3",          # passed to whisper.load_model(name=...)
        speech_encoder_type="whisper",      # only needed if you use build_speech_encoder()
        speech_encoder_hidden_size=1280,    # Whisper large encoder output dim
        speech_encoder_ds_rate=5,           # used by speech projector downstream
    )

    encoder = WhisperWrappedEncoder.load(model_config)
    print(encoder)