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


#! will print the model architecture
#* AudioEncoder(
#   (conv1): Conv1d(128, 1280, kernel_size=(3,), stride=(1,), padding=(1,))
#   (conv2): Conv1d(1280, 1280, kernel_size=(3,), stride=(2,), padding=(1,))
#   (blocks): ModuleList(
#     (0-31): 32 x ResidualAttentionBlock(
#       (attn): MultiHeadAttention(
#         (query): Linear(in_features=1280, out_features=1280, bias=True)
#         (key): Linear(in_features=1280, out_features=1280, bias=False)
#         (value): Linear(in_features=1280, out_features=1280, bias=True)
#         (out): Linear(in_features=1280, out_features=1280, bias=True)
#       )
#       (attn_ln): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#       (mlp): Sequential(
#         (0): Linear(in_features=1280, out_features=5120, bias=True)
#         (1): GELU(approximate='none')
#         (2): Linear(in_features=5120, out_features=1280, bias=True)
#       )
#       (mlp_ln): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#     )
#   )
#   (ln_post): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
# )