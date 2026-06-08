# Adopted from https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/projector.py
#* nothing just a linear projection layer to match the embedding size of the llama model

import torch
import torch.nn as nn


class EncoderProjectorConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.speech_encoder_ds_rate #* downsampling rate of the speech encoder
        self.encoder_dim = config.speech_encoder_hidden_size
        self.llm_dim = config.hidden_size
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.hidden_size)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k #* discard the remaining frames
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :] #* taking all the elements of the batch , all the elements except the num_frames_to_discard, everything else remains the same
        seq_len = x.size(1)
        
        x = x.contiguous() #* instead of allocating new memory, we are just rearranging the memory , but with contiguous we are allocating new memory, without this some operations cannot be performed
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x