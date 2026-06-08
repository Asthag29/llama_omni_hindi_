# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

#! Training data pipeline: JSON conversations → Llama prompt strings → token IDs + masked labels.
#! Primary path for this repo: preprocess() → preprocess_llama_3() (default_conversation = conv_llama_3).
#! Inference only uses tokenizer_speech_token() from this file (see infer/infer.py).

import copy
import torch
import transformers
import tokenizers

from typing import Dict, Sequence

from omni_speech.constants import IGNORE_INDEX, DEFAULT_SPEECH_TOKEN
from omni_speech import conversation as conversation_lib
from omni_speech.model import *
from omni_speech.arguments import DataArguments
from omni_speech.constants import SPEECH_TOKEN_INDEX

from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

# # Model Constants
# IGNORE_INDEX = -100
# SPEECH_TOKEN_INDEX = -200
# DEFAULT_SPEECH_TOKEN = "<speech>"


# -----------------------------------------------------------------------------
#! tokenizer_speech_token — splits on <speech>, tokenizes text chunks, inserts SPEECH_TOKEN_INDEX (-200).
# -----------------------------------------------------------------------------
def tokenizer_speech_token(prompt, tokenizer, speech_token_index=SPEECH_TOKEN_INDEX, return_tensors=None):
    #* split prompt on <speech>, tokenize each text chunk → list of token-id lists
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<speech>')]

    #! [X, sep] -> [X, sep, X, sep, X, sep, ...]
    def insert_separator(X, sep):
        #* interleave chunks with separator: [chunk0, sep, chunk1, sep, chunk2, ...]
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]  #* drop trailing separator

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        #* appending the bos token to the input ids only if the first token is the bos token, for the consecutive speech tokens, we don't need to add the bos token again
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [speech_token_index] * (offset + 1)):
        input_ids.extend(x[offset:]) #* offset is taken from 0 not 1,don't get confused

    if return_tensors is not None:
        if return_tensors == 'pt': #* is it pytorch?
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

#! output example: [BOS, ..., -200, ..., -200, ...]  (-200 = speech placeholder)


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

#* just for the formatting of the data
    for source in sources:
        for sentence in source:
            if DEFAULT_SPEECH_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_SPEECH_TOKEN, '').strip()
                sentence['value'] = DEFAULT_SPEECH_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

    return sources



#! preprocess_llama_2 — Llama-2 chat format; only runs when default_conversation.sep_style == LLAMA_2.
## todo: can skip unless training with Llama-2 template (set default_conversation = conv_llama_2).
def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy() #* default conversation is llama_3
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())


    # Tokenize conversations

    if has_speech:
        input_ids = torch.stack([tokenizer_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_speech:
                round_len = len(tokenizer_speech_token(rou, tokenizer))
                instruction_len = len(tokenizer_speech_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


#! sources shape (one entry per training sample):
#!   sources = [
#!       [  # one conversation
#!           {"from": "human", "value": "<speech>\n..."},
#!           {"from": "gpt",   "value": "Hello, how are you?"}
#!       ],
#!       ...
#!   ]


# -----------------------------------------------------------------------------
#! preprocess_llama_3 — main training preprocess for Llama-3.1 / Omni (current default).
#! Returns input_ids (full prompt) + labels (assistant reply only; rest masked with IGNORE_INDEX).
# -----------------------------------------------------------------------------
def preprocess_llama_3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()  #* currently conv_llama_3
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}  #* map dataset keys → template roles (user, assistant)

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):  #* i = index of training sample in batch
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        assert len(source) == 2, "now only support single-turn conversation"  #* requires human + gpt turns

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"]) #* structure of the message is [[role, value], [role , value] ]etc etc 
        conversations.append(conv.get_prompt()) #* returns the final list of strings which could be understood by the llama model(consisting of start token, etc,e tc )
#*conversations = [
#   "<|begin_of_text|>...system...<|eot_id|>...user...<speech>\nTranscribe...<|eot_id|>...assistant...Hello...<|eot_id|>",
#   "<|begin_of_text|>...system...<|eot_id|>...user...<speech>\nAnother...<|eot_id|>...assistant...Reply...<|eot_id|>",
#   ...
# ]
    # Tokenize conversations

#! conversations example (one string per sample):
#!   "<|begin_of_text|>...system...<|eot_id|>...user...<speech>\n...<|eot_id|>...assistant...Hello...<|eot_id|>"

    # --- Step 2: tokenize ---
    if has_speech:
        input_ids = torch.stack([tokenizer_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0) #*[1, 128000, ..., -200, 13, 1234, ..., 5678, ...], special tokens like bos , etc are also tokenized into the ids
## todo: shape is [number of conversations, number of tokens], but differnt conversations have different number of tokens, so how to handle this?
#  ↑ normal tokens   ↑ -200 placeholder for speech
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length, #* max length of the tokens in the conversation
            truncation=True, #* truncate the conversation if it is longer than the max length
        ).input_ids

    targets = input_ids.clone() #! targets need to be generated from the manpulation of the input ids

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets
    sep = "<|start_header_id|>" + conv.roles[1] + "<|end_header_id|>\n\n"
    for conversation, target in zip(conversations, targets): #* ek conversation, ek target
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  #* count the total number of non-padded tokens in the target
 
 #! will be used for training the model
        #* ignore the first token in the target, it is the bos token
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        parts = conversation.split(sep) #* split the conversation into parts,using the gpt role as the separator
        parts[0] += sep #* part consiting of the human role and the message + assistant header

        if has_speech:
            conversation_len = len(tokenizer_speech_token(conversation, tokenizer))
            instruction_len = len(tokenizer_speech_token(parts[0], tokenizer)) - 1
        else:
            conversation_len = len(tokenizer(conversation).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

        target[cur_len : cur_len + instruction_len] = IGNORE_INDEX #* ignoring the human message + assistant header
        cur_len += conversation_len
        target[cur_len:] = IGNORE_INDEX #* ignoring the padding
        #! the taget only contains the assistant message without the assistant header
    
        # if cur_len < tokenizer.model_max_length:
        #     if cur_len != total_len:
        #         target[:] = IGNORE_INDEX
        #         print(
        #             f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
        #             f" (ignored)"
        #         )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_speech:
        input_ids = torch.stack([tokenizer_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_speech:
                round_len = len(tokenizer_speech_token(rou, tokenizer))
                instruction_len = len(tokenizer_speech_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # FIXME: tokenizer bug
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_SPEECH_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_SPEECH_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_speech_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


#! preprocess() — router; picks handler based on default_conversation template in conversation.py.
def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_speech=has_speech)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_speech=has_speech)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:
        return preprocess_llama_3(sources, tokenizer, has_speech=has_speech)
    raise NotImplementedError


if __name__ == "__main__":
    source = [
        {"from": "human", "value": "<speech>\nPlease directly answer..."},
        {"from": "gpt",   "value": "Hello, how are you?"}
    ]
    print("1. imports done")
    tokenizer = transformers.AutoTokenizer.from_pretrained("models/llama", use_fast=True)
    print("2. tokenizer loaded")
    result = print(preprocess([source], tokenizer, has_speech=True))
    print("3. preprocess done")
    print(result)
    print("cool it's working 🚀")
