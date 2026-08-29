---
language:
- hi
license: other
base_model: ICTNLP/Llama-3.1-8B-Omni
library_name: peft
tags:
- speech-to-speech
- hindi
- llama-omni
---

# 🦙🎧 Hindi LLaMA-Omni: Hindi Speech Interaction with Large Language Models

> Hindi LLaMA-Omni is a Hindi speech-to-speech model built upon [LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni). It uses Whisper for speech understanding, a fine-tuned Hindi LLaMA-Omni backbone for response generation, and IndicF5 to synthesize answers in one fixed default Hindi voice.

[![Model](https://img.shields.io/badge/🤗%20Model-hindi--llama--omni--model-yellow)](https://huggingface.co/Pastaaaaa2003/hindi-llama-omni-model)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Hindi--speech--instruct-blue)](https://huggingface.co/datasets/Pastaaaaa2003/Hindi-speech-instruct)
[![Base](https://img.shields.io/badge/Base-LLaMA--Omni-green)](https://github.com/ictnlp/LLaMA-Omni)
[![TTS](https://img.shields.io/badge/TTS-IndicF5-orange)](https://github.com/AI4Bharat/IndicF5)

![Hindi LLaMA-Omni architecture](images/image.png)



- 🗣️ **Hindi speech interaction:** accepts Hindi speech questions and returns Hindi speech answers.
- 🧠 **Built on LLaMA-Omni:** keeps the Whisper encoder and speech projector structure from the original speech-language architecture.
- 🇮🇳 **Fine-tuned for Hindi instruction following:** trained with Hindi instruction data converted into speech-question form.
- 🎙️ **IndicF5 speech generation:** replaces the original unit-vocoder output path with a fixed default Hindi voice.
- 📊 **Evaluated in Hindi:** includes IndicQA, MT-Bench-Hi, IFEval-Hi, and GSM8K-Hi evaluation scripts.

## 💡 Architecture

The runtime flow is:

1. A user records or uploads a Hindi speech question.
2. Whisper encodes the speech input.
3. The speech projector maps audio features into the LLaMA-Omni language backbone.
4. The fine-tuned Hindi backbone generates a Hindi text response.
5. A bundled reference recording provides the fixed speaker identity for IndicF5.
6. IndicF5 synthesizes the final answer in the fixed default voice.

## 📚 Data

This project uses Hindi instruction-following examples converted into speech-question form. Since public datasets with Hindi speech questions paired with text answers do not exist, the training data was generated from Hindi instruction corpora and converted into audio for speech-instruction fine-tuning.

Dataset card: [`Pastaaaaa2003/Hindi-speech-instruct`](https://huggingface.co/datasets/Pastaaaaa2003/Hindi-speech-instruct)

The text instruction mixture includes AI4Bharat Indic-Instruct style sources such as `anudesh`, `flan_v2`, `hh-rlhf`, and `lm_sys`.

## ⚖️ License and attribution

The Hindi stage-2 adapter is published at
[`Pastaaaaa2003/hindi-llama-omni-model`](https://huggingface.co/Pastaaaaa2003/hindi-llama-omni-model).
This is a thin release: this repository contains code only, and the model
repository contains only the final stage-2 adapter. The other checkpoints are
downloaded from their upstream owners. Follow their respective terms,
including the academic, non-commercial restriction for
[LLaMA-Omni](https://huggingface.co/ICTNLP/Llama-3.1-8B-Omni), and cite the
original [LLaMA-Omni paper](https://arxiv.org/abs/2409.06666).

## 🛠️ Install

The supported runtime is Linux with an NVIDIA GPU (about 24 GB VRAM) and
CUDA 12.1. Keep about 40 GB free disk space for the environment and
checkpoints. Apple Silicon is not currently a supported runtime for the
Gradio server.

#### 1. Clone this repository and create an environment

Create and activate any Python 3.11.14 environment. The example below uses
the standard-library `venv`:

```bash
git clone https://github.com/Asthag29/llama_omni_hindi_.git
cd llama_omni_hindi_
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

#### 2. Install dependencies

Install the CUDA-enabled PyTorch build appropriate for your system, then
install the project packages:

```bash
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "f5_tts @ git+https://github.com/AI4Bharat/IndicF5.git@13f7c4d627cc10111aea8fe9c0039462cacacdc7"
```

## ⚡ Download model checkpoints

[IndicF5](https://huggingface.co/ai4bharat/IndicF5) is a gated repository.
Request access on that page first, then download it:

```bash
hf download ai4bharat/IndicF5 \
  --revision ba85abedf18dc479a447eaa0eccbd76ab78a47d5 \
  --local-dir models/indicf5
```

Download the remaining public checkpoints with:

```bash
python omni_speech/datasets/downloader/download_models.py
```

The script fetches the pinned LLaMA-Omni weights, Whisper large-v3, and the
Hindi stage-2 adapter into `models/`, and verifies the Whisper file. Run it
again if a download is interrupted.

Only the final stage-2 adapter is published by this project. Stage-1
checkpoints are training artifacts and are not required for inference.

The final layout must be:

```text
models/
├── llama/            # LLaMA-Omni base model
├── speech_encoder/   # Whisper large-v3 weights
├── indicf5/          # IndicF5 model snapshot
└── hindi_ckpt/
    └── stage_2/      # Hindi stage-2 inference checkpoint
        └── speech_text/
            └── checkpoints/
                └── last/  # adapter and speech projector
```

## 🎧 Run the Gradio demo

Start each command in a separate terminal. Activate the environment and run
the commands from the repository root.

**Terminal 1 — controller**

```bash
source .venv/bin/activate
python -m omni_speech.serve.controller --host 127.0.0.1 --port 21001
```

**Terminal 2 — Hindi model worker**

```bash
source .venv/bin/activate
python -m omni_speech.serve.model_worker \
  --host 127.0.0.1 \
  --port 21002 \
  --worker-address http://127.0.0.1:21002 \
  --controller-address http://127.0.0.1:21001 \
  --model-path models/llama \
  --model-name llama-omni-hindi \
  --checkpoint models/hindi_ckpt/stage_2/speech_text/checkpoints/last \
  --config configs/stage_2.yaml \
  --device cuda
```

Wait until the worker reports that it has registered with the controller.

**Terminal 3 — web interface**

```bash
source .venv/bin/activate
python -m omni_speech.serve.gradio_web_server \
  --host 127.0.0.1 \
  --port 7860 \
  --controller-url http://127.0.0.1:21001 \
  --indicf5-model-path models/indicf5
```

Open <http://127.0.0.1:7860/> and record or upload a Hindi speech question.
The demo returns a Hindi audio answer using `data/inference.wav` as the fixed
IndicF5 reference voice. Its fixed reference transcript is `तुम कौन हो`;
Whisper is not used to transcribe reference audio.

## 🧪 Speech-input text-response test

This command tests the speech-understanding and Hindi-text generation stages
without starting the web server. It prints the Hindi response in the terminal;
use the Gradio demo above for the complete speech-to-speech response.

```bash
python -m omni_speech.infer.inference \
  --audio path/to/hindi-question.wav \
  --checkpoint models/hindi_ckpt/stage_2/speech_text/checkpoints/last
```

The tracked `data/inference.wav` file can be used as the default test audio.

## 📊 Evaluation

Evaluation scripts live in `evaluations/`, and the full results discussion is maintained in [`evaluations/results/summary.md`](evaluations/results/summary.md).

Run the evaluations from an activated environment:

```bash
python evaluations/indicQA.py
python evaluations/mt_bench_hi.py
python evaluations/if_eval_hi.py
python evaluations/gsm8k_hi.py
```

The scripts write result files under `evaluations/results/`.

### Summary

Across the available evaluations, the fine-tuned model improves Hindi QA, answer overlap, semantic similarity, extraction, STEM, humanities, writing, and instruction-following metrics. These gains align with the Hindi instruction data used for training, which emphasizes direct question answering, transformation, extraction, formatting, and concise assistant responses.


## 🙏 Acknowledgements

- [LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni): base speech-language architecture and code structure.
- [Whisper](https://github.com/openai/whisper): speech encoder for spoken input.
- [IndicF5](https://github.com/AI4Bharat/IndicF5): Hindi/Indic speech-synthesis backend.
- [AI4Bharat](https://ai4bharat.iitm.ac.in/): Indic instruction and speech resources.



