# 🦙🎧 Hindi LLaMA-Omni: Hindi Speech Interaction with Large Language Models

> Hindi LLaMA-Omni is a Hindi speech-to-speech model built upon [LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni). It uses Whisper for speech understanding, a LoRA-fine-tuned Hindi LLaMA-Omni backbone for response generation, and IndicF5 to synthesize answers in one fixed default Hindi voice.

[![Model](https://img.shields.io/badge/🤗%20Model-hindi--llama--omni--model-yellow)](https://huggingface.co/Pastaaaaa2003/hindi-llama-omni-model)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Hindi--speech--instruct-blue)](https://huggingface.co/datasets/Pastaaaaa2003/Hindi-speech-instruct)
[![Base](https://img.shields.io/badge/Base-LLaMA--Omni-green)](https://github.com/ictnlp/LLaMA-Omni)
[![TTS](https://img.shields.io/badge/TTS-IndicF5-orange)](https://github.com/AI4Bharat/IndicF5)

![Hindi LLaMA-Omni architecture](images/image.png)

- 🗣️ **Hindi speech interaction:** accepts Hindi speech questions and returns Hindi speech answers.
- 🧠 **Built on LLaMA-Omni:** keeps the Whisper encoder and speech projector structure from the original speech-language architecture.
- 🇮🇳 **Fine-tuned for Hindi instruction following:** two-stage LoRA training on Hindi instruction data converted into speech-question form.
- 🎙️ **IndicF5 speech generation:** replaces the original unit-vocoder output path with a fixed default Hindi voice.
- 📊 **Evaluated in Hindi:** IndicQA, MT-Bench-Hi, IFEval-Hi, and GSM8K-Hi scripts with published results.

## 💡 Architecture

The runtime flow is:

1. A user records or uploads a Hindi speech question.
2. Whisper large-v3 encodes the speech input (frozen).
3. The speech projector maps audio features into the LLaMA-Omni language backbone.
4. The Hindi-fine-tuned backbone (LoRA adapter) generates a Hindi text response.
5. A bundled reference recording provides the fixed speaker identity for IndicF5.
6. IndicF5 synthesizes the final answer in the fixed default voice.

## 🗂️ Repository layout

```text
omni_speech/
├── model/            # LLaMA-Omni architecture: speech encoder, projector, LLM
├── tts/              # IndicF5 speech-synthesis wrapper
├── training/         # stage1.py, stage2.py, speech_module.py (Hydra + PyTorch Lightning)
├── datasets/         # text-data downloader, preprocessing, splits, local speech-data builder
├── infer/            # inference.py: speech in, Hindi text out
└── serve/            # controller, model_worker, gradio_web_server
configs/              # stage_1.yaml, stage_2.yaml
evaluations/          # benchmark scripts and results/summary.md
tests/                # pytest suite
pyproject.toml        # dependencies (requirements.txt just installs the project)
check_models.py       # verifies every checkpoint file is present
data/inference.wav    # IndicF5 reference voice; also the default test question
data/splits/          # validation and test ids shared by both training stages
```

## 🛠️ Install

The supported runtime is Linux with an NVIDIA GPU (about 24 GB VRAM for
inference) and CUDA 12.1. Keep about 40 GB free disk space for the environment
and checkpoints. Apple Silicon is not a supported runtime for the Gradio server.

Python 3.11 is required. Dependencies are declared in `pyproject.toml`, with
two extras: `eval` (benchmark scripts) and `test` (pytest).

```bash
git clone https://github.com/Asthag29/llama_omni_hindi_.git
cd llama_omni_hindi_
```

**With [uv](https://docs.astral.sh/uv/)** (recommended; installs the exact
versions in `uv.lock`, including the CUDA 12.1 PyTorch build):

```bash
uv sync --extra eval --extra test
source .venv/bin/activate
```

**With pip**, install the CUDA 12.1 PyTorch build first, then the project:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.1.2+cu121 torchaudio==2.1.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt    # same as: pip install -e ".[eval,test]"
```

Both install the project in editable mode, which the Hydra configs and example
audio rely on. `transformers` is pinned to `4.43.4` because LLaMA-Omni's model
code depends on it. Do not bump it; the "IndicF5 weights" note below explains
how the IndicF5 loader copes with that version.

## ⚡ Download model checkpoints

1. Download [`ICTNLP/Llama-3.1-8B-Omni`](https://huggingface.co/ICTNLP/Llama-3.1-8B-Omni) into `models/llama/`:

```bash
hf download ICTNLP/Llama-3.1-8B-Omni --local-dir models/llama
```

2. Download Whisper large-v3:

```python
import whisper
whisper.load_model("large-v3", download_root="models/speech_encoder/")
```

3. [IndicF5](https://huggingface.co/ai4bharat/IndicF5) is a gated repository.
   Request access on that page first, then download it:

```bash
hf download ai4bharat/IndicF5 --local-dir models/indicf5
```

4. Download this project's Hindi stage-2 adapter:

```bash
hf download Pastaaaaa2003/hindi-llama-omni-model --local-dir models/hindi
```

Then run the checkpoint check from the repository root. It confirms every
required file is present and that Whisper matches the official SHA-256:

```bash
python check_models.py
```

Do not start Gradio or `inference.py` until this check succeeds. The final
layout is:

```text
models/
├── llama/            # LLaMA-Omni base model (config, tokenizer, 4 safetensors shards)
├── speech_encoder/   # large-v3.pt
├── indicf5/          # IndicF5 snapshot (config.json, model.py, model.safetensors, checkpoints/vocab.txt)
└── hindi/            # adapter_config.json, adapter_model.safetensors, speech_projector.safetensors
```

Only the final stage-2 adapter is published. Stage-1 checkpoints are training
artifacts and are not required for inference.

#### IndicF5 weights

IndicF5 names its GRN and Vocos layer-scale parameters `gamma` / `beta`. The
pinned `transformers==4.43.4` rewrites those substrings to `weight` / `bias`
inside `from_pretrained`, so 16 tensors silently miss the model and keep their
random init. The loader therefore builds the model from its config and loads
`model.safetensors` with `load_state_dict(strict=True)` instead; every
checkpoint tensor lands, and `tests/test_indicf5.py` fails if that ever
regresses.

## 🎧 Run the Gradio demo

Start each command in a separate terminal from the repository root with the
environment activated.

**Terminal 1 — controller**

```bash
python -m omni_speech.serve.controller --host 127.0.0.1 --port 21001
```

**Terminal 2 — Hindi model worker**

```bash
python -m omni_speech.serve.model_worker \
  --host 127.0.0.1 \
  --port 21002 \
  --worker-address http://127.0.0.1:21002 \
  --controller-address http://127.0.0.1:21001 \
  --model-name llama-omni-hindi \
  --checkpoint models/hindi \
  --config configs/stage_2.yaml \
  --device cuda
```

Wait until the worker reports that it has registered with the controller. The
base model location (`models/llama`) is read from the config.

**Terminal 3 — web interface**

```bash
python -m omni_speech.serve.gradio_web_server \
  --host 127.0.0.1 \
  --port 7860 \
  --controller-url http://127.0.0.1:21001 \
  --indicf5-model-path models/indicf5
```

Open <http://127.0.0.1:7860/> and record or upload a Hindi speech question.
The demo returns a Hindi audio answer using `data/inference.wav` as the fixed
IndicF5 reference voice with its fixed transcript (the first lines of the poem
*तितली रानी*, set in `DEFAULT_REFERENCE_TEXT` in `gradio_web_server.py`);
Whisper is not used to transcribe reference audio. Add `--share` for a public
Gradio link, or `--indicf5-device cpu` to keep IndicF5 off the GPU.

## 🧪 Speech-input text-response test

This runs the speech-understanding and Hindi-text stages without the web
server and prints the Hindi response in the terminal. Use the Gradio demo for
the full speech-to-speech path.

```bash
python -m omni_speech.infer.inference \
  --audio path/to/hindi-question.wav \
  --checkpoint models/hindi
```

With no `--audio`, it uses the tracked `data/inference.wav`. Generation is
greedy by default; `--temperature`, `--top-p`, `--num-beams`, and
`--max-new-tokens` (default 256) are available.

## 🏋️ Training

Training is split into two stages because training the speech projector and
backbone together from the English instruction-tuned weights was unstable for
Hindi (large gradients, diverging loss). Stage 1 first adapts the backbone to
Hindi on text only; stage 2 then adds the speech projector.

| | Stage 1 | Stage 2 |
| --- | --- | --- |
| Entry point | `omni_speech.training.stage1` | `omni_speech.training.stage2` |
| Config | `configs/stage_1.yaml` | `configs/stage_2.yaml` |
| Trainable | LLM LoRA | LLM LoRA + speech projector |
| Frozen | Whisper, speech projector | Whisper |
| Input | Hindi text conversations (local JSON) | Hindi speech clips of at most 30 s with text answers (local parquet) |
| Init | `models/llama` | stage-1 `best_model` |
| Output | `outputs/stage_1/backbone_text/` | `outputs/stage_2/speech_text/` |

Shared hyperparameters (all in the YAML configs): LoRA `r=128`, `alpha=64`,
`dropout=0.05` on all attention and MLP projections; learning rate `1e-4`
with 5% warmup and cosine decay; batch size 2 × 7 gradient-accumulation
steps; 3 epochs; gradient checkpointing; gradient clipping at 1.8.

Precision is `bf16-mixed`: the frozen base model and the forward pass are
bf16, while the trainable LoRA and projector weights and the optimizer state
are kept in fp32 so that small updates are not rounded away. Training checks
this at startup and stops if the trainable weights are not fp32, so do not
use `bf16-true` or `16-true`. Each stage runs on one GPU and peaks at about
29 GB of GPU memory at the maximum sequence length of 2,048 tokens.

Each run writes to its output directory:

- `checkpoints/best`: the best weights so far by validation loss, present from
  the first validation, so a job that is killed still leaves usable weights.
- `best_model/` and `final_model/` once training completes: the best and the
  last weights. Each holds `adapter_config.json`, `adapter_model.safetensors`,
  `speech_projector.safetensors`, and `checkpoint_meta.json` (about 1.4 GB).
- `logs/train.log` and `csv/metrics.csv`: training loss averaged over the
  optimizer steps since the previous row, validation loss, and learning rate,
  written at every validation and every 500 steps in between.

Both stages log to Weights & Biases by default (`logging.wandb: true`,
project `hindi_llama_omni`). Run `wandb login` first or pass
`logging.wandb=false` to keep only the local logs. Any config key can be
overridden on the command line with Hydra syntax, for example
`logging.output_dir=outputs/stage_1/my_run`.

#### Stage 1: Hindi backbone on text

Stage 1 reads `data/instruct/hindi_instruct_conversations.json`, a JSON array
of `{"id": ..., "conversations": [{"from": "human", ...}, {"from": "gpt", ...}]}`
entries. To rebuild it from the
[`ai4bharat/indic-instruct-data-v0.1`](https://huggingface.co/datasets/ai4bharat/indic-instruct-data-v0.1)
Hindi splits (`anudesh`, `flan_v2`, `hh-rlhf`, `lm_sys`):

```bash
python -m omni_speech.datasets.downloader.hindi_text_downloader   # writes data/<split>_dataset.json; --splits picks a subset
# merge the splits you want into data/instruct/hindi_instruct.json, then:
python -m omni_speech.datasets.processing.format_hindi_instruct   # -> data/instruct/hindi_instruct_conversations.json
```

The data is split 80/10/10 into train, validation, and test within each of
the four sources (seed 42). The held-out ids are tracked in
`data/splits/validation_ids.txt` and `data/splits/test_ids.txt`; every other
sample is train. For the 102,055-sample file used here that is 81,643 /
10,206 / 10,206. Stage 1 prints the split per source at startup and stops if
the id lists are missing. To regenerate them after changing the data:

```bash
python -m omni_speech.datasets.processing.split_hindi_instruct          # rewrites data/splits/*.txt
python -m omni_speech.datasets.processing.split_hindi_instruct --write-json   # also train/validation/test .json files
```

Then train:

```bash
python -m omni_speech.training.stage1
```

The test split is never used during training; it is available through the
data module's `test_dataloader()` for a final `trainer.test` run.

#### Stage 2: speech projector + backbone

Whisper hears only the first 30 seconds of a clip, and in the
[`Pastaaaaa2003/Hindi-speech-instruct`](https://huggingface.co/datasets/Pastaaaaa2003/Hindi-speech-instruct)
dataset about half of the spoken questions are longer than that. Stage 2
therefore trains on a local copy that keeps only clips of at most 30 s whose
question is also in the stage-1 file, split with the same ids as stage 1.
Build it once:

```bash
python -m omni_speech.datasets.processing.build_stage2_local   # -> data/speech/
```

The job downloads one source file at a time into `/tmp`, keeps the usable
rows, and deletes the source file, so only the result (about 9 GB) lands on
disk. It uses about 0.5 GB of memory, takes roughly 20–40 minutes, and resumes
if interrupted. For the data used here it keeps 49,787 clips (160 hours):

| Source | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| LMSYS | 19,790 | 2,525 | 2,505 |
| Flan v2 | 12,949 | 1,618 | 1,616 |
| Anudesh | 5,810 | 725 | 730 |
| HH-RLHF | 1,218 | 151 | 150 |
| **Total** | **39,767** | **5,019** | **5,001** |

Then train:

```bash
python -m omni_speech.training.stage2
```

Stage 2 reads `data/speech/` (`data.speech_dir`), takes its sample counts and
learning-rate schedule from `data/speech/manifest.json` (8,523 optimizer steps
for the data above), and stops with a clear message if the directory has not
been built. It starts from stage 1's `best_model` (`model.init_checkpoint`;
pass the path if stage 1 used a different output directory) and uses the same
instruction prompt as inference, `DEFAULT_SPEECH_PROMPT` in
`omni_speech/constants.py`. Mel features (128 bins) are computed on the fly,
and every clip is read exactly once per pass.

The held-out validation and test clips are the same questions that stage 1
held out, so they are unseen in both stages. As in stage 1, the test split is
available through `test_dataloader()` and is never used during training.

Copy the result to `models/hindi/` to serve it:

```bash
cp -rL outputs/stage_2/speech_text/best_model models/hindi
```

## 📊 Evaluation

The scripts in `evaluations/` benchmark the Hindi **text** backbone: the base
LLaMA-Omni model (`--model base`), the fine-tuned adapter
(`--model finetuned`), or both (`--model both`, the default where available).
They need the `eval` extra and load the published adapter in `models/hindi`
unless `--checkpoint` points elsewhere. Run them from the repository root:

```bash
python evaluations/indic_qa.py
python evaluations/mt_bench_hi.py
python evaluations/gsm8k_hi.py
python evaluations/if_eval_hi.py
```

Results are written under `evaluations/results/`. `--limit N` runs on a
subset, and IndicQA accepts `--skip-bertscore` to avoid loading the BERTScore
model.

The full results discussion is in
[`evaluations/results/summary.md`](evaluations/results/summary.md). Headline
numbers (base → fine-tuned):

| Benchmark | Metric | Base | Fine-tuned |
| --- | --- | ---: | ---: |
| IndicQA | Perplexity | 13.22 | **7.75** |
| IndicQA | F1 | 34.14% | **68.43%** |
| IndicQA | BERTScore F1 | 75.13% | **86.39%** |
| MT-Bench-Hi | Overall avg (LLM judge, 200 prompts) | 3.52 | **4.59** |
| IFEval-Hi | Prompt strict accuracy | 19.58% | **24.41%** |
| IFEval-Hi | Instruction normalized accuracy | 30.37% | **36.95%** |

The fine-tuned model improves QA, extraction, humanities, writing, and
instruction following. On MT-Bench-Hi the base model remains stronger on
coding, math, reasoning, and roleplay, which are under-represented in the
Hindi instruction data.

## ✅ Tests

```bash
pytest
```

`tests/test_indicf5.py` loads the real IndicF5 checkpoint and is skipped
automatically if `models/indicf5/` has not been downloaded.

## 📚 Data

Public datasets with Hindi speech questions paired with text answers do not
exist, so the training data was generated from Hindi instruction corpora and
converted into audio for speech-instruction fine-tuning. The text mixture is
the Hindi split of AI4Bharat Indic-Instruct (`anudesh`, `flan_v2`, `hh-rlhf`,
`lm_sys`).

Dataset card: [`Pastaaaaa2003/Hindi-speech-instruct`](https://huggingface.co/datasets/Pastaaaaa2003/Hindi-speech-instruct)

## ⚖️ License and attribution

The Hindi stage-2 adapter is published at
[`Pastaaaaa2003/hindi-llama-omni-model`](https://huggingface.co/Pastaaaaa2003/hindi-llama-omni-model).
This is a thin release: this repository contains code only, and the model
repository contains only the final stage-2 adapter. The other checkpoints are
downloaded from their upstream owners. Follow their respective terms,
including the academic, non-commercial restriction for
[LLaMA-Omni](https://huggingface.co/ICTNLP/Llama-3.1-8B-Omni), and cite the
original [LLaMA-Omni paper](https://arxiv.org/abs/2409.06666).

## 🙏 Acknowledgements

- [LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni): base speech-language architecture and code structure.
- [Whisper](https://github.com/openai/whisper): speech encoder for spoken input.
- [IndicF5](https://github.com/AI4Bharat/IndicF5): Hindi/Indic speech-synthesis backend.
- [AI4Bharat](https://ai4bharat.iitm.ac.in/): Indic instruction and speech resources.
