# Hindi LLaMA-Omni Final Report

## Motivation

The goal of this project was to build a Hindi speech-to-speech assistant based on the LLaMA-Omni architecture. The motivation was to make the model accept spoken Hindi questions, understand them through a speech encoder and language model backbone, and eventually generate spoken Hindi responses. Since the original LLaMA-Omni release mainly targeted English speech interaction, this project focused on adapting the model and training pipeline for Hindi.

The work began with a practical gap: there was no ready-to-use Hindi instruction dataset where the question was provided as audio and the answer was provided as text. This made dataset creation, training pipeline reconstruction, and Hindi-specific fine-tuning necessary before speech-to-speech generation could be attempted reliably.

## 1. Original Model

### LLaMA-Omni Architecture

![LLaMA-Omni architecture](evals_images/image.png)

The original LLaMA-Omni architecture combines a speech encoder, a speech adaptor/projector, a large language model, a speech decoder, and a vocoder. The user speech is first converted into audio features, then those features are projected into the same embedding space used by the language model. The language model can then reason over speech-conditioned inputs while the downstream speech decoder and vocoder support spoken output generation.

In this project, the main focus was the speech understanding path:

- audio input
- Whisper speech encoder
- speech projector/adaptor
- LLaMA language model backbone
- text response generation

The base language model used in this work was the LLaMA 3.1 8B backbone.

### Whisper Speech Encoder

The speech encoder uses the encoder part of Whisper. Only the Whisper encoder is used, not the full Whisper sequence-to-sequence model. The audio waveform is converted into a Whisper log-mel spectrogram, and the encoder transforms this into a sequence of speech features.

In the implementation, the Whisper encoder outputs hidden representations with size `1280`. The Whisper encoder also reduces the time dimension, so the downstream model receives a shorter sequence of speech features than the original mel sequence.

### Speech Adaptor / Projector

The speech projector is the bridge between Whisper and LLaMA. Whisper produces speech features in its own hidden space, while LLaMA expects token embeddings in the LLaMA hidden space. The projector adapts the speech features so they can be inserted into the LLaMA input embedding sequence.

The projector used here performs two jobs:

- it downsamples consecutive Whisper frames using the speech encoder downsampling rate
- it maps the concatenated Whisper features into the LLaMA hidden dimension

The implemented projector is a two-layer network:

- input: `speech_encoder_hidden_size * speech_encoder_ds_rate`
- hidden layer: `2048`
- output: LLaMA hidden size

After projection, the speech features replace the special speech token positions in the LLaMA input embeddings. The labels for those speech feature positions are masked with `IGNORE_INDEX`, so the model is trained to predict only the text answer tokens, not the inserted speech frames.

## 2. My Contribution

### Hindi Speech Instruction Dataset

The first contribution was creating a Hindi instruction dataset suitable for speech-conditioned training. The needed format was:

- question: Hindi audio
- answer: Hindi text

This was necessary because the available Hindi instruction datasets were text-only, while the speech model required spoken questions paired with textual answers.

Relevant Hugging Face dataset links:

- Hindi speech instruction dataset: [Pastaaaaa2003/Hindi-speech-instruct](https://huggingface.co/datasets/Pastaaaaa2003/Hindi-speech-instruct)
- Streaming parquet dataset used by the training config: [Pastaaaaa2003/hindi-llama-omni](https://huggingface.co/datasets/Pastaaaaa2003/hindi-llama-omni)

The text instruction sources included Hindi splits from `ai4bharat/indic-instruct-data-v0.1`, including `anudesh`, `flan_v2`, `hh-rlhf`, and `lm_sys`. These were normalized into a conversation format and then paired with generated Hindi speech for the user-side questions.

### Training Pipeline Reconstruction

The original model release did not include a complete training pipeline for this Hindi adaptation. I added the training pipeline needed to fine-tune the model, including:

- text-only Hindi backbone fine-tuning
- speech-conditioned training with the Whisper encoder and speech projector
- Hugging Face streaming dataset support
- validation handling for streaming parquet shards
- LoRA checkpoint saving/loading
- BLEU, perplexity, and Hindi benchmark evaluation scripts

This made it possible to train and evaluate the model end-to-end rather than only running inference from released weights.

### Replacing The English Instruction-Tuned Backbone

The original setup used an English instruction-tuned backbone. In my experiments, starting from those English-tuned weights was unstable for Hindi speech training. The gradients became very large and the loss curve was unstable.

To make the training more stable, I replaced the English instruction-tuned backbone with the base LLaMA 3.1 8B model and adapted it to Hindi through a first-stage text-only LoRA fine-tune. This produced a better Hindi initialization before adding the speech projector into training.

## 3. Training

### Why The Training Was Split Into Two Stages

The original model was trained in a single stage where the speech projector and language model backbone were trained together. I initially tried a similar approach, but it did not work well for the Hindi setting. The loss curve became unstable and the gradient norm showed signs of instability.

The final approach used two stages:

- Stage 1: fine-tune the LLaMA backbone on Hindi instruction text
- Stage 2: fine-tune the Hindi-adapted backbone and speech projector together on speech-text instruction data

This two-stage setup worked better because the language model first learned Hindi instruction-following behavior before it had to learn the harder speech-to-text alignment problem.

### Stage 1: Hindi Backbone Fine-Tuning

In the first stage, only the LLaMA backbone was fine-tuned with LoRA on Hindi instruction text. The speech projector and speech encoder were not trained in this stage.

LoRA and training configuration:

- base model: LLaMA 3.1 8B
- LoRA enabled: yes
- LoRA rank `r`: `128`
- LoRA alpha: `64`
- LoRA dropout: `0.05`
- batch size: `2`
- gradient accumulation steps: `7`
- learning rate: `0.000107467137359001`
- scheduler: cosine
- warmup ratio: `0.05`
- precision: `bf16-mixed`
- max gradient norm: `1.8`
- gradient checkpointing: enabled
- trained modules: LLaMA backbone LoRA weights
- frozen modules: speech encoder and speech projector

Stage 1 training curves:

![Stage 1 validation loss](evals_images/image_4.png)

![Stage 1 accumulated training loss](evals_images/image_3.png)

![Stage 1 global gradient norm](evals_images/image_2.png)

The Stage 1 curves show that the Hindi backbone fine-tuning became much more stable over time. The validation loss moved downward, the training loss decreased gradually, and the gradient norm converged from very high early values to a much more stable range.

### Stage 2: Speech Projector + Backbone Fine-Tuning

In the second stage, the Hindi-adapted backbone was used as the initialization checkpoint. The model was then trained on the speech instruction dataset with both the speech projector and LLaMA LoRA backbone trainable.

Stage 2 configuration:

- initialization checkpoint: Hindi backbone LoRA checkpoint
- input type: Whisper log-mel features
- mel size: `128`
- train samples: `105000`
- validation samples: `5720`
- LoRA rank `r`: `128`
- LoRA alpha: `64`
- LoRA dropout: `0.05`
- batch size: `2`
- gradient accumulation steps: `7`
- learning rate: `0.000107467137359001`
- scheduler: cosine
- warmup ratio: `0.05`
- weight decay: `0.01`
- precision: `bf16-mixed`
- max gradient norm: `1.8`
- trained modules: speech projector and LLaMA LoRA backbone
- frozen modules: Whisper speech encoder

The W&B run for this stage is available here: [speech-hf-streaming run qnxitjb5](https://wandb.ai/asthadu29-ludwig-maximilianuniversity-of-munich/hindi_llama_omni/runs/qnxitjb5?nw=nwuserasthadu29)

Stage 2 training curves:

![Stage 2 training loss](evals_images/image_8.png)

![Stage 2 validation loss](evals_images/image_7.png)

![Stage 2 global gradient norm](evals_images/image6.png)

![Stage 2 learning rate schedule](evals_images/image_5.png)

![Stage 2 LoRA and speech projector gradient norms](evals_images/image_1.png)

The Stage 2 results show that the two-stage strategy stabilized training. The training loss decreased, the validation loss improved until it reached a stable region, and the global gradient norm converged instead of exploding. The separate LoRA and speech projector gradient plots also show that the projector received strong updates early and then settled into a more controlled range.

## 4. Evaluation

### Speech Validation: BLEU And Perplexity

The speech validation evaluation was run on `200` validation samples from the streaming Hindi speech dataset.

Fine-tuned streaming model results:

- evaluated samples: `200`
- BLEU: `0.0489`
- perplexity: `3.4350`
- mean negative log-likelihood: `1.2340`
- total label tokens: `29713`

Base vs fine-tuned speech validation comparison:

- `default_omni`: BLEU `0.0289`, perplexity `7.658`
- `streaming_llama`: BLEU `0.0489`, perplexity `3.4350`

### Hindi Text Benchmarking

For benchmarking the Hindi language backbone, I used two compact Hindi evaluation datasets:

- `mteb/IndicSentiment`, Hindi test split
- `AdaMLLab/indicxnli_repaired`, Hindi validation split

Each task was evaluated on `200` samples. The comparison was between the base LLaMA model and the Hindi fine-tuned LoRA backbone.

Base model results:

- IndicSentiment Hindi accuracy: `0.895`
- IndicXNLI Hindi accuracy: `0.330`
- macro accuracy: `0.6125`
- invalid rate: `0.0`

After Hindi backbone fine-tuning:

- IndicSentiment Hindi accuracy: `0.920`
- IndicXNLI Hindi accuracy: `0.745`
- macro accuracy: `0.8325`
- invalid rate: `0.0`

Improvement:

- IndicSentiment: `+0.025`
- IndicXNLI: `+0.415`
- macro accuracy: `+0.220`

The largest improvement came from IndicXNLI, where the model moved from near-random performance to much stronger Hindi natural language inference performance. This supports the decision to adapt the backbone to Hindi before speech projector training.

## 5. Summary

This project adapted LLaMA-Omni toward Hindi speech interaction by creating the missing Hindi speech instruction data, reconstructing the training pipeline, replacing the unstable English instruction-tuned initialization with a base LLaMA 3.1 8B backbone, and introducing a two-stage training recipe.

The key finding was that direct single-stage speech training was unstable for this Hindi setup. A first-stage Hindi text LoRA fine-tune produced a better backbone, and the second-stage speech projector plus backbone fine-tune showed stable loss and gradient behavior. The Hindi backbone benchmark improved from `0.6125` macro accuracy to `0.8325`, showing that the model became substantially better at Hindi instruction and reasoning tasks before being used for speech-conditioned training.
