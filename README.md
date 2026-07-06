llama omni model for hindi language

## Model Downloads

The training and inference configs expect the LLaMA-Omni checkpoint at `models/llama`.
That checkpoint config uses the Whisper encoder `large-v3`.

Install the Hugging Face CLI if needed:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s
hf auth login
```

Download the LLaMA-Omni model used by this repo:

```bash
mkdir -p models/llama
hf download ICTNLP/Llama-3.1-8B-Omni \
  --local-dir models/llama \
  --type model
```

Download or pre-cache the Whisper Large v3 encoder weights:

```bash
python - <<'PY'
import whisper

whisper.load_model("large-v3", download_root="models/speech_encoder")
PY
```

The repo only uses the Whisper encoder part. The active config value is stored in
`models/llama/config.json` as `speech_encoder: "large-v3"`.