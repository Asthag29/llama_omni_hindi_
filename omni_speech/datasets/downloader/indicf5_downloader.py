from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="ai4bharat/IndicF5",
    local_dir="models/indicf5",
)
