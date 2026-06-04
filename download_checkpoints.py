from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="aryan100ddede/DermaAI",
    repo_type="model",
    local_dir="./checkpoints"
)
