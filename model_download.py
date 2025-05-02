from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="CompVis/stable-diffusion-v1-4",
    token=True,
    local_dir="./models/stable-diffusion-v1-4",
)