from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="mamoeed/apebench-pde-simulation-data",
    local_dir="./datasets/",
)