from huggingface_hub import snapshot_download

local_dir = "/nas/common_data/huggingface/dreambooth/dog"
snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)