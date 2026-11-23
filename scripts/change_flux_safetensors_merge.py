import argparse
import json
import os
import subprocess
from typing import List

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file


class ModelMerger:
    def __init__(self, args):
        self.model_id = args.model_id
        self.cache_dir = args.cache_dir
        self.output_dir = args.output_dir
        self.subfolder = args.subfolder
        self.force_download = args.force_download
        self.revision = args.revision

    def download_model_files(self) -> List[str]:
        """Download all necessary model files from Hugging Face"""
        print(f"Downloading model files from {self.model_id}")

        os.makedirs(self.cache_dir, exist_ok=True)

        files = [
            f"{self.subfolder}/diffusion_pytorch_model-00001-of-00003.safetensors",
            f"{self.subfolder}/diffusion_pytorch_model-00002-of-00003.safetensors",
            f"{self.subfolder}/diffusion_pytorch_model-00003-of-00003.safetensors",
            f"{self.subfolder}/diffusion_pytorch_model.safetensors.index.json"
        ]

        downloaded_files = []
        for file in files:
            print(f"Downloading {file}...")
            try:
                local_file = self._download_single_file(file)
                downloaded_files.append(local_file)
                print(f"Successfully downloaded to: {local_file}")
            except Exception as e:
                print(f"Error downloading {file}: {str(e)}")
                if "401" in str(e):
                    self._handle_auth_error(file, downloaded_files)
                else:
                    raise e

        return downloaded_files

    def _download_single_file(self, filename: str) -> str:
        """Download a single file from the repository"""
        return hf_hub_download(
            repo_id=self.model_id,
            filename=filename,
            revision=self.revision,
            cache_dir=self.cache_dir,
            force_download=self.force_download
        )

    def _handle_auth_error(self, filename: str, downloaded_files: List[str]):
        """Handle authentication errors by prompting login"""
        print("Authentication required. Please login to Hugging Face:")
        subprocess.run(["huggingface-cli", "login"])
        local_file = self._download_single_file(filename)
        downloaded_files.append(local_file)

    def merge_safetensors(self, input_files: List[str], output_file: str, index_file: str):
        """Merge split safetensor files into one"""
        print("Merging safetensor files...")

        with open(index_file, 'r') as f:
            json.load(f)

        merged_tensors = {}

        for input_file in sorted(input_files):
            print(f"Loading tensors from {input_file}")
            tensors = load_file(input_file)
            merged_tensors.update(tensors)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        print(f"Saving merged model to {output_file}")
        save_file(merged_tensors, output_file)
        print("Merge complete!")

    def process(self):
        """Main process to download and merge model files"""
        downloaded_files = self.download_model_files()

        safetensor_files = [f for f in downloaded_files if f.endswith('.safetensors') and 'of-00003' in f]
        index_file = [f for f in downloaded_files if f.endswith('.index.json')][0]

        print("\nDownloaded files:")
        for file in downloaded_files:
            print(f"- {file}")

        output_file = os.path.join(self.output_dir, "diffusion_pytorch_model.safetensors")

        self.merge_safetensors(safetensor_files, output_file, index_file)
        print(f"\nMerged model saved to: {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Merge split safetensor model files")

    parser.add_argument(
        "--model_id",
        type=str,
        default="black-forest-labs/FLUX.1-schnell",
        help="Hugging Face model ID"
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./model_cache",
        help="Directory to cache downloaded files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./merged_model",
        help="Directory to save merged model"
    )

    parser.add_argument(
        "--subfolder",
        type=str,
        default="transformer",
        help="Subfolder in the repository containing model files"
    )

    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force download even if files exist in cache"
    )

    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Repository revision to download from"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    merger = ModelMerger(args)
    merger.process()

if __name__ == "__main__":
    main()

"""
# Basic usage with defaults
python merge_model.py

# Specify custom paths and model
python change_flux_safetensors_merge.py \
    --model_id "black-forest-labs/FLUX.1-schnell" \
    --cache_dir "./custom_cache" \
    --output_dir "./custom_output" \
    --subfolder "transformer" \
    --force_download

# Use a specific revision
python change_flux_safetensors_merge.py --revision "main" --force_download
"""
