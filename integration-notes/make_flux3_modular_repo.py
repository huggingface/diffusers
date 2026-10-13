"""Build the diffusers modular repo of FLUX.3 image-small from a BFL model package.

Converts the transformer and the image VAE, copies the text encoder and tokenizer as they are, writes the
`modular_model_index.json`, and (optionally) pushes the result to the Hub.

    .venv/bin/python integration-notes/make_flux3_modular_repo.py \
        --package /raid/yiyi/f3-image-small --output /raid/yiyi/flux3-image-modular-release \
        [--variant sd] [--push_to_hub diffusers-internal-dev/f3-image-diffusers]

`--variant` takes the DiT weights and config from `variants/<name>/` of the package (the shared encoders stay at the
root); a flow-map config (the step-distilled `sd` variant) gets `Flux3DistilledAutoBlocks`.

Scheduler, guider and image processor are `from_config` components created by the blocks (shift 6.93, guidance
3.5), so they are not stored in the repo.
"""

import argparse
import json
import os
import subprocess
import sys


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: E402

from diffusers import AutoencoderKLFlux2, Flux3Transformer2DModel  # noqa: E402
from diffusers.modular_pipelines import Flux3AutoBlocks, Flux3DistilledAutoBlocks  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument(
    "--package",
    required=True,
    help="the BFL package directory (config.json, model.safetensors, ae.safetensors, text_encoder/, variants/)",
)
parser.add_argument("--variant", default=None, help="package variant (variants/<name>/), e.g. sd")
parser.add_argument("--output", required=True)
parser.add_argument("--push_to_hub", default=None, help="repo id to push to (private)")
args = parser.parse_args()

python = sys.executable
dit_dir = os.path.join(args.package, "variants", args.variant) if args.variant else args.package
with open(os.path.join(dit_dir, "config.json")) as f:
    package_config = json.load(f)
blocks_class = Flux3DistilledAutoBlocks if package_config["dit"].get("flow_map") else Flux3AutoBlocks
# newer packages keep the tokenizer files in text_encoder/
tokenizer_dir = os.path.join(args.package, package_config["text_encoder"].get("tokenizer", "tokenizer"))

subprocess.run(
    [
        python,
        os.path.join(REPO, "scripts", "convert_flux3_to_diffusers.py"),
        "--checkpoint_path",
        os.path.join(dit_dir, "model.safetensors"),
        "--config_path",
        os.path.join(dit_dir, "config.json"),
        "--output_path",
        args.output,
    ],
    check=True,
)
subprocess.run(
    [
        python,
        os.path.join(REPO, "scripts", "convert_flux2_to_diffusers.py"),
        "--vae",
        "--checkpoint_path",
        os.path.join(args.package, "ae.safetensors"),
        "--output_path",
        args.output,
    ],
    check=True,
)

pipe = blocks_class().init_pipeline()
pipe.update_components(
    transformer=Flux3Transformer2DModel.from_pretrained(args.output, subfolder="transformer", dtype=torch.bfloat16),
    vae=AutoencoderKLFlux2.from_pretrained(args.output, subfolder="vae"),  # fp32, as shipped
    text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(args.package, "text_encoder"), dtype=torch.bfloat16
    ),
    tokenizer=AutoProcessor.from_pretrained(tokenizer_dir),
)
pipe.save_pretrained(
    args.output,
    overwrite_modular_index=True,
    push_to_hub=args.push_to_hub is not None,
    repo_id=args.push_to_hub,
    private=True,
)
print(f"saved {args.output}")
