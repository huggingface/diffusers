"""The two examples of MiniMax's H3-Image guide (text-to-image and edit) through the diffusers pipeline.

The checkpoint and the example inputs come from `diffusers-internal-dev/minimax-h3-image`. Writes
outputs/{t2i,edit}.png, to compare with reference/{t2i,edit}_reference.png in that repo (what MiniMax's SGLang server
produced for the same requests). Expect the same scene and quality but different details: the prompt embeddings are
not bit-identical to SGLang's. See run_examples_diffusers_ref_embeds.py for the run that matches the reference.
"""

import json
import os
import time

import torch
from huggingface_hub import snapshot_download
from PIL import Image

from diffusers import ComponentsManager, ModularPipeline


REPO = "diffusers-internal-dev/minimax-h3-image"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

reference = os.path.join(snapshot_download(REPO, allow_patterns="reference/*"), "reference")

manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")
pipe = ModularPipeline.from_pretrained(REPO, components_manager=manager)
pipe.load_components(dtype=torch.bfloat16)
# The reference runs flash-attention 3; this backend needs the `kernels` package.
pipe.transformer.set_attention_backend(os.environ.get("ATTN_BACKEND", "_flash_3_hub"))

with open(f"{reference}/request.t2i.json") as f:
    request = json.load(f)
width, height = (int(v) for v in request["size"].split("x"))
start = time.time()
image = pipe(
    prompt=request["prompt"],
    height=height,
    width=width,
    generator=torch.Generator("cpu").manual_seed(request["seed"]),
    output="images",
)[0]
image.save(f"{OUT}/t2i.png")
print(f"t2i {image.size} in {time.time() - start:.0f}s", flush=True)

with open(f"{reference}/prompt.edit.txt") as f:
    prompt = f.read()
start = time.time()
image = pipe(
    prompt=prompt,
    image=Image.open(f"{reference}/picture1.png"),
    height=4096,
    width=4096,
    generator=torch.Generator("cpu").manual_seed(1234),
    output="images",
)[0]
image.save(f"{OUT}/edit.png")
print(f"edit {image.size} in {time.time() - start:.0f}s", flush=True)
