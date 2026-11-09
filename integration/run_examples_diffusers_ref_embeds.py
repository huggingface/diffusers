"""The two examples of MiniMax's H3-Image guide through the diffusers pipeline, with the reference's prompt embeddings.

Same as run_examples_diffusers.py, except for two things:

* the text encoder block is popped from the pipeline, and the prompt embeddings MiniMax's SGLang server computed for
  the same request (reference/{t2i,edit}_prompt_embeds.safetensors in `diffusers-internal-dev/minimax-h3-image`) are
  passed in instead;
* for the edit, the initial noises are drawn outside the pipeline in the reference's order and passed in as `latents`
  and `reference_noise` (see `reference_order_noise`). Text-to-image has a single draw, so a seeded generator is enough.

Everything else is the diffusers pipeline. With those inputs shared, the outputs are visually identical to
reference/{t2i,edit}_reference.png (PSNR 54-56 dB; the final latents are bit-identical, the remaining difference is
the VAE decoder's kernels).

Writes outputs/{t2i,edit}_ref_embeds.png. For a bit-exact edit, run on the PyTorch build of the reference
(torch 2.14.1): the VAE encoder's TF32 convolutions round differently across cuDNN builds.
"""

import json
import os
import time

import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from safetensors.torch import load_file

from diffusers import ComponentsManager, ModularPipeline
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import get_image_canvas_size
from diffusers.modular_pipelines.minimax_h3_image.before_encoder import bound_short_edge
from diffusers.utils.torch_utils import randn_tensor


REPO = "diffusers-internal-dev/minimax-h3-image"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

reference = os.path.join(snapshot_download(REPO, allow_patterns="reference/*"), "reference")


def reference_order_noise(pipe, seed, height, width, images=()):
    """The initial noises of a request, drawn in the reference's order.

    The reference draws from one generator: the noise of the generated image, then one noise tensor per reference
    image, then the VAE posterior sample of every reference. The diffusers pipeline samples the posteriors first (in
    its VAE encoder block), so to reproduce the reference the two noises are drawn here and passed in, which leaves
    the returned generator exactly where the reference takes its posterior samples.
    """
    generator = torch.Generator("cpu").manual_seed(seed)
    multiple, ratio, channels = pipe.canvas_multiple, pipe.vae_spatial_compression_ratio, pipe.vae_latent_channels
    canvas_height, canvas_width = get_image_canvas_size(height, width, 2048 * 2048, multiple)
    latents = randn_tensor((1, channels, canvas_height // ratio, canvas_width // ratio), generator=generator)
    reference_noise = []
    for image in images:
        # The latent grid of a reference: its size after the VAE's short-edge bounds, cropped to whole patches.
        ref_width, ref_height = bound_short_edge(
            *ImageOps.exif_transpose(image).size,
            pipe.config.reference_min_short_edge,
            pipe.config.reference_max_short_edge,
            multiple,
        )
        shape = (1, channels, ref_height // ratio // 2 * 2, ref_width // ratio // 2 * 2)
        reference_noise.append(randn_tensor(shape, generator=generator))
    return {"latents": latents, "reference_noise": reference_noise, "generator": generator}


manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")

# The pipeline without its text encoder: `prompt_embeds` and `text_token_tags` become inputs.
blocks = ModularPipeline.from_pretrained(REPO).blocks
blocks.sub_blocks.pop("text_encoder")
pipe = blocks.init_pipeline(REPO, components_manager=manager)
pipe.load_components(dtype=torch.bfloat16)
# The reference runs flash-attention 3; this backend needs the `kernels` package.
pipe.transformer.set_attention_backend(os.environ.get("ATTN_BACKEND", "_flash_3_hub"))

with open(f"{reference}/request.t2i.json") as f:
    request = json.load(f)
width, height = (int(v) for v in request["size"].split("x"))
start = time.time()
image = pipe(
    **load_file(f"{reference}/t2i_prompt_embeds.safetensors"),
    height=height,
    width=width,
    generator=torch.Generator("cpu").manual_seed(request["seed"]),
    output="images",
)[0]
image.save(f"{OUT}/t2i_ref_embeds.png")
print(f"t2i {image.size} in {time.time() - start:.0f}s", flush=True)

picture = Image.open(f"{reference}/picture1.png")
start = time.time()
image = pipe(
    **load_file(f"{reference}/edit_prompt_embeds.safetensors"),
    image=picture,
    height=4096,
    width=4096,
    **reference_order_noise(pipe, seed=1234, height=4096, width=4096, images=[picture]),
    output="images",
)[0]
image.save(f"{OUT}/edit_ref_embeds.png")
print(f"edit {image.size} in {time.time() - start:.0f}s", flush=True)
