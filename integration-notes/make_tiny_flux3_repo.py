"""Build a tiny local FLUX.3 image modular repo for fast tests.

Creates randomly-initialized tiny versions of the four loadable components with
self-consistent geometry (the transformer's token width derives from the tiny VAE
config the same way the real repo's does), saves them under ../tiny-flux3-image-modular
(next to this repo) with a handwritten modular_model_index.json, and smoke-tests a
2-step t2i run and a 2-reference i2i run.

The text encoder + processor are vendored from trl-internal-testing's tiny Qwen3-VL
(hidden 16, 2 layers, real Qwen vocab/processor files); tests pass
text_encoder_out_layers=(1,), so joint_attention_dim = 16 * 1 = 16.

Upload later as hf-internal-testing/tiny-flux3-image-modular and point the tests there.
"""

import json
import os
import sys

import PIL.Image
import torch
from huggingface_hub import snapshot_download


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffusers import AutoencoderKLFlux2, Flux3Transformer2DModel, ModularPipeline  # noqa: E402


OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tiny-flux3-image-modular"))
TINY_QWEN3VL = "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration"

torch.manual_seed(0)

# text encoder + processor: vendor the tiny Qwen3-VL
qwen_dir = snapshot_download(TINY_QWEN3VL)
os.makedirs(OUT, exist_ok=True)
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: E402


text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(qwen_dir)
text_encoder.save_pretrained(os.path.join(OUT, "text_encoder"))
tokenizer = AutoProcessor.from_pretrained(qwen_dir)
tokenizer.save_pretrained(os.path.join(OUT, "tokenizer"))
text_hidden = text_encoder.config.text_config.hidden_size

vae = AutoencoderKLFlux2(
    block_out_channels=[32, 64],
    down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
    up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
    layers_per_block=1,
    latent_channels=4,
    norm_num_groups=16,
    patch_size=[2, 2],
    sample_size=32,
)
vae.save_pretrained(os.path.join(OUT, "vae"))

transformer = Flux3Transformer2DModel(
    in_channels=vae.config.latent_channels * 4,
    cond_in_channels=vae.config.latent_channels * 4,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=text_hidden,  # tests pass text_encoder_out_layers=(1,)
    timestep_channels=32,
    axes_dims_rope=(2, 2, 2, 2),
)
transformer.save_pretrained(os.path.join(OUT, "transformer"))

index = {
    "_class_name": "Flux3ModularPipeline",
    "_blocks_class_name": "Flux3AutoBlocks",
    "_diffusers_version": "0.40.0.dev0",
}
for name, lib, cls in [
    ("transformer", "diffusers", "Flux3Transformer2DModel"),
    ("vae", "diffusers", "AutoencoderKLFlux2"),
    ("text_encoder", "transformers", "Qwen3VLForConditionalGeneration"),
    ("tokenizer", "transformers", "AutoProcessor"),
]:
    index[name] = [
        lib,
        cls,
        {
            "type_hint": [lib, cls],
            "pretrained_model_name_or_path": OUT,
            "subfolder": name,
            "variant": None,
            "revision": None,
        },
    ]
with open(os.path.join(OUT, "modular_model_index.json"), "w") as f:
    json.dump(index, f, indent=2)
    f.write("\n")

print(f"saved tiny repo to {OUT}")

# smoke tests: 2-step t2i, and i2i with two references of different sizes
pipe = ModularPipeline.from_pretrained(OUT)
pipe.load_components(torch_dtype=torch.float32)
common = {
    "num_inference_steps": 2,
    "max_sequence_length": 8,
    "text_encoder_out_layers": (1,),
    "generator": torch.Generator().manual_seed(0),
}
out = pipe(prompt="a tiny fox", height=64, width=64, **common, output="images")
print("t2i smoke test OK:", out[0].size)
references = [PIL.Image.new("RGB", (64, 48), "red"), PIL.Image.new("RGB", (32, 32), "blue")]
out = pipe(prompt="a tiny fox", image=references, height=64, width=64, **common, output="images")
print("i2i smoke test OK:", out[0].size)
