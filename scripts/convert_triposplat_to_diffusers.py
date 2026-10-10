# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
from pathlib import Path

import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    TripoSplatAutoBlocks,
    TripoSplatPipeline,
)
from diffusers.loaders.single_file_utils import (
    convert_triposplat_birefnet_checkpoint_to_diffusers,
    convert_triposplat_decoder_checkpoint_to_diffusers,
    convert_triposplat_transformer_checkpoint_to_diffusers,
)
from diffusers.models.autoencoders.autoencoder_triposplat import TripoSplatGaussianDecoder
from diffusers.models.transformers.transformer_triposplat import TripoSplatTransformer3DModel
from diffusers.pipelines.triposplat.modeling_birefnet import BiRefNetModel


def _load_converted(model_class, state_dict, **kwargs):
    with init_empty_weights():
        model = model_class(**kwargs)
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.eval()


def convert_triposplat_transformer(state_dict):
    return _load_converted(
        TripoSplatTransformer3DModel, convert_triposplat_transformer_checkpoint_to_diffusers(state_dict)
    )


def convert_triposplat_decoder(state_dict):
    return _load_converted(TripoSplatGaussianDecoder, convert_triposplat_decoder_checkpoint_to_diffusers(state_dict))


def convert_triposplat_birefnet(state_dict):
    return _load_converted(BiRefNetModel, convert_triposplat_birefnet_checkpoint_to_diffusers(state_dict))


def convert_triposplat_image_encoder(state_dict):
    config = DINOv3ViTConfig(
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=20,
        patch_size=16,
        num_register_tokens=4,
        layer_norm_eps=1e-5,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        use_gated_mlp=True,
        hidden_act="silu",
        rope_theta=100.0,
    )
    with init_empty_weights():
        model = DINOv3ViTModel(config)
    target_keys = model.state_dict()
    converted = {
        (f"model.{key}" if key.startswith("layer.") and f"model.{key}" in target_keys else key): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(converted, strict=True, assign=True)
    return model.eval()


def convert_triposplat_image_vae(state_dict):
    return _load_converted(AutoencoderKLFlux2, state_dict, batch_norm_eps=1e-5)


def main():
    parser = argparse.ArgumentParser(description="Convert the official TripoSplat inference checkpoints to Diffusers.")
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--revision", default="56a96e603204ec410c4da60c13ea4fa09a2169a9")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="auto uses the reference component dtypes; an explicit dtype applies to every component.",
    )
    parser.add_argument("--include_background_remover", action="store_true")
    args = parser.parse_args()
    files = {
        "transformer": ("diffusion_models/triposplat_fp16.safetensors", convert_triposplat_transformer),
        "decoder": ("vae/triposplat_vae_decoder_fp16.safetensors", convert_triposplat_decoder),
        "image_encoder": ("clip_vision/dino_v3_vit_h.safetensors", convert_triposplat_image_encoder),
        "vae": ("vae/flux2-vae.safetensors", convert_triposplat_image_vae),
    }
    if args.include_background_remover:
        files["background_remover"] = ("background_removal/birefnet.safetensors", convert_triposplat_birefnet)
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = Path(
            snapshot_download(
                "VAST-AI/TripoSplat", revision=args.revision, allow_patterns=[item[0] for item in files.values()]
            )
        )
    components = {}
    for name, (filename, convert) in files.items():
        model = convert(load_file(str(checkpoint_dir / filename)))
        if args.dtype == "auto":
            dtype = torch.bfloat16 if name in ("image_encoder", "vae") else torch.float16
        else:
            dtype = getattr(torch, args.dtype)
        model.to(dtype=dtype).save_pretrained(args.output_dir / name)
        reloaded = type(model).from_pretrained(args.output_dir / name, torch_dtype=dtype)
        if set(model.state_dict()) != set(reloaded.state_dict()):
            raise RuntimeError(f"The {name} checkpoint did not round-trip.")
        for key, value in model.state_dict().items():
            if not torch.equal(value, reloaded.state_dict()[key]):
                raise RuntimeError(f"The {name} checkpoint changed {key} on reload.")
        components[name] = reloaded
        del model
        print(f"Converted {name}", flush=True)
    components.setdefault("background_remover", None)
    pipeline = TripoSplatPipeline(**components, scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0))
    pipeline.save_pretrained(args.output_dir)
    modular = TripoSplatAutoBlocks().init_pipeline()
    modular.update_components(**pipeline.components)
    modular.save_pretrained(str(args.output_dir), overwrite_modular_index=True)
    print(f"Saved standard and modular pipelines to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
