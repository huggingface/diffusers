import argparse

import safetensors.torch
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, GenerationConfig, Mistral3ForConditionalGeneration

from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2Pipeline, Flux2Transformer2DModel
from diffusers.loaders.conversion.checkpoint import convert_component_checkpoint
from diffusers.loaders.conversion.configs.flux2 import get_flux2_transformer_config


"""
# VAE

python scripts/recipes/flux2.py  \
--original_state_dict_repo_id "diffusers-internal-dev/new-model-image" \
--vae_filename "flux2-vae.sft" \
--output_path "/raid/yiyi/dummy-flux2-diffusers" \
--vae

# DiT

python scripts/recipes/flux2.py \
  --original_state_dict_repo_id diffusers-internal-dev/new-model-image \
  --dit_filename flux-dev-dummy.sft \
  --dit \
  --output_path .

# Full pipe

python scripts/recipes/flux2.py \
  --original_state_dict_repo_id diffusers-internal-dev/new-model-image \
  --dit_filename flux-dev-dummy.sft \
  --vae_filename "flux2-vae.sft" \
  --dit --vae --full_pipe \
  --output_path .
"""


parser = argparse.ArgumentParser()
parser.add_argument("--original_state_dict_repo_id", default=None, type=str)
parser.add_argument("--vae_filename", default="flux2-vae.sft", type=str)
parser.add_argument("--dit_filename", default="flux2-dev.safetensors", type=str)
parser.add_argument("--vae", action="store_true")
parser.add_argument("--dit", action="store_true")
parser.add_argument("--vae_dtype", type=str, default="fp32")
parser.add_argument("--dit_dtype", type=str, default="bf16")
parser.add_argument("--checkpoint_path", default=None, type=str)
parser.add_argument("--full_pipe", action="store_true")
parser.add_argument("--output_path", type=str)

args = parser.parse_args()


def load_original_checkpoint(args, filename):
    if args.original_state_dict_repo_id is not None:
        ckpt_path = hf_hub_download(repo_id=args.original_state_dict_repo_id, filename=filename)
    elif args.checkpoint_path is not None:
        ckpt_path = args.checkpoint_path
    else:
        raise ValueError(" please provide either `original_state_dict_repo_id` or a local `checkpoint_path`")

    original_state_dict = safetensors.torch.load_file(ckpt_path)
    return original_state_dict


def convert_flux2_vae_checkpoint_to_diffusers(original_state_dict, config):
    return convert_component_checkpoint(original_state_dict, config, "AutoencoderKLFlux2")


# in SD3 original implementation of AdaLayerNormContinuous, it split linear projection output into shift, scale;
# while in diffusers it split into scale, shift. Here we swap the linear projection weights in order to be able to use
# diffusers implementation


def convert_flux2_transformer_to_diffusers(original_state_dict, model_type):
    config = get_flux2_transformer_config(model_type)["diffusers_config"]
    return Flux2Transformer2DModel.from_single_file(original_state_dict, config=config)


def main(args):
    if args.vae:
        original_vae_ckpt = load_original_checkpoint(args, filename=args.vae_filename)
        vae = AutoencoderKLFlux2()
        converted_vae_state_dict = convert_flux2_vae_checkpoint_to_diffusers(original_vae_ckpt, vae.config)
        vae.load_state_dict(converted_vae_state_dict, strict=True)
        if not args.full_pipe:
            vae_dtype = torch.bfloat16 if args.vae_dtype == "bf16" else torch.float32
            vae.to(vae_dtype).save_pretrained(f"{args.output_path}/vae")

    if args.dit:
        original_dit_ckpt = load_original_checkpoint(args, filename=args.dit_filename)

        if "klein-4b" in args.dit_filename:
            model_type = "klein-4b"
        elif "klein-9b" in args.dit_filename:
            model_type = "klein-9b"
        else:
            model_type = "flux2-dev"
        transformer = convert_flux2_transformer_to_diffusers(original_dit_ckpt, model_type)
        if not args.full_pipe:
            dit_dtype = torch.bfloat16 if args.dit_dtype == "bf16" else torch.float32
            transformer.to(dit_dtype).save_pretrained(f"{args.output_path}/transformer")

    if args.full_pipe:
        tokenizer_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
        text_encoder_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
        generate_config = GenerationConfig.from_pretrained(text_encoder_id)
        generate_config.do_sample = True
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            text_encoder_id, generation_config=generate_config, torch_dtype=torch.bfloat16
        )
        tokenizer = AutoProcessor.from_pretrained(tokenizer_id)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "black-forest-labs/FLUX.1-dev", subfolder="scheduler"
        )

        if_distilled = "base" not in args.dit_filename

        pipe = Flux2Pipeline(
            vae=vae,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            if_distilled=if_distilled,
        )
        pipe.save_pretrained(args.output_path)


if __name__ == "__main__":
    main(args)
