import argparse
import os

import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, OmniGenTransformer2DModel, OmniGenPipeline


def main(args):
    # checkpoint from https://huggingface.co/Shitao/OmniGen-v1

    if not os.path.exists(args.origin_ckpt_path):
        print("Model not found, downloading...")
        cache_folder = os.getenv('HF_HUB_CACHE')
        args.origin_ckpt_path = snapshot_download(repo_id=args.origin_ckpt_path,
                                       cache_dir=cache_folder,
                                       ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5',
                                                        'model.pt'])
        print(f"Downloaded model to {args.origin_ckpt_path}")

    ckpt = os.path.join(args.origin_ckpt_path, 'model.safetensors')
    ckpt = load_file(ckpt, device="cpu")

    mapping_dict = {
        "pos_embed": "patch_embedding.pos_embed",
        "x_embedder.proj.weight": "patch_embedding.output_image_proj.weight",
        "x_embedder.proj.bias": "patch_embedding.output_image_proj.bias",
        "input_x_embedder.proj.weight": "patch_embedding.input_image_proj.weight",
        "input_x_embedder.proj.bias": "patch_embedding.input_image_proj.bias",
        "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
        "final_layer.adaLN_modulation.1.bias": "norm_out.linear.bias",
        "final_layer.linear.weight": "proj_out.weight",
        "final_layer.linear.bias": "proj_out.bias",

    }

    converted_state_dict = {}
    for k, v in ckpt.items():
        if k in mapping_dict:
            converted_state_dict[mapping_dict[k]] = v
        else:
            converted_state_dict[k] = v

    transformer_config = AutoConfig.from_pretrained(args.origin_ckpt_path)

    transformer = OmniGenTransformer2DModel(
        transformer_config=transformer_config,
        patch_size=2,
        in_channels=4,
        pos_embed_max_size=192,
    )
    transformer.load_state_dict(converted_state_dict, strict=True)

    num_model_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total number of transformer parameters: {num_model_params}")

    scheduler = FlowMatchEulerDiscreteScheduler()

    vae = AutoencoderKL.from_pretrained(os.path.join(args.origin_ckpt_path, "vae"), torch_dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.origin_ckpt_path)


    pipeline = OmniGenPipeline(
        tokenizer=tokenizer, transformer=transformer, vae=vae, scheduler=scheduler
    )
    pipeline.save_pretrained(args.dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--origin_ckpt_path", default="Shitao/OmniGen-v1", type=str, required=False, help="Path to the checkpoint to convert."
    )

    parser.add_argument("--dump_path", default="OmniGen-v1-diffusers", type=str, required=True, help="Path to the output pipeline.")

    args = parser.parse_args()
    main(args)
