"""
This script demonstrates how to extract a LoRA checkpoint from a fully finetuned model with the CogVideoX model.

To make it work for other models:

* Change the model class. Here we use `CogVideoXTransformer3DModel`. For Flux, it would be `FluxTransformer2DModel`,
for example. (TODO: more reason to add `AutoModel`).
* Spply path to the base checkpoint via `base_ckpt_path`.
* Supply path to the fully fine-tuned checkpoint via `--finetune_ckpt_path`.
* Change the `--rank` as needed.

Example usage:

```bash
python extract_lora_from_model.py \
    --base_ckpt_path=THUDM/CogVideoX-5b \
    --finetune_ckpt_path=finetrainers/cakeify-v0 \
    --lora_out_path=cakeify_lora.safetensors
```

Script is adapted from
https://github.com/Stability-AI/stability-ComfyUI-nodes/blob/001154622564b17223ce0191803c5fff7b87146c/control_lora_create.py
"""

import argparse

import torch
from safetensors.torch import save_file
from tqdm.auto import tqdm

from diffusers import CogVideoXTransformer3DModel


RANK = 64
CLAMP_QUANTILE = 0.99


# Comes from
# https://github.com/Stability-AI/stability-ComfyUI-nodes/blob/001154622564b17223ce0191803c5fff7b87146c/control_lora_create.py#L9
def extract_lora(diff, rank):
    # Important to use CUDA otherwise, very slow!
    if torch.cuda.is_available():
        diff = diff.to("cuda")

    is_conv2d = len(diff.shape) == 4
    kernel_size = None if not is_conv2d else diff.size()[2:4]
    is_conv2d_3x3 = is_conv2d and kernel_size != (1, 1)
    out_dim, in_dim = diff.size()[0:2]
    rank = min(rank, in_dim, out_dim)

    if is_conv2d:
        if is_conv2d_3x3:
            diff = diff.flatten(start_dim=1)
        else:
            diff = diff.squeeze()

    U, S, Vh = torch.linalg.svd(diff.float())
    U = U[:, :rank]
    S = S[:rank]
    U = U @ torch.diag(S)
    Vh = Vh[:rank, :]

    dist = torch.cat([U.flatten(), Vh.flatten()])
    hi_val = torch.quantile(dist, CLAMP_QUANTILE)
    low_val = -hi_val

    U = U.clamp(low_val, hi_val)
    Vh = Vh.clamp(low_val, hi_val)
    if is_conv2d:
        U = U.reshape(out_dim, rank, 1, 1)
        Vh = Vh.reshape(rank, in_dim, kernel_size[0], kernel_size[1])
    return (U.cpu(), Vh.cpu())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_ckpt_path",
        default=None,
        type=str,
        required=True,
        help="Base checkpoint path from which the model was finetuned. Can be a model ID on the Hub.",
    )
    parser.add_argument(
        "--base_subfolder",
        default="transformer",
        type=str,
        help="subfolder to load the base checkpoint from if any.",
    )
    parser.add_argument(
        "--finetune_ckpt_path",
        default=None,
        type=str,
        required=True,
        help="Fully fine-tuned checkpoint path. Can be a model ID on the Hub.",
    )
    parser.add_argument(
        "--finetune_subfolder",
        default=None,
        type=str,
        help="subfolder to load the fulle finetuned checkpoint from if any.",
    )
    parser.add_argument("--rank", default=64, type=int)
    parser.add_argument("--lora_out_path", default=None, type=str, required=True)
    args = parser.parse_args()

    if not args.lora_out_path.endswith(".safetensors"):
        raise ValueError("`lora_out_path` must end with `.safetensors`.")

    return args


@torch.no_grad()
def main(args):
    model_finetuned = CogVideoXTransformer3DModel.from_pretrained(
        args.finetune_ckpt_path, subfolder=args.finetune_subfolder, torch_dtype=torch.bfloat16
    )
    state_dict_ft = model_finetuned.state_dict()

    # Change the `subfolder` as needed.
    base_model = CogVideoXTransformer3DModel.from_pretrained(
        args.base_ckpt_path, subfolder=args.base_subfolder, torch_dtype=torch.bfloat16
    )
    state_dict = base_model.state_dict()
    output_dict = {}

    for k in tqdm(state_dict, desc="Extracting LoRA..."):
        original_param = state_dict[k]
        finetuned_param = state_dict_ft[k]
        if len(original_param.shape) >= 2:
            diff = finetuned_param.float() - original_param.float()
            out = extract_lora(diff, RANK)
            name = k

            if name.endswith(".weight"):
                name = name[: -len(".weight")]
            down_key = "{}.lora_A.weight".format(name)
            up_key = "{}.lora_B.weight".format(name)

            output_dict[up_key] = out[0].contiguous().to(finetuned_param.dtype)
            output_dict[down_key] = out[1].contiguous().to(finetuned_param.dtype)

    prefix = "transformer" if "transformer" in base_model.__class__.__name__.lower() else "unet"
    output_dict = {f"{prefix}.{k}": v for k, v in output_dict.items()}
    save_file(output_dict, args.lora_out_path)
    print(f"LoRA saved and it contains {len(output_dict)} keys.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
