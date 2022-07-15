#!/usr/bin/env python3
import json
import os
from diffusers import UNetUnconditionalModel
from scripts.convert_ldm_original_checkpoint_to_diffusers import convert_ldm_checkpoint
from huggingface_hub import hf_hub_download
import torch

model_id = "fusing/latent-diffusion-celeba-256"
subfolder = "unet"
#model_id = "fusing/unet-ldm-dummy"
#subfolder = None

checkpoint = "diffusion_model.pt"
config = "config.json"

if subfolder is not None:
    checkpoint = os.path.join(subfolder, checkpoint)
    config = os.path.join(subfolder, config)

original_checkpoint = torch.load(hf_hub_download(model_id, checkpoint))
config_path = hf_hub_download(model_id, config)

with open(config_path) as f:
    config = json.load(f)

checkpoint = convert_ldm_checkpoint(original_checkpoint, config)


def current_codebase_conversion():
    model = UNetUnconditionalModel.from_pretrained(model_id, subfolder=subfolder, ldm=True)
    model.eval()

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    noise = torch.randn(1, model.config.in_channels, model.config.image_size, model.config.image_size)
    time_step = torch.tensor([10] * noise.shape[0])

    with torch.no_grad():
        output = model(noise, time_step)

    return model.state_dict()


currently_converted_checkpoint = current_codebase_conversion()


def diff_between_checkpoints(ch_0, ch_1):
    all_layers_included = False

    if not set(ch_0.keys()) == set(ch_1.keys()):
        print(f"Contained in ch_0 and not in ch_1 (Total: {len((set(ch_0.keys()) - set(ch_1.keys())))})")
        for key in sorted(list((set(ch_0.keys()) - set(ch_1.keys())))):
            print(f"\t{key}")

        print(f"Contained in ch_1 and not in ch_0 (Total: {len((set(ch_1.keys()) - set(ch_0.keys())))})")
        for key in sorted(list((set(ch_1.keys()) - set(ch_0.keys())))):
            print(f"\t{key}")
    else:
        print("Keys are the same between the two checkpoints")
        all_layers_included = True

    keys = ch_0.keys()
    non_equal_keys = []

    if all_layers_included:
        for key in keys:
            try:
                if not torch.allclose(ch_0[key].cpu(), ch_1[key].cpu()):
                    non_equal_keys.append(f'{key}. Diff: {torch.max(torch.abs(ch_0[key].cpu() - ch_1[key].cpu()))}')

            except RuntimeError as e:
                print(e)
                non_equal_keys.append(f'{key}. Diff in shape: {ch_0[key].size()} vs {ch_1[key].size()}')

        if len(non_equal_keys):
            non_equal_keys = '\n\t'.join(non_equal_keys)
            print(f"These keys do not satisfy equivalence requirement:\n\t{non_equal_keys}")
        else:
            print("All keys are equal across checkpoints.")


diff_between_checkpoints(currently_converted_checkpoint, checkpoint)
torch.save(checkpoint, "/path/to/checkpoint/")
