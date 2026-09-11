import argparse
import tempfile

import torch
from accelerate import load_checkpoint_and_dispatch

from diffusers.loaders.conversion import get_conversion
from diffusers.loaders.conversion.configs.shap_e import PRIOR_CONFIG, PRIOR_IMAGE_CONFIG, RENDERER_CONFIG
from diffusers.models.transformers.prior_transformer import PriorTransformer
from diffusers.pipelines.shap_e import ShapERenderer


"""
Example - From the diffusers root directory:

Download weights:
```sh
$ wget  "https://openaipublic.azureedge.net/main/shap-e/text_cond.pt"
```

Convert the model:
```sh
$ python scripts/recipes/shap_e.py \
      --prior_checkpoint_path  /home/yiyi_huggingface_co/shap-e/shap_e_model_cache/text_cond.pt \
      --prior_image_checkpoint_path /home/yiyi_huggingface_co/shap-e/shap_e_model_cache/image_cond.pt \
      --transmitter_checkpoint_path /home/yiyi_huggingface_co/shap-e/shap_e_model_cache/transmitter.pt\
      --dump_path /home/yiyi_huggingface_co/model_repo/shap-e-img2img/shap_e_renderer\
      --debug renderer
```
"""


# prior


def prior_model_from_original_config():
    model = PriorTransformer(**PRIOR_CONFIG)

    return model


def prior_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    state = {key: value for key, value in checkpoint.items() if key.startswith("wrapped.")}
    return get_conversion("PriorTransformer", {**dict(model.config), "original_format": "shap_e"}).to_diffusers(state)


# done prior


# prior_image (only slightly different from prior)


# Uses default arguments


def prior_image_model_from_original_config():
    model = PriorTransformer(**PRIOR_IMAGE_CONFIG)

    return model


def prior_image_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    state = {key: value for key, value in checkpoint.items() if key.startswith("wrapped.")}
    return get_conversion("PriorTransformer", {**dict(model.config), "original_format": "shap_e"}).to_diffusers(state)


# done prior_image


# renderer

## create the lookup table for marching cubes method used in MeshDecoder


def renderer_model_from_original_config():
    model = ShapERenderer(**RENDERER_CONFIG)

    return model


def renderer_model_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    state = {
        key: value
        for key, value in checkpoint.items()
        if key.startswith(("renderer.nerstf.mlp.", "encoder.params_proj.projections."))
    }
    return get_conversion("ShapERenderer", dict(model.config)).to_diffusers(state)


# done renderer


# TODO maybe document and/or can do more efficiently (build indices in for loop and extract once for each split?)


# done unet utils


# Driver functions


def prior(*, args, checkpoint_map_location):
    print("loading prior")

    prior_checkpoint = torch.load(args.prior_checkpoint_path, map_location=checkpoint_map_location)

    prior_model = prior_model_from_original_config()

    prior_diffusers_checkpoint = prior_original_checkpoint_to_diffusers_checkpoint(prior_model, prior_checkpoint)

    del prior_checkpoint

    load_prior_checkpoint_to_model(prior_diffusers_checkpoint, prior_model)

    print("done loading prior")

    return prior_model


def prior_image(*, args, checkpoint_map_location):
    print("loading prior_image")

    print(f"load checkpoint from {args.prior_image_checkpoint_path}")
    prior_checkpoint = torch.load(args.prior_image_checkpoint_path, map_location=checkpoint_map_location)

    prior_model = prior_image_model_from_original_config()

    prior_diffusers_checkpoint = prior_image_original_checkpoint_to_diffusers_checkpoint(prior_model, prior_checkpoint)

    del prior_checkpoint

    load_prior_checkpoint_to_model(prior_diffusers_checkpoint, prior_model)

    print("done loading prior_image")

    return prior_model


def renderer(*, args, checkpoint_map_location):
    print(" loading renderer")

    renderer_checkpoint = torch.load(args.transmitter_checkpoint_path, map_location=checkpoint_map_location)

    renderer_model = renderer_model_from_original_config()

    renderer_diffusers_checkpoint = renderer_model_original_checkpoint_to_diffusers_checkpoint(
        renderer_model, renderer_checkpoint
    )

    del renderer_checkpoint

    load_checkpoint_to_model(renderer_diffusers_checkpoint, renderer_model, strict=True)

    print("done loading renderer")

    return renderer_model


# prior model will expect clip_mean and clip_std, which are missing from the state_dict
PRIOR_EXPECTED_MISSING_KEYS = ["clip_mean", "clip_std"]


def load_prior_checkpoint_to_model(checkpoint, model):
    with tempfile.NamedTemporaryFile() as file:
        torch.save(checkpoint, file.name)
        del checkpoint
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(file.name), strict=False)
        missing_keys = list(set(missing_keys) - set(PRIOR_EXPECTED_MISSING_KEYS))

        if len(unexpected_keys) > 0:
            raise ValueError(f"Unexpected keys when loading prior model: {unexpected_keys}")
        if len(missing_keys) > 0:
            raise ValueError(f"Missing keys when loading prior model: {missing_keys}")


def load_checkpoint_to_model(checkpoint, model, strict=False):
    with tempfile.NamedTemporaryFile() as file:
        torch.save(checkpoint, file.name)
        del checkpoint
        if strict:
            model.load_state_dict(torch.load(file.name), strict=True)
        else:
            load_checkpoint_and_dispatch(model, file.name, device_map="auto")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")

    parser.add_argument(
        "--prior_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the prior checkpoint to convert.",
    )

    parser.add_argument(
        "--prior_image_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the prior_image checkpoint to convert.",
    )

    parser.add_argument(
        "--transmitter_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the transmitter checkpoint to convert.",
    )

    parser.add_argument(
        "--checkpoint_load_device",
        default="cpu",
        type=str,
        required=False,
        help="The device passed to `map_location` when loading checkpoints.",
    )

    parser.add_argument(
        "--debug",
        default=None,
        type=str,
        required=False,
        help="Only run a specific stage of the convert script. Used for debugging",
    )

    args = parser.parse_args()

    print(f"loading checkpoints to {args.checkpoint_load_device}")

    checkpoint_map_location = torch.device(args.checkpoint_load_device)

    if args.debug is not None:
        print(f"debug: only executing {args.debug}")

    if args.debug is None:
        print("YiYi TO-DO")
    elif args.debug == "prior":
        prior_model = prior(args=args, checkpoint_map_location=checkpoint_map_location)
        prior_model.save_pretrained(args.dump_path)
    elif args.debug == "prior_image":
        prior_model = prior_image(args=args, checkpoint_map_location=checkpoint_map_location)
        prior_model.save_pretrained(args.dump_path)
    elif args.debug == "renderer":
        renderer_model = renderer(args=args, checkpoint_map_location=checkpoint_map_location)
        renderer_model.save_pretrained(args.dump_path)
    else:
        raise ValueError(f"unknown debug value : {args.debug}")
