# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
""" Conversion script for the Stable Diffusion checkpoints."""

import re
from contextlib import nullcontext
from io import BytesIO

import requests
import torch
import yaml
from safetensors.torch import load_file as safe_load
from transformers import (
    BertTokenizerFast,
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from ..models import AutoencoderKL, UNet2DConditionModel
from ..pipelines.latent_diffusion.pipeline_latent_diffusion import LDMBertConfig, LDMBertModel
from ..schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ..utils import is_accelerate_available, logging


if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

CONFIG_URLS = {
    "v1": "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml",
    "v2": "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml",
    "xl": "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml",
    "xl_refiner": "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_refiner.yaml",
    "upscale": "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/x4-upscaling.yaml",
}

CHECKPOINT_KEY_NAMES = {
    "v2": "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight",
    "xl_base": "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias",
    "xl_refiner": "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias",
}

SCHEDULER_DEFAULT_CONFIG = {
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "interpolation_type": "linear",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "timestep_spacing": "leading",
}

DIFFUSERS_TO_LDM_MAPPING = {
    "unet": {
        "layers": {
            "time_embedding.linear_1.weight": "time_embed.0.weight",
            "time_embedding.linear_1.bias": "time_embed.0.bias",
            "time_embedding.linear_2.weight": "time_embed.2.weight",
            "time_embedding.linear_2.bias": "time_embed.2.bias",
            "conv_in.weight": "input_blocks.0.0.weight",
            "conv_in.bias": "input_blocks.0.0.bias",
            "conv_norm_out.weight": "out.0.weight",
            "conv_norm_out.bias": "out.0.bias",
            "conv_out.weight": "out.2.weight",
            "conv_out.bias": "out.2.bias",
        },
        "class_embed_type": {
            "class_embedding.linear_1.weight": "label_emb.0.0.weight",
            "class_embedding.linear_1.bias": "label_emb.0.0.bias",
            "class_embedding.linear_2.weight": "label_emb.0.2.weight",
            "class_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
        "addition_embed_type": {
            "add_embedding.linear_1.weight": "label_emb.0.0.weight",
            "add_embedding.linear_1.bias": "label_emb.0.0.bias",
            "add_embedding.linear_2.weight": "label_emb.0.2.weight",
            "add_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
    },
    "vae": {
        "encoder.conv_in.weight": "encoder.conv_in.weight",
        "encoder.conv_in.bias": "encoder.conv_in.bias",
        "encoder.conv_out.weight": "encoder.conv_out.weight",
        "encoder.conv_out.bias": "encoder.conv_out.bias",
        "encoder.conv_norm_out.weight": "encoder.norm_out.weight",
        "encoder.conv_norm_out.bias": "encoder.norm_out.bias",
        "decoder.conv_in.weight": "decoder.conv_in.weight",
        "decoder.conv_in.bias": "decoder.conv_in.bias",
        "decoder.conv_out.weight": "decoder.conv_out.weight",
        "decoder.conv_out.bias": "decoder.conv_out.bias",
        "decoder.conv_norm_out.weight": "decoder.norm_out.weight",
        "decoder.conv_norm_out.bias": "decoder.norm_out.bias",
        "quant_conv.weight": "quant_conv.weight",
        "quant_conv.bias": "quant_conv.bias",
        "post_quant_conv.weight": "post_quant_conv.weight",
        "post_quant_conv.bias": "post_quant_conv.bias",
    },
    "openclip": {
        # "positional_embedding": "text_model.embeddings.position_embedding.weight",
        # "token_embedding.weight": "text_model.embeddings.token_embedding.weight",
        "ln_final.weight": "text_model.final_layer_norm.weight",
        "ln_final.bias": "text_model.final_layer_norm.bias",
        "text_projection": "text_projection.weight",
        "resblocks.": "text_model.encoder.layers.",
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        ".c_fc.": ".fc1.",
        ".c_proj.": ".fc2.",
        ".attn": ".self_attn",
        "ln_final.": "transformer.text_model.final_layer_norm.",
        "token_embedding.weight": "transformer.text_model.embeddings.token_embedding.weight",
        "positional_embedding": "transformer.text_model.embeddings.position_embedding.weight",
    },
}

LDM_VAE_KEY = "first_stage_model."
LDM_UNET_KEY = "model.diffusion_model."
LDM_CLIP_CONFIG_NAME = "openai/clip-vit-large-patch14"
LDM_CLIP_PREFIX_TO_REMOVE = ["cond_stage_model.transformer.", "conditioner.embedders.0.transformer."]

textenc_conversion_lst = [
    ("positional_embedding", "text_model.embeddings.position_embedding.weight"),
    ("token_embedding.weight", "text_model.embeddings.token_embedding.weight"),
    ("ln_final.weight", "text_model.final_layer_norm.weight"),
    ("ln_final.bias", "text_model.final_layer_norm.bias"),
    ("text_projection", "text_projection.weight"),
]
textenc_conversion_map = {x[0]: x[1] for x in textenc_conversion_lst}

textenc_transformer_conversion_lst = [
    # (stable-diffusion, HF Diffusers)
    ("resblocks.", "text_model.encoder.layers."),
    ("ln_1", "layer_norm1"),
    ("ln_2", "layer_norm2"),
    (".c_fc.", ".fc1."),
    (".c_proj.", ".fc2."),
    (".attn", ".self_attn"),
    ("ln_final.", "transformer.text_model.final_layer_norm."),
    ("token_embedding.weight", "transformer.text_model.embeddings.token_embedding.weight"),
    ("positional_embedding", "transformer.text_model.embeddings.position_embedding.weight"),
]
protected = {re.escape(x[0]): x[1] for x in textenc_transformer_conversion_lst}
textenc_pattern = re.compile("|".join(protected.keys()))


def fetch_original_config_file_from_url(pipeline_class_name, checkpoint):
    if CHECKPOINT_KEY_NAMES["v2"] in checkpoint and checkpoint[CHECKPOINT_KEY_NAMES["v2"]].shape[-1] == 1024:
        config_url = CONFIG_URLS["v2"]

    elif CHECKPOINT_KEY_NAMES["xl_base"] in checkpoint:
        config_url = CONFIG_URLS["xl"]

    elif CHECKPOINT_KEY_NAMES["xl_refiner"] in checkpoint:
        config_url = CONFIG_URLS["xl_refiner"]

    elif pipeline_class_name == "StableDiffusionUpscalePipeline":
        config_url = CONFIG_URLS["upscale"]

    else:
        config_url = CONFIG_URLS["v1"]

    original_config_file = BytesIO(requests.get(config_url).content)

    return original_config_file


def fetch_original_config_file_from_file(config_files: list):
    if "v2" in config_files:
        return config_files["v2"]

    elif "xl" in config_files:
        return config_files["xl"]

    elif "xl_refiner" in config_files:
        return config_files["xl_refiner"]

    else:
        return config_files["v1"]


def fetch_original_config(pipeline_class_name, checkpoint, original_config_file=None, config_files=None):
    if original_config_file:
        original_config = yaml.safe_load(original_config_file)
        return original_config

    elif config_files:
        original_config_file = fetch_original_config_file_from_file(config_files)

    else:
        original_config_file = fetch_original_config_file_from_url(pipeline_class_name, checkpoint)

    original_config = yaml.safe_load(original_config_file)

    return original_config


def load_checkpoint(checkpoint_path_or_dict, device=None, from_safetensors=True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(checkpoint_path_or_dict, str):
        if from_safetensors:
            checkpoint = safe_load(checkpoint_path_or_dict, device="cpu")

        else:
            checkpoint = torch.load(checkpoint_path_or_dict, map_location=device)

    elif isinstance(checkpoint_path_or_dict, dict):
        checkpoint = checkpoint_path_or_dict

    return checkpoint


def infer_model_type(pipeline_class_name, original_config, model_type=None, **kwargs):
    if model_type is not None:
        return model_type

    if pipeline_class_name in ["StableUnCLIPPipeline", "StableUnCLIPImg2ImgPipeline"]:
        model_type = "FrozenOpenCLIPEmbedder"
        return model_type

    has_cond_stage_config = (
        "cond_stage_config" in original_config["model"]["params"]
        and original_config["model"]["params"]["cond_stage_config"] is not None
    )
    has_network_config = (
        "network_config" in original_config["model"]["params"]
        and original_config["model"]["params"]["network_config"] is not None
    )

    if has_cond_stage_config:
        model_type = original_config["model"]["params"]["cond_stage_config"]["target"].split(".")[-1]

    elif has_network_config:
        context_dim = original_config["model"]["params"]["network_config"]["params"]["context_dim"]
        if context_dim == 2048:
            model_type = "SDXL"
        else:
            model_type = "SDXL-Refiner"
    else:
        raise ValueError("Unable to infer model type from config")

    logger.debug(f"No `model_type` given, `model_type` inferred as: {model_type}")

    return model_type


def get_default_scheduler_config():
    return SCHEDULER_DEFAULT_CONFIG


def determine_image_size(pipeline_class_name, original_config, checkpoint, **kwargs):
    image_size = kwargs.get("image_size", 512)
    global_step = checkpoint["global_step"] if "global_step" in checkpoint else None
    model_type = infer_model_type(pipeline_class_name, original_config, **kwargs)

    if pipeline_class_name == "StableDiffusionUpscalePipeline":
        image_size = original_config["model"]["params"].unet_config.params.image_size
        return image_size

    elif model_type in ["SDXL", "SDXL-Refiner"]:
        image_size = 1024
        return image_size

    elif (
        "parameterization" in original_config["model"]["params"]
        and original_config["model"]["params"]["parameterization"] == "v"
    ):
        # NOTE: For stable diffusion 2 base one has to pass `image_size==512`
        # as it relies on a brittle global step parameter here
        image_size = 512 if global_step == 875000 else 768
        return image_size

    return image_size


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.shave_segments
def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.conv_attn_to_linear
def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


def create_unet_diffusers_config(original_config, image_size: int):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    if (
        "unet_config" in original_config["model"]["params"]
        and original_config["model"]["params"]["unet_config"] is not None
    ):
        unet_params = original_config["model"]["params"]["unet_config"]["params"]
    else:
        unet_params = original_config["model"]["params"]["network_config"]["params"]

    vae_params = original_config["model"]["params"]["first_stage_config"]["params"]["ddconfig"]
    block_out_channels = [unet_params["model_channels"] * mult for mult in unet_params["channel_mult"]]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if resolution in unet_params["attention_resolutions"] else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if resolution in unet_params["attention_resolutions"] else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    if unet_params["transformer_depth"] is not None:
        transformer_layers_per_block = (
            unet_params["transformer_depth"]
            if isinstance(unet_params["transformer_depth"], int)
            else list(unet_params["transformer_depth"])
        )
    else:
        transformer_layers_per_block = 1

    vae_scale_factor = 2 ** (len(vae_params["ch_mult"]) - 1)

    head_dim = unet_params["num_heads"] if "num_heads" in unet_params else None
    use_linear_projection = (
        unet_params["use_linear_in_transformer"] if "use_linear_in_transformer" in unet_params else False
    )
    if use_linear_projection:
        # stable diffusion 2-base-512 and 2-768
        if head_dim is None:
            head_dim_mult = unet_params["model_channels"] // unet_params["num_head_channels"]
            head_dim = [head_dim_mult * c for c in list(unet_params["channel_mult"])]

    class_embed_type = None
    addition_embed_type = None
    addition_time_embed_dim = None
    projection_class_embeddings_input_dim = None
    context_dim = None

    if unet_params["context_dim"] is not None:
        context_dim = (
            unet_params["context_dim"]
            if isinstance(unet_params["context_dim"], int)
            else unet_params["context_dim"][0]
        )

    if "num_classes" in unet_params:
        if unet_params["num_classes"] == "sequential":
            if context_dim in [2048, 1280]:
                # SDXL
                addition_embed_type = "text_time"
                addition_time_embed_dim = 256
            else:
                class_embed_type = "projection"
            assert "adm_in_channels" in unet_params
            projection_class_embeddings_input_dim = unet_params["adm_in_channels"]

    config = {
        "sample_size": image_size // vae_scale_factor,
        "in_channels": unet_params["in_channels"],
        "down_block_types": tuple(down_block_types),
        "block_out_channels": tuple(block_out_channels),
        "layers_per_block": unet_params["num_res_blocks"],
        "cross_attention_dim": context_dim,
        "attention_head_dim": head_dim,
        "use_linear_projection": use_linear_projection,
        "class_embed_type": class_embed_type,
        "addition_embed_type": addition_embed_type,
        "addition_time_embed_dim": addition_time_embed_dim,
        "projection_class_embeddings_input_dim": projection_class_embeddings_input_dim,
        "transformer_layers_per_block": transformer_layers_per_block,
    }

    if "disable_self_attentions" in unet_params:
        config["only_cross_attention"] = unet_params["disable_self_attentions"]

    if "num_classes" in unet_params and isinstance(unet_params["num_classes"], int):
        config["num_class_embeds"] = unet_params["num_classes"]

    config["out_channels"] = unet_params["out_channels"]
    config["up_block_types"] = tuple(up_block_types)

    return config


def create_vae_diffusers_config(original_config, image_size: int):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    vae_params = original_config["model"]["params"]["first_stage_config"]["params"]["ddconfig"]

    block_out_channels = [vae_params["ch"] * mult for mult in vae_params["ch_mult"]]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    config = {
        "sample_size": image_size,
        "in_channels": vae_params["in_channels"],
        "out_channels": vae_params["out_ch"],
        "down_block_types": tuple(down_block_types),
        "up_block_types": tuple(up_block_types),
        "block_out_channels": tuple(block_out_channels),
        "latent_channels": vae_params["z_channels"],
        "layers_per_block": vae_params["num_res_blocks"],
    }

    return config


def update_unet_resnet_ldm_to_diffusers(ldm_keys, new_checkpoint, checkpoint, mapping=None):
    for ldm_key in ldm_keys:
        diffusers_key = (
            ldm_key.replace("in_layers.0", "norm1")
            .replace("in_layers.2", "conv1")
            .replace("out_layers.0", "norm2")
            .replace("out_layers.3", "conv2")
            .replace("emb_layers.1", "time_emb_proj")
            .replace("skip_connection", "conv_shortcut")
        )
        if mapping:
            diffusers_key = diffusers_key.replace(mapping["old"], mapping["new"])
        new_checkpoint[diffusers_key] = checkpoint.pop(ldm_key)


def update_unet_attention_ldm_to_diffusers(ldm_keys, new_checkpoint, checkpoint, mapping):
    for ldm_key in ldm_keys:
        diffusers_key = ldm_key.replace(mapping["old"], mapping["new"])
        new_checkpoint[diffusers_key] = checkpoint.pop(ldm_key)


def convert_ldm_unet_checkpoint(checkpoint, config, path=None, extract_ema=False, skip_extract_state_dict=False):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """

    if skip_extract_state_dict:
        unet_state_dict = checkpoint
    else:
        # extract state_dict for UNet
        unet_state_dict = {}
        keys = list(checkpoint.keys())

        unet_key = LDM_UNET_KEY

        # at least a 100 parameters have to start with `model_ema` in order for the checkpoint to be EMA
        if sum(k.startswith("model_ema") for k in keys) > 100 and extract_ema:
            logger.warning(f"Checkpoint {path} has both EMA and non-EMA weights.")
            logger.warning(
                "In this conversion only the EMA weights are extracted. If you want to instead extract the non-EMA"
                " weights (useful to continue fine-tuning), please make sure to remove the `--extract_ema` flag."
            )
            for key in keys:
                if key.startswith("model.diffusion_model"):
                    flat_ema_key = "model_ema." + "".join(key.split(".")[1:])
                    unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(flat_ema_key)
        else:
            if sum(k.startswith("model_ema") for k in keys) > 100:
                logger.warning(
                    "In this conversion only the non-EMA weights are extracted. If you want to instead extract the EMA"
                    " weights (usually better for inference), please make sure to add the `--extract_ema` flag."
                )

            for key in keys:
                if key.startswith(unet_key):
                    unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

    new_checkpoint = {}
    ldm_unet_keys = DIFFUSERS_TO_LDM_MAPPING["unet"]["layers"]
    for diffusers_key, ldm_key in ldm_unet_keys.items():
        new_checkpoint[diffusers_key] = unet_state_dict[ldm_key]

    if config["class_embed_type"] in ["timestep", "projection"]:
        class_embed_keys = DIFFUSERS_TO_LDM_MAPPING["unet"]["class_embed_type"]
        for diffusers_key, ldm_key in class_embed_keys.items():
            new_checkpoint[diffusers_key] = unet_state_dict[ldm_key]

    if config["addition_embed_type"] == "text_time":
        addition_embed_keys = DIFFUSERS_TO_LDM_MAPPING["unet"]["addition_embed_type"]
        for diffusers_key, ldm_key in addition_embed_keys.items():
            new_checkpoint[diffusers_key] = unet_state_dict[ldm_key]

    # Relevant to StableDiffusionUpscalePipeline
    if "num_class_embeds" in config:
        if (config["num_class_embeds"] is not None) and ("label_emb.weight" in unet_state_dict):
            new_checkpoint["class_embedding.weight"] = unet_state_dict["label_emb.weight"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    # Down blocks
    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        update_unet_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            unet_state_dict,
            {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"},
        )

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]
        if attentions:
            update_unet_attention_ldm_to_diffusers(
                attentions,
                new_checkpoint,
                unet_state_dict,
                {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"},
            )

    # Mid blocks
    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    update_unet_resnet_ldm_to_diffusers(
        resnet_0, new_checkpoint, unet_state_dict, mapping={"old": "middle_block.0", "new": "mid_block.resnets.0"}
    )
    update_unet_resnet_ldm_to_diffusers(
        resnet_1, new_checkpoint, unet_state_dict, mapping={"old": "middle_block.2", "new": "mid_block.resnets.1"}
    )
    update_unet_attention_ldm_to_diffusers(
        attentions, new_checkpoint, unet_state_dict, mapping={"old": "middle_block.1", "new": "mid_block.attentions.0"}
    )

    # Up Blocks
    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)

        resnets = [
            key for key in output_blocks[i] if f"output_blocks.{i}.0" in key and f"output_blocks.{i}.0.op" not in key
        ]
        update_unet_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            unet_state_dict,
            {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"},
        )

        attentions = [
            key for key in output_blocks[i] if f"output_blocks.{i}.1" in key and f"output_blocks.{i}.1.conv" not in key
        ]
        if attentions:
            update_unet_attention_ldm_to_diffusers(
                attentions,
                new_checkpoint,
                unet_state_dict,
                {"old": f"output_blocks.{i}.1", "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}"},
            )

        if f"output_blocks.{i}.1.conv.weight" in unet_state_dict:
            new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                f"output_blocks.{i}.1.conv.weight"
            ]
            new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                f"output_blocks.{i}.1.conv.bias"
            ]
        if f"output_blocks.{i}.2.conv.weight" in unet_state_dict:
            new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                f"output_blocks.{i}.2.conv.weight"
            ]
            new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                f"output_blocks.{i}.2.conv.bias"
            ]

    return new_checkpoint


def update_vae_resnet_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping):
    for ldm_key in keys:
        diffusers_key = ldm_key.replace(mapping["old"], mapping["new"]).replace("nin_shortcut", "conv_shortcut")
        new_checkpoint[diffusers_key] = checkpoint.pop(ldm_key)


def update_vae_attentions_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping):
    for ldm_key in keys:
        diffusers_key = (
            ldm_key.replace(mapping["old"], mapping["new"])
            .replace("norm.weight", "group_norm.weight")
            .replace("norm.bias", "group_norm.bias")
            .replace("q.weight", "to_q.weight")
            .replace("q.bias", "to_q.bias")
            .replace("k.weight", "to_k.weight")
            .replace("k.bias", "to_k.bias")
            .replace("v.weight", "to_v.weight")
            .replace("v.bias", "to_v.bias")
            .replace("proj_out.weight", "to_out.0.weight")
            .replace("proj_out.bias", "to_out.0.bias")
        )
        new_checkpoint[diffusers_key] = checkpoint.pop(ldm_key)

        # proj_attn.weight has to be converted from conv 1D to linear
        shape = new_checkpoint[diffusers_key].shape

        if len(shape) == 3:
            new_checkpoint[diffusers_key] = new_checkpoint[diffusers_key][:, :, 0]
        elif len(shape) == 4:
            new_checkpoint[diffusers_key] = new_checkpoint[diffusers_key][:, :, 0, 0]


def convert_ldm_vae_checkpoint(checkpoint, config):
    # extract state dict for VAE
    # remove the LDM_VAE_KEY prefix from the ldm checkpoint keys so that it is easier to map them to diffusers keys
    vae_state_dict = {}
    keys = list(checkpoint.keys())
    vae_key = LDM_VAE_KEY if any(k.startswith(LDM_VAE_KEY) for k in keys) else ""
    for key in keys:
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

    new_checkpoint = {}
    vae_diffusers_ldm_map = DIFFUSERS_TO_LDM_MAPPING["vae"]
    for diffusers_key, ldm_key in vae_diffusers_ldm_map.items():
        if ldm_key not in vae_state_dict:
            continue
        new_checkpoint[diffusers_key] = vae_state_dict[ldm_key]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len(config["down_block_types"])
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"},
        )
        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.bias"
            )

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"},
        )

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    update_vae_attentions_ldm_to_diffusers(
        mid_attentions, new_checkpoint, vae_state_dict, mapping={"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    )

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len(config["up_block_types"])
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"},
        )
        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"},
        )

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    update_vae_attentions_ldm_to_diffusers(
        mid_attentions, new_checkpoint, vae_state_dict, mapping={"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    )
    conv_attn_to_linear(new_checkpoint)

    return new_checkpoint


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.convert_ldm_bert_checkpoint
def convert_ldm_bert_checkpoint(checkpoint, config):
    def _copy_attn_layer(hf_attn_layer, pt_attn_layer):
        hf_attn_layer.q_proj.weight.data = pt_attn_layer.to_q.weight
        hf_attn_layer.k_proj.weight.data = pt_attn_layer.to_k.weight
        hf_attn_layer.v_proj.weight.data = pt_attn_layer.to_v.weight

        hf_attn_layer.out_proj.weight = pt_attn_layer.to_out.weight
        hf_attn_layer.out_proj.bias = pt_attn_layer.to_out.bias

    def _copy_linear(hf_linear, pt_linear):
        hf_linear.weight = pt_linear.weight
        hf_linear.bias = pt_linear.bias

    def _copy_layer(hf_layer, pt_layer):
        # copy layer norms
        _copy_linear(hf_layer.self_attn_layer_norm, pt_layer[0][0])
        _copy_linear(hf_layer.final_layer_norm, pt_layer[1][0])

        # copy attn
        _copy_attn_layer(hf_layer.self_attn, pt_layer[0][1])

        # copy MLP
        pt_mlp = pt_layer[1][1]
        _copy_linear(hf_layer.fc1, pt_mlp.net[0][0])
        _copy_linear(hf_layer.fc2, pt_mlp.net[2])

    def _copy_layers(hf_layers, pt_layers):
        for i, hf_layer in enumerate(hf_layers):
            if i != 0:
                i += i
            pt_layer = pt_layers[i : i + 2]
            _copy_layer(hf_layer, pt_layer)

    hf_model = LDMBertModel(config).eval()

    # copy  embeds
    hf_model.model.embed_tokens.weight = checkpoint.transformer.token_emb.weight
    hf_model.model.embed_positions.weight.data = checkpoint.transformer.pos_emb.emb.weight

    # copy layer norm
    _copy_linear(hf_model.model.layer_norm, checkpoint.transformer.norm)

    # copy hidden layers
    _copy_layers(hf_model.model.layers, checkpoint.transformer.attn_layers.layers)

    _copy_linear(hf_model.to_logits, checkpoint.transformer.to_logits)

    return hf_model


def convert_ldm_clip_checkpoint(checkpoint, local_files_only=False):
    try:
        config = CLIPTextConfig.from_pretrained(LDM_CLIP_CONFIG_NAME, local_files_only=local_files_only)
    except Exception:
        raise ValueError(
            f"With local_files_only set to {local_files_only}, you must first locally save the configuration in the following path: 'openai/clip-vit-large-patch14'."
        )

    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        text_model = CLIPTextModel(config)

    keys = list(checkpoint.keys())
    text_model_dict = {}

    remove_prefixes = LDM_CLIP_PREFIX_TO_REMOVE

    for key in keys:
        for prefix in remove_prefixes:
            if key.startswith(prefix):
                diffusers_key = key.replace(prefix, "")
                text_model_dict[diffusers_key] = checkpoint[key]

    if is_accelerate_available():
        for param_name, param in text_model_dict.items():
            set_module_tensor_to_device(text_model, param_name, "cpu", value=param)
    else:
        if not (hasattr(text_model, "embeddings") and hasattr(text_model.embeddings.position_ids)):
            text_model_dict.pop("text_model.embeddings.position_ids", None)

        text_model.load_state_dict(text_model_dict)

    return text_model


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.convert_open_clip_checkpoint
def convert_open_clip_checkpoint(
    checkpoint,
    config_name,
    prefix="cond_stage_model.model.",
    has_projection=False,
    local_files_only=False,
    **config_kwargs,
):
    try:
        config = CLIPTextConfig.from_pretrained(config_name, **config_kwargs, local_files_only=local_files_only)
    except Exception:
        raise ValueError(
            f"With local_files_only set to {local_files_only}, you must first locally save the configuration in the following path: '{config_name}'."
        )

    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        text_model = CLIPTextModelWithProjection(config) if has_projection else CLIPTextModel(config)

    keys = list(checkpoint.keys())

    keys_to_ignore = []
    if config_name == "stabilityai/stable-diffusion-2" and config.num_hidden_layers == 23:
        # make sure to remove all keys > 22
        keys_to_ignore += [k for k in keys if k.startswith("cond_stage_model.model.transformer.resblocks.23")]
        keys_to_ignore += ["cond_stage_model.model.text_projection"]

    text_model_dict = {}

    if prefix + "text_projection" in checkpoint:
        d_model = int(checkpoint[prefix + "text_projection"].shape[0])
    else:
        d_model = 1024

    text_model_dict["text_model.embeddings.position_ids"] = text_model.text_model.embeddings.get_buffer("position_ids")

    for key in keys:
        if key in keys_to_ignore:
            continue
        if key[len(prefix) :] in textenc_conversion_map:
            if key.endswith("text_projection"):
                value = checkpoint[key].T.contiguous()
            else:
                value = checkpoint[key]

            text_model_dict[textenc_conversion_map[key[len(prefix) :]]] = value

        if key.startswith(prefix + "transformer."):
            new_key = key[len(prefix + "transformer.") :]
            if new_key.endswith(".in_proj_weight"):
                new_key = new_key[: -len(".in_proj_weight")]
                new_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], new_key)
                text_model_dict[new_key + ".q_proj.weight"] = checkpoint[key][:d_model, :]
                text_model_dict[new_key + ".k_proj.weight"] = checkpoint[key][d_model : d_model * 2, :]
                text_model_dict[new_key + ".v_proj.weight"] = checkpoint[key][d_model * 2 :, :]
            elif new_key.endswith(".in_proj_bias"):
                new_key = new_key[: -len(".in_proj_bias")]
                new_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], new_key)
                text_model_dict[new_key + ".q_proj.bias"] = checkpoint[key][:d_model]
                text_model_dict[new_key + ".k_proj.bias"] = checkpoint[key][d_model : d_model * 2]
                text_model_dict[new_key + ".v_proj.bias"] = checkpoint[key][d_model * 2 :]
            else:
                new_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], new_key)

                text_model_dict[new_key] = checkpoint[key]

    if is_accelerate_available():
        for param_name, param in text_model_dict.items():
            set_module_tensor_to_device(text_model, param_name, "cpu", value=param)
    else:
        if not (hasattr(text_model, "embeddings") and hasattr(text_model.embeddings.position_ids)):
            text_model_dict.pop("text_model.embeddings.position_ids", None)

        text_model.load_state_dict(text_model_dict)

    return text_model


def stable_unclip_image_encoder(original_config, local_files_only=False):
    """
    Returns the image processor and clip image encoder for the img2img unclip pipeline.

    We currently know of two types of stable unclip models which separately use the clip and the openclip image
    encoders.
    """

    image_embedder_config = original_config["model"]["params"].embedder_config

    sd_clip_image_embedder_class = image_embedder_config.target
    sd_clip_image_embedder_class = sd_clip_image_embedder_class.split(".")[-1]

    if sd_clip_image_embedder_class == "ClipImageEmbedder":
        clip_model_name = image_embedder_config.params.model

        if clip_model_name == "ViT-L/14":
            feature_extractor = CLIPImageProcessor()
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14", local_files_only=local_files_only
            )
        else:
            raise NotImplementedError(f"Unknown CLIP checkpoint name in stable diffusion checkpoint {clip_model_name}")

    elif sd_clip_image_embedder_class == "FrozenOpenCLIPImageEmbedder":
        feature_extractor = CLIPImageProcessor()
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", local_files_only=local_files_only
        )
    else:
        raise NotImplementedError(
            f"Unknown CLIP image embedder class in stable diffusion checkpoint {sd_clip_image_embedder_class}"
        )

    return feature_extractor, image_encoder


def create_ldm_bert_config(original_config):
    bert_params = original_config["model"]["params"].cond_stage_config.params
    config = LDMBertConfig(
        d_model=bert_params.n_embed,
        encoder_layers=bert_params.n_layer,
        encoder_ffn_dim=bert_params.n_embed * 4,
    )
    return config


def create_unet_model(pipeline_class_name, original_config, checkpoint, checkpoint_path_or_dict, **kwargs):
    if "num_in_channels" in kwargs:
        num_in_channels = kwargs.get("num_in_channels")

    elif pipeline_class_name in [
        "StableDiffusionInpaintPipeline",
        "StableDiffusionXLInpaintPipeline",
        "StableDiffusionXLControlNetInpaintPipeline",
    ]:
        num_in_channels = 9

    elif pipeline_class_name == "StableDiffusionUpscalePipeline":
        num_in_channels = 7

    else:
        num_in_channels = 4

    image_size = determine_image_size(pipeline_class_name, original_config, checkpoint, **kwargs)

    upcast_attention = kwargs.get("upcast_attention", False)
    extract_ema = kwargs.get("extract_ema", False)

    unet_config = create_unet_diffusers_config(original_config, image_size=image_size)
    unet_config["in_channels"] = num_in_channels
    unet_config["upcast_attention"] = upcast_attention

    path = checkpoint_path_or_dict if isinstance(checkpoint_path_or_dict, str) else ""
    diffusers_format_unet_checkpoint = convert_ldm_unet_checkpoint(
        checkpoint, unet_config, path=path, extract_ema=extract_ema
    )
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        unet = UNet2DConditionModel(**unet_config)

    if is_accelerate_available():
        for param_name, param in diffusers_format_unet_checkpoint.items():
            set_module_tensor_to_device(unet, param_name, "cpu", value=param)
    else:
        unet.load_state_dict(diffusers_format_unet_checkpoint)

    return {"unet": unet}


def create_vae_model(pipeline_class_name, original_config, checkpoint, checkpoint_path_or_dict, **kwargs):
    image_size = determine_image_size(pipeline_class_name, original_config, checkpoint, **kwargs)

    vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
    diffusers_format_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)
    ctx = init_empty_weights if is_accelerate_available() else nullcontext

    with ctx():
        vae = AutoencoderKL(**vae_config)

    if is_accelerate_available():
        for param_name, param in diffusers_format_vae_checkpoint.items():
            set_module_tensor_to_device(vae, param_name, "cpu", value=param)
    else:
        vae.load_state_dict(diffusers_format_vae_checkpoint)

    return {"vae": vae}


def create_text_encoders_and_tokenizers(
    pipeline_class_name, original_config, checkpoint, checkpoint_path_or_dict, **kwargs
):
    model_type = infer_model_type(pipeline_class_name, original_config)
    local_files_only = kwargs.get("local_files_only", False)

    if model_type == "FrozenOpenCLIPEmbedder":
        config_name = "stabilityai/stable-diffusion-2"
        config_kwargs = {"subfolder": "text_encoder"}

        try:
            text_encoder = convert_open_clip_checkpoint(
                checkpoint, config_name, local_files_only=local_files_only, **config_kwargs
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                config_name, subfolder="tokenizer", local_files_only=local_files_only
            )
        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the text_encoder in the following path: '{config_name}'."
            )
        else:
            return {"text_encoder": text_encoder, "tokenizer": tokenizer}

    elif model_type == "FrozenCLIPEmbedder":
        try:
            config_name = "openai/clip-vit-large-patch14"
            text_encoder = convert_ldm_clip_checkpoint(checkpoint, local_files_only=local_files_only)
            tokenizer = CLIPTokenizer.from_pretrained(config_name, local_files_only=local_files_only)

        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the tokenizer in the following path: '{config_name}'."
            )
        else:
            return {"text_encoder": text_encoder, "tokenizer": tokenizer}

    elif model_type == "SDXL-Refiner":
        config_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        config_kwargs = {"projection_dim": 1280}
        prefix = "conditioner.embedders.0.model."

        try:
            tokenizer_2 = CLIPTokenizer.from_pretrained(config_name, pad_token="!", local_files_only=local_files_only)
            text_encoder_2 = convert_open_clip_checkpoint(
                checkpoint,
                config_name,
                prefix=prefix,
                has_projection=True,
                local_files_only=local_files_only,
                **config_kwargs,
            )
        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the text_encoder_2 and tokenizer_2 in the following path: {config_name} with `pad_token` set to '!'."
            )

        else:
            return {
                "text_encoder": None,
                "tokenizer": None,
                "tokenizer_2": tokenizer_2,
                "text_encoder_2": text_encoder_2,
            }

    elif model_type == "SDXL":
        try:
            config_name = "openai/clip-vit-large-patch14"
            tokenizer = CLIPTokenizer.from_pretrained(config_name, local_files_only=local_files_only)
            text_encoder = convert_ldm_clip_checkpoint(checkpoint, local_files_only=local_files_only)

        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the text_encoder and tokenizer in the following path: 'openai/clip-vit-large-patch14'."
            )

        try:
            config_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
            config_kwargs = {"projection_dim": 1280}
            prefix = "conditioner.embedders.1.model."
            tokenizer_2 = CLIPTokenizer.from_pretrained(config_name, pad_token="!", local_files_only=local_files_only)
            text_encoder_2 = convert_open_clip_checkpoint(
                checkpoint,
                config_name,
                prefix=prefix,
                has_projection=True,
                local_files_only=local_files_only,
                **config_kwargs,
            )
        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the text_encoder_2 and tokenizer_2 in the following path: {config_name} with `pad_token` set to '!'."
            )

        return {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "tokenizer_2": tokenizer_2,
            "text_encoder_2": text_encoder_2,
        }

    elif pipeline_class_name == "LDMTextToImagePipeline":
        text_config = create_ldm_bert_config(original_config)
        text_encoder = convert_ldm_bert_checkpoint(checkpoint, text_config)
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", local_files_only=local_files_only)

        return {"text_encoder": text_encoder, "tokenizer": tokenizer}

    return


def create_scheduler(pipeline_class_name, original_config, checkpoint, checkpoint_path_or_dict, **kwargs):
    scheduler_config = get_default_scheduler_config()
    model_type = infer_model_type(pipeline_class_name, original_config)

    scheduler_type = kwargs.get("scheduler_type", "ddim")
    prediction_type = kwargs.get("prediction_type", None)
    global_step = checkpoint["global_step"] if "global_step" in checkpoint else None

    num_train_timesteps = getattr(original_config["model"]["params"], "timesteps", None) or 1000
    scheduler_config["num_train_timesteps"] = num_train_timesteps

    if (
        "parameterization" in original_config["model"]["params"]
        and original_config["model"]["params"]["parameterization"] == "v"
    ):
        if prediction_type is None:
            # NOTE: For stable diffusion 2 base it is recommended to pass `prediction_type=="epsilon"`
            # as it relies on a brittle global step parameter here
            prediction_type = "epsilon" if global_step == 875000 else "v_prediction"

    else:
        prediction_type = prediction_type or "epsilon"

    scheduler_config["prediction_type"] = prediction_type

    if model_type in ["SDXL", "SDXL-Refiner"]:
        scheduler_type = "euler"

    else:
        beta_start = original_config["model"]["params"].get("linear_start", 0.02)
        beta_end = original_config["model"]["params"].get("linear_end", 0.085)
        scheduler_config["beta_start"] = beta_start
        scheduler_config["beta_end"] = beta_end
        scheduler_config["beta_schedule"] = "scaled_linear"
        scheduler_config["clip_sample"] = False
        scheduler_config["set_alpha_to_one"] = False

        scheduler_type = "ddim"

    if scheduler_type == "pndm":
        scheduler_config["skip_prk_steps"] = True
        scheduler = PNDMScheduler.from_config(scheduler_config)

    elif scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler.from_config(scheduler_config)

    elif scheduler_type == "heun":
        scheduler = HeunDiscreteScheduler.from_config(scheduler_config)

    elif scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler.from_config(scheduler_config)

    elif scheduler_type == "euler-ancestral":
        scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)

    elif scheduler_type == "dpm":
        scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)

    elif scheduler_type == "ddim":
        scheduler = DDIMScheduler.from_config(scheduler_config)

    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

    if pipeline_class_name == "StableDiffusionUpscalePipeline":
        scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", subfolder="scheduler")
        low_res_scheduler = DDPMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", subfolder="low_res_scheduler"
        )

        return {
            "scheduler": scheduler,
            "low_res_scheduler": low_res_scheduler,
        }

    return {"scheduler": scheduler}
