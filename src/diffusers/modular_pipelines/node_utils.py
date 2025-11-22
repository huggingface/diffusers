import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch

from ..configuration_utils import ConfigMixin
from ..image_processor import PipelineImageInput
from .modular_pipeline import ModularPipelineBlocks, SequentialPipelineBlocks
from .modular_pipeline_utils import InputParam


logger = logging.getLogger(__name__)

# YiYi Notes: this is actually for SDXL, put it here for now
SDXL_INPUTS_SCHEMA = {
    "prompt": InputParam(
        "prompt", type_hint=Union[str, List[str]], description="The prompt or prompts to guide the image generation"
    ),
    "prompt_2": InputParam(
        "prompt_2",
        type_hint=Union[str, List[str]],
        description="The prompt or prompts to be sent to the tokenizer_2 and text_encoder_2",
    ),
    "negative_prompt": InputParam(
        "negative_prompt",
        type_hint=Union[str, List[str]],
        description="The prompt or prompts not to guide the image generation",
    ),
    "negative_prompt_2": InputParam(
        "negative_prompt_2",
        type_hint=Union[str, List[str]],
        description="The negative prompt or prompts for text_encoder_2",
    ),
    "cross_attention_kwargs": InputParam(
        "cross_attention_kwargs",
        type_hint=Optional[dict],
        description="Kwargs dictionary passed to the AttentionProcessor",
    ),
    "clip_skip": InputParam(
        "clip_skip", type_hint=Optional[int], description="Number of layers to skip in CLIP text encoder"
    ),
    "image": InputParam(
        "image",
        type_hint=PipelineImageInput,
        required=True,
        description="The image(s) to modify for img2img or inpainting",
    ),
    "mask_image": InputParam(
        "mask_image",
        type_hint=PipelineImageInput,
        required=True,
        description="Mask image for inpainting, white pixels will be repainted",
    ),
    "generator": InputParam(
        "generator",
        type_hint=Optional[Union[torch.Generator, List[torch.Generator]]],
        description="Generator(s) for deterministic generation",
    ),
    "height": InputParam("height", type_hint=Optional[int], description="Height in pixels of the generated image"),
    "width": InputParam("width", type_hint=Optional[int], description="Width in pixels of the generated image"),
    "num_images_per_prompt": InputParam(
        "num_images_per_prompt", type_hint=int, default=1, description="Number of images to generate per prompt"
    ),
    "num_inference_steps": InputParam(
        "num_inference_steps", type_hint=int, default=50, description="Number of denoising steps"
    ),
    "timesteps": InputParam(
        "timesteps", type_hint=Optional[torch.Tensor], description="Custom timesteps for the denoising process"
    ),
    "sigmas": InputParam(
        "sigmas", type_hint=Optional[torch.Tensor], description="Custom sigmas for the denoising process"
    ),
    "denoising_end": InputParam(
        "denoising_end",
        type_hint=Optional[float],
        description="Fraction of denoising process to complete before termination",
    ),
    # YiYi Notes: img2img defaults to 0.3, inpainting defaults to 0.9999
    "strength": InputParam(
        "strength", type_hint=float, default=0.3, description="How much to transform the reference image"
    ),
    "denoising_start": InputParam(
        "denoising_start", type_hint=Optional[float], description="Starting point of the denoising process"
    ),
    "latents": InputParam(
        "latents", type_hint=Optional[torch.Tensor], description="Pre-generated noisy latents for image generation"
    ),
    "padding_mask_crop": InputParam(
        "padding_mask_crop",
        type_hint=Optional[Tuple[int, int]],
        description="Size of margin in crop for image and mask",
    ),
    "original_size": InputParam(
        "original_size",
        type_hint=Optional[Tuple[int, int]],
        description="Original size of the image for SDXL's micro-conditioning",
    ),
    "target_size": InputParam(
        "target_size", type_hint=Optional[Tuple[int, int]], description="Target size for SDXL's micro-conditioning"
    ),
    "negative_original_size": InputParam(
        "negative_original_size",
        type_hint=Optional[Tuple[int, int]],
        description="Negative conditioning based on image resolution",
    ),
    "negative_target_size": InputParam(
        "negative_target_size",
        type_hint=Optional[Tuple[int, int]],
        description="Negative conditioning based on target resolution",
    ),
    "crops_coords_top_left": InputParam(
        "crops_coords_top_left",
        type_hint=Tuple[int, int],
        default=(0, 0),
        description="Top-left coordinates for SDXL's micro-conditioning",
    ),
    "negative_crops_coords_top_left": InputParam(
        "negative_crops_coords_top_left",
        type_hint=Tuple[int, int],
        default=(0, 0),
        description="Negative conditioning crop coordinates",
    ),
    "aesthetic_score": InputParam(
        "aesthetic_score", type_hint=float, default=6.0, description="Simulates aesthetic score of generated image"
    ),
    "negative_aesthetic_score": InputParam(
        "negative_aesthetic_score", type_hint=float, default=2.0, description="Simulates negative aesthetic score"
    ),
    "eta": InputParam("eta", type_hint=float, default=0.0, description="Parameter η in the DDIM paper"),
    "output_type": InputParam(
        "output_type", type_hint=str, default="pil", description="Output format (pil/tensor/np.array)"
    ),
    "ip_adapter_image": InputParam(
        "ip_adapter_image",
        type_hint=PipelineImageInput,
        required=True,
        description="Image(s) to be used as IP adapter",
    ),
    "control_image": InputParam(
        "control_image", type_hint=PipelineImageInput, required=True, description="ControlNet input condition"
    ),
    "control_guidance_start": InputParam(
        "control_guidance_start",
        type_hint=Union[float, List[float]],
        default=0.0,
        description="When ControlNet starts applying",
    ),
    "control_guidance_end": InputParam(
        "control_guidance_end",
        type_hint=Union[float, List[float]],
        default=1.0,
        description="When ControlNet stops applying",
    ),
    "controlnet_conditioning_scale": InputParam(
        "controlnet_conditioning_scale",
        type_hint=Union[float, List[float]],
        default=1.0,
        description="Scale factor for ControlNet outputs",
    ),
    "guess_mode": InputParam(
        "guess_mode",
        type_hint=bool,
        default=False,
        description="Enables ControlNet encoder to recognize input without prompts",
    ),
    "control_mode": InputParam(
        "control_mode", type_hint=List[int], required=True, description="Control mode for union controlnet"
    ),
}

SDXL_INTERMEDIATE_INPUTS_SCHEMA = {
    "prompt_embeds": InputParam(
        "prompt_embeds",
        type_hint=torch.Tensor,
        required=True,
        description="Text embeddings used to guide image generation",
    ),
    "negative_prompt_embeds": InputParam(
        "negative_prompt_embeds", type_hint=torch.Tensor, description="Negative text embeddings"
    ),
    "pooled_prompt_embeds": InputParam(
        "pooled_prompt_embeds", type_hint=torch.Tensor, required=True, description="Pooled text embeddings"
    ),
    "negative_pooled_prompt_embeds": InputParam(
        "negative_pooled_prompt_embeds", type_hint=torch.Tensor, description="Negative pooled text embeddings"
    ),
    "batch_size": InputParam("batch_size", type_hint=int, required=True, description="Number of prompts"),
    "dtype": InputParam("dtype", type_hint=torch.dtype, description="Data type of model tensor inputs"),
    "preprocess_kwargs": InputParam(
        "preprocess_kwargs", type_hint=Optional[dict], description="Kwargs for ImageProcessor"
    ),
    "latents": InputParam(
        "latents", type_hint=torch.Tensor, required=True, description="Initial latents for denoising process"
    ),
    "timesteps": InputParam("timesteps", type_hint=torch.Tensor, required=True, description="Timesteps for inference"),
    "num_inference_steps": InputParam(
        "num_inference_steps", type_hint=int, required=True, description="Number of denoising steps"
    ),
    "latent_timestep": InputParam(
        "latent_timestep", type_hint=torch.Tensor, required=True, description="Initial noise level timestep"
    ),
    "image_latents": InputParam(
        "image_latents", type_hint=torch.Tensor, required=True, description="Latents representing reference image"
    ),
    "mask": InputParam("mask", type_hint=torch.Tensor, required=True, description="Mask for inpainting"),
    "masked_image_latents": InputParam(
        "masked_image_latents", type_hint=torch.Tensor, description="Masked image latents for inpainting"
    ),
    "add_time_ids": InputParam(
        "add_time_ids", type_hint=torch.Tensor, required=True, description="Time ids for conditioning"
    ),
    "negative_add_time_ids": InputParam(
        "negative_add_time_ids", type_hint=torch.Tensor, description="Negative time ids"
    ),
    "timestep_cond": InputParam("timestep_cond", type_hint=torch.Tensor, description="Timestep conditioning for LCM"),
    "noise": InputParam("noise", type_hint=torch.Tensor, description="Noise added to image latents"),
    "crops_coords": InputParam("crops_coords", type_hint=Optional[Tuple[int]], description="Crop coordinates"),
    "ip_adapter_embeds": InputParam(
        "ip_adapter_embeds", type_hint=List[torch.Tensor], description="Image embeddings for IP-Adapter"
    ),
    "negative_ip_adapter_embeds": InputParam(
        "negative_ip_adapter_embeds",
        type_hint=List[torch.Tensor],
        description="Negative image embeddings for IP-Adapter",
    ),
    "images": InputParam(
        "images",
        type_hint=Union[List[PIL.Image.Image], List[torch.Tensor], List[np.array]],
        required=True,
        description="Generated images",
    ),
}

SDXL_PARAM_SCHEMA = {**SDXL_INPUTS_SCHEMA, **SDXL_INTERMEDIATE_INPUTS_SCHEMA}


DEFAULT_PARAM_MAPS = {
    "prompt": {
        "label": "Prompt",
        "type": "string",
        "default": "a bear sitting in a chair drinking a milkshake",
        "display": "textarea",
    },
    "negative_prompt": {
        "label": "Negative Prompt",
        "type": "string",
        "default": "deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
        "display": "textarea",
    },
    "num_inference_steps": {
        "label": "Steps",
        "type": "int",
        "default": 25,
        "min": 1,
        "max": 1000,
    },
    "seed": {
        "label": "Seed",
        "type": "int",
        "default": 0,
        "min": 0,
        "display": "random",
    },
    "width": {
        "label": "Width",
        "type": "int",
        "display": "text",
        "default": 1024,
        "min": 8,
        "max": 8192,
        "step": 8,
        "group": "dimensions",
    },
    "height": {
        "label": "Height",
        "type": "int",
        "display": "text",
        "default": 1024,
        "min": 8,
        "max": 8192,
        "step": 8,
        "group": "dimensions",
    },
    "images": {
        "label": "Images",
        "type": "image",
        "display": "output",
    },
    "image": {
        "label": "Image",
        "type": "image",
        "display": "input",
    },
}

DEFAULT_TYPE_MAPS = {
    "int": {
        "type": "int",
        "default": 0,
        "min": 0,
    },
    "float": {
        "type": "float",
        "default": 0.0,
        "min": 0.0,
    },
    "str": {
        "type": "string",
        "default": "",
    },
    "bool": {
        "type": "boolean",
        "default": False,
    },
    "image": {
        "type": "image",
    },
}

DEFAULT_MODEL_KEYS = ["unet", "vae", "text_encoder", "tokenizer", "controlnet", "transformer", "image_encoder"]
DEFAULT_CATEGORY = "Modular Diffusers"
DEFAULT_EXCLUDE_MODEL_KEYS = ["processor", "feature_extractor", "safety_checker"]
DEFAULT_PARAMS_GROUPS_KEYS = {
    "text_encoders": ["text_encoder", "tokenizer"],
    "ip_adapter_embeds": ["ip_adapter_embeds"],
    "prompt_embeddings": ["prompt_embeds"],
}


def get_group_name(name, group_params_keys=DEFAULT_PARAMS_GROUPS_KEYS):
    """
    Get the group name for a given parameter name, if not part of a group, return None e.g. "prompt_embeds" ->
    "text_embeds", "text_encoder" -> "text_encoders", "prompt" -> None
    """
    if name is None:
        return None
    for group_name, group_keys in group_params_keys.items():
        for group_key in group_keys:
            if group_key in name:
                return group_name
    return None


class ModularNode(ConfigMixin):
    """
    A ModularNode is a base class to build UI nodes using diffusers. Currently only supports Mellon. It is a wrapper
    around a ModularPipelineBlocks object.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    config_name = "node_config.json"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        trust_remote_code: Optional[bool] = None,
        **kwargs,
    ):
        blocks = ModularPipelineBlocks.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        )
        return cls(blocks, **kwargs)

    def __init__(self, blocks, category=DEFAULT_CATEGORY, label=None, **kwargs):
        self.blocks = blocks

        if label is None:
            label = self.blocks.__class__.__name__
        # blocks param name -> mellon param name
        self.name_mapping = {}

        input_params = {}
        # pass or create a default param dict for each input
        # e.g. for prompt,
        #       prompt = {
        #               "name": "text_input", # the name of the input in node definition, could be different from the input name in diffusers
        #               "label": "Prompt",
        #               "type": "string",
        #               "default": "a bear sitting in a chair drinking a milkshake",
        #               "display": "textarea"}
        # if type is not specified, it'll be a "custom" param of its own type
        # e.g. you can pass ModularNode(scheduler = {name :"scheduler"})
        #  it will get this spec in node definition {"scheduler": {"label": "Scheduler", "type": "scheduler", "display": "input"}}
        #  name can be a dict, in that case, it is part of a "dict" input in mellon nodes, e.g. text_encoder= {name: {"text_encoders": "text_encoder"}}
        inputs = self.blocks.inputs + self.blocks.intermediate_inputs
        for inp in inputs:
            param = kwargs.pop(inp.name, None)
            if param:
                # user can pass a param dict for all inputs, e.g. ModularNode(prompt = {...})
                input_params[inp.name] = param
                mellon_name = param.pop("name", inp.name)
                if mellon_name != inp.name:
                    self.name_mapping[inp.name] = mellon_name
                continue

            if inp.name not in DEFAULT_PARAM_MAPS and not inp.required and not get_group_name(inp.name):
                continue

            if inp.name in DEFAULT_PARAM_MAPS:
                # first check if it's in the default param map, if so, directly use that
                param = DEFAULT_PARAM_MAPS[inp.name].copy()
            elif get_group_name(inp.name):
                param = get_group_name(inp.name)
                if inp.name not in self.name_mapping:
                    self.name_mapping[inp.name] = param
            else:
                # if not, check if it's in the SDXL input schema, if so,
                # 1. use the type hint to determine the type
                # 2. use the default param dict for the type e.g. if "steps" is a "int" type, {"steps": {"type": "int", "default": 0, "min": 0}}
                if inp.type_hint is not None:
                    type_str = str(inp.type_hint).lower()
                else:
                    inp_spec = SDXL_PARAM_SCHEMA.get(inp.name, None)
                    type_str = str(inp_spec.type_hint).lower() if inp_spec else ""
                for type_key, type_param in DEFAULT_TYPE_MAPS.items():
                    if type_key in type_str:
                        param = type_param.copy()
                        param["label"] = inp.name
                        param["display"] = "input"
                        break
                else:
                    param = inp.name
            # add the param dict to the inp_params dict
            input_params[inp.name] = param

        component_params = {}
        for comp in self.blocks.expected_components:
            param = kwargs.pop(comp.name, None)
            if param:
                component_params[comp.name] = param
                mellon_name = param.pop("name", comp.name)
                if mellon_name != comp.name:
                    self.name_mapping[comp.name] = mellon_name
                continue

            to_exclude = False
            for exclude_key in DEFAULT_EXCLUDE_MODEL_KEYS:
                if exclude_key in comp.name:
                    to_exclude = True
                    break
            if to_exclude:
                continue

            if get_group_name(comp.name):
                param = get_group_name(comp.name)
                if comp.name not in self.name_mapping:
                    self.name_mapping[comp.name] = param
            elif comp.name in DEFAULT_MODEL_KEYS:
                param = {"label": comp.name, "type": "diffusers_auto_model", "display": "input"}
            else:
                param = comp.name
            # add the param dict to the model_params dict
            component_params[comp.name] = param

        output_params = {}
        if isinstance(self.blocks, SequentialPipelineBlocks):
            last_block_name = list(self.blocks.sub_blocks.keys())[-1]
            outputs = self.blocks.sub_blocks[last_block_name].intermediate_outputs
        else:
            outputs = self.blocks.intermediate_outputs

        for out in outputs:
            param = kwargs.pop(out.name, None)
            if param:
                output_params[out.name] = param
                mellon_name = param.pop("name", out.name)
                if mellon_name != out.name:
                    self.name_mapping[out.name] = mellon_name
                continue

            if out.name in DEFAULT_PARAM_MAPS:
                param = DEFAULT_PARAM_MAPS[out.name].copy()
                param["display"] = "output"
            else:
                group_name = get_group_name(out.name)
                if group_name:
                    param = group_name
                    if out.name not in self.name_mapping:
                        self.name_mapping[out.name] = param
                else:
                    param = out.name
            # add the param dict to the outputs dict
            output_params[out.name] = param

        if len(kwargs) > 0:
            logger.warning(f"Unused kwargs: {kwargs}")

        register_dict = {
            "category": category,
            "label": label,
            "input_params": input_params,
            "component_params": component_params,
            "output_params": output_params,
            "name_mapping": self.name_mapping,
        }
        self.register_to_config(**register_dict)

    def setup(self, components_manager, collection=None):
        self.pipeline = self.blocks.init_pipeline(components_manager=components_manager, collection=collection)
        self._components_manager = components_manager

    @property
    def mellon_config(self):
        return self._convert_to_mellon_config()

    def _convert_to_mellon_config(self):
        node = {}
        node["label"] = self.config.label
        node["category"] = self.config.category

        node_param = {}
        for inp_name, inp_param in self.config.input_params.items():
            if inp_name in self.name_mapping:
                mellon_name = self.name_mapping[inp_name]
            else:
                mellon_name = inp_name
            if isinstance(inp_param, str):
                param = {
                    "label": inp_param,
                    "type": inp_param,
                    "display": "input",
                }
            else:
                param = inp_param

            if mellon_name not in node_param:
                node_param[mellon_name] = param
            else:
                logger.debug(f"Input param {mellon_name} already exists in node_param, skipping {inp_name}")

        for comp_name, comp_param in self.config.component_params.items():
            if comp_name in self.name_mapping:
                mellon_name = self.name_mapping[comp_name]
            else:
                mellon_name = comp_name
            if isinstance(comp_param, str):
                param = {
                    "label": comp_param,
                    "type": comp_param,
                    "display": "input",
                }
            else:
                param = comp_param

            if mellon_name not in node_param:
                node_param[mellon_name] = param
            else:
                logger.debug(f"Component param {comp_param} already exists in node_param, skipping {comp_name}")

        for out_name, out_param in self.config.output_params.items():
            if out_name in self.name_mapping:
                mellon_name = self.name_mapping[out_name]
            else:
                mellon_name = out_name
            if isinstance(out_param, str):
                param = {
                    "label": out_param,
                    "type": out_param,
                    "display": "output",
                }
            else:
                param = out_param

            if mellon_name not in node_param:
                node_param[mellon_name] = param
            else:
                logger.debug(f"Output param {out_param} already exists in node_param, skipping {out_name}")
        node["params"] = node_param
        return node

    def save_mellon_config(self, file_path):
        """
        Save the Mellon configuration to a JSON file.

        Args:
            file_path (str or Path): Path where the JSON file will be saved

        Returns:
            Path: Path to the saved config file
        """
        file_path = Path(file_path)

        # Create directory if it doesn't exist
        os.makedirs(file_path.parent, exist_ok=True)

        # Create a combined dictionary with module definition and name mapping
        config = {"module": self.mellon_config, "name_mapping": self.name_mapping}

        # Save the config to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Mellon config and name mapping saved to {file_path}")

        return file_path

    @classmethod
    def load_mellon_config(cls, file_path):
        """
        Load a Mellon configuration from a JSON file.

        Args:
            file_path (str or Path): Path to the JSON file containing Mellon config

        Returns:
            dict: The loaded combined configuration containing 'module' and 'name_mapping'
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        logger.info(f"Mellon config loaded from {file_path}")

        return config

    def process_inputs(self, **kwargs):
        params_components = {}
        for comp_name, comp_param in self.config.component_params.items():
            logger.debug(f"component: {comp_name}")
            mellon_comp_name = self.name_mapping.get(comp_name, comp_name)
            if mellon_comp_name in kwargs:
                if isinstance(kwargs[mellon_comp_name], dict) and comp_name in kwargs[mellon_comp_name]:
                    comp = kwargs[mellon_comp_name].pop(comp_name)
                else:
                    comp = kwargs.pop(mellon_comp_name)
                if comp:
                    params_components[comp_name] = self._components_manager.get_one(comp["model_id"])

        params_run = {}
        for inp_name, inp_param in self.config.input_params.items():
            logger.debug(f"input: {inp_name}")
            mellon_inp_name = self.name_mapping.get(inp_name, inp_name)
            if mellon_inp_name in kwargs:
                if isinstance(kwargs[mellon_inp_name], dict) and inp_name in kwargs[mellon_inp_name]:
                    inp = kwargs[mellon_inp_name].pop(inp_name)
                else:
                    inp = kwargs.pop(mellon_inp_name)
                if inp is not None:
                    params_run[inp_name] = inp

        return_output_names = list(self.config.output_params.keys())

        return params_components, params_run, return_output_names

    def execute(self, **kwargs):
        params_components, params_run, return_output_names = self.process_inputs(**kwargs)

        self.pipeline.update_components(**params_components)
        output = self.pipeline(**params_run, output=return_output_names)
        return output
