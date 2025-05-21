from ..configuration_utils import ConfigMixin
from .modular_pipeline import SequentialPipelineBlocks
from .modular_pipeline_utils import InputParam, OutputParam
from ..image_processor import PipelineImageInput

from typing import Union, List, Optional, Tuple
import torch
import PIL
import numpy as np
import logging
logger = logging.getLogger(__name__)

# YiYi Notes: this is actually for SDXL, put it here for now
SDXL_INPUTS_SCHEMA = {
    "prompt": InputParam("prompt", type_hint=Union[str, List[str]], description="The prompt or prompts to guide the image generation"),
    "prompt_2": InputParam("prompt_2", type_hint=Union[str, List[str]], description="The prompt or prompts to be sent to the tokenizer_2 and text_encoder_2"),
    "negative_prompt": InputParam("negative_prompt", type_hint=Union[str, List[str]], description="The prompt or prompts not to guide the image generation"),
    "negative_prompt_2": InputParam("negative_prompt_2", type_hint=Union[str, List[str]], description="The negative prompt or prompts for text_encoder_2"),
    "cross_attention_kwargs": InputParam("cross_attention_kwargs", type_hint=Optional[dict], description="Kwargs dictionary passed to the AttentionProcessor"),
    "clip_skip": InputParam("clip_skip", type_hint=Optional[int], description="Number of layers to skip in CLIP text encoder"),
    "image": InputParam("image", type_hint=PipelineImageInput, required=True, description="The image(s) to modify for img2img or inpainting"),
    "mask_image": InputParam("mask_image", type_hint=PipelineImageInput, required=True, description="Mask image for inpainting, white pixels will be repainted"),
    "generator": InputParam("generator", type_hint=Optional[Union[torch.Generator, List[torch.Generator]]], description="Generator(s) for deterministic generation"),
    "height": InputParam("height", type_hint=Optional[int], description="Height in pixels of the generated image"),
    "width": InputParam("width", type_hint=Optional[int], description="Width in pixels of the generated image"),
    "num_images_per_prompt": InputParam("num_images_per_prompt", type_hint=int, default=1, description="Number of images to generate per prompt"),
    "num_inference_steps": InputParam("num_inference_steps", type_hint=int, default=50, description="Number of denoising steps"),
    "timesteps": InputParam("timesteps", type_hint=Optional[torch.Tensor], description="Custom timesteps for the denoising process"),
    "sigmas": InputParam("sigmas", type_hint=Optional[torch.Tensor], description="Custom sigmas for the denoising process"),
    "denoising_end": InputParam("denoising_end", type_hint=Optional[float], description="Fraction of denoising process to complete before termination"),
    # YiYi Notes: img2img defaults to 0.3, inpainting defaults to 0.9999
    "strength": InputParam("strength", type_hint=float, default=0.3, description="How much to transform the reference image"),
    "denoising_start": InputParam("denoising_start", type_hint=Optional[float], description="Starting point of the denoising process"),
    "latents": InputParam("latents", type_hint=Optional[torch.Tensor], description="Pre-generated noisy latents for image generation"),
    "padding_mask_crop": InputParam("padding_mask_crop", type_hint=Optional[Tuple[int, int]], description="Size of margin in crop for image and mask"),
    "original_size": InputParam("original_size", type_hint=Optional[Tuple[int, int]], description="Original size of the image for SDXL's micro-conditioning"),
    "target_size": InputParam("target_size", type_hint=Optional[Tuple[int, int]], description="Target size for SDXL's micro-conditioning"),
    "negative_original_size": InputParam("negative_original_size", type_hint=Optional[Tuple[int, int]], description="Negative conditioning based on image resolution"),
    "negative_target_size": InputParam("negative_target_size", type_hint=Optional[Tuple[int, int]], description="Negative conditioning based on target resolution"),
    "crops_coords_top_left": InputParam("crops_coords_top_left", type_hint=Tuple[int, int], default=(0, 0), description="Top-left coordinates for SDXL's micro-conditioning"),
    "negative_crops_coords_top_left": InputParam("negative_crops_coords_top_left", type_hint=Tuple[int, int], default=(0, 0), description="Negative conditioning crop coordinates"),
    "aesthetic_score": InputParam("aesthetic_score", type_hint=float, default=6.0, description="Simulates aesthetic score of generated image"),
    "negative_aesthetic_score": InputParam("negative_aesthetic_score", type_hint=float, default=2.0, description="Simulates negative aesthetic score"),
    "eta": InputParam("eta", type_hint=float, default=0.0, description="Parameter η in the DDIM paper"),
    "output_type": InputParam("output_type", type_hint=str, default="pil", description="Output format (pil/tensor/np.array)"),
    "ip_adapter_image": InputParam("ip_adapter_image", type_hint=PipelineImageInput, required=True, description="Image(s) to be used as IP adapter"),
    "control_image": InputParam("control_image", type_hint=PipelineImageInput, required=True, description="ControlNet input condition"),
    "control_guidance_start": InputParam("control_guidance_start", type_hint=Union[float, List[float]], default=0.0, description="When ControlNet starts applying"),
    "control_guidance_end": InputParam("control_guidance_end", type_hint=Union[float, List[float]], default=1.0, description="When ControlNet stops applying"),
    "controlnet_conditioning_scale": InputParam("controlnet_conditioning_scale", type_hint=Union[float, List[float]], default=1.0, description="Scale factor for ControlNet outputs"),
    "guess_mode": InputParam("guess_mode", type_hint=bool, default=False, description="Enables ControlNet encoder to recognize input without prompts"),
    "control_mode": InputParam("control_mode", type_hint=List[int], required=True, description="Control mode for union controlnet")
}

SDXL_INTERMEDIATE_INPUTS_SCHEMA = {
    "prompt_embeds": InputParam("prompt_embeds", type_hint=torch.Tensor, required=True, description="Text embeddings used to guide image generation"),
    "negative_prompt_embeds": InputParam("negative_prompt_embeds", type_hint=torch.Tensor, description="Negative text embeddings"),
    "pooled_prompt_embeds": InputParam("pooled_prompt_embeds", type_hint=torch.Tensor, required=True, description="Pooled text embeddings"),
    "negative_pooled_prompt_embeds": InputParam("negative_pooled_prompt_embeds", type_hint=torch.Tensor, description="Negative pooled text embeddings"),
    "batch_size": InputParam("batch_size", type_hint=int, required=True, description="Number of prompts"),
    "dtype": InputParam("dtype", type_hint=torch.dtype, description="Data type of model tensor inputs"),
    "preprocess_kwargs": InputParam("preprocess_kwargs", type_hint=Optional[dict], description="Kwargs for ImageProcessor"),
    "latents": InputParam("latents", type_hint=torch.Tensor, required=True, description="Initial latents for denoising process"),
    "timesteps": InputParam("timesteps", type_hint=torch.Tensor, required=True, description="Timesteps for inference"),
    "num_inference_steps": InputParam("num_inference_steps", type_hint=int, required=True, description="Number of denoising steps"),
    "latent_timestep": InputParam("latent_timestep", type_hint=torch.Tensor, required=True, description="Initial noise level timestep"),
    "image_latents": InputParam("image_latents", type_hint=torch.Tensor, required=True, description="Latents representing reference image"),
    "mask": InputParam("mask", type_hint=torch.Tensor, required=True, description="Mask for inpainting"),
    "masked_image_latents": InputParam("masked_image_latents", type_hint=torch.Tensor, description="Masked image latents for inpainting"),
    "add_time_ids": InputParam("add_time_ids", type_hint=torch.Tensor, required=True, description="Time ids for conditioning"),
    "negative_add_time_ids": InputParam("negative_add_time_ids", type_hint=torch.Tensor, description="Negative time ids"),
    "timestep_cond": InputParam("timestep_cond", type_hint=torch.Tensor, description="Timestep conditioning for LCM"),
    "noise": InputParam("noise", type_hint=torch.Tensor, description="Noise added to image latents"),
    "crops_coords": InputParam("crops_coords", type_hint=Optional[Tuple[int]], description="Crop coordinates"),
    "ip_adapter_embeds": InputParam("ip_adapter_embeds", type_hint=List[torch.Tensor], description="Image embeddings for IP-Adapter"),
    "negative_ip_adapter_embeds": InputParam("negative_ip_adapter_embeds", type_hint=List[torch.Tensor], description="Negative image embeddings for IP-Adapter"),
    "images": InputParam("images", type_hint=Union[List[PIL.Image.Image], List[torch.Tensor], List[np.array]], required=True, description="Generated images")
}

SDXL_INTERMEDIATE_OUTPUTS_SCHEMA = {
    "prompt_embeds": OutputParam("prompt_embeds", type_hint=torch.Tensor, description="Text embeddings used to guide image generation"),
    "negative_prompt_embeds": OutputParam("negative_prompt_embeds", type_hint=torch.Tensor, description="Negative text embeddings"),
    "pooled_prompt_embeds": OutputParam("pooled_prompt_embeds", type_hint=torch.Tensor, description="Pooled text embeddings"),
    "negative_pooled_prompt_embeds": OutputParam("negative_pooled_prompt_embeds", type_hint=torch.Tensor, description="Negative pooled text embeddings"),
    "batch_size": OutputParam("batch_size", type_hint=int, description="Number of prompts"),
    "dtype": OutputParam("dtype", type_hint=torch.dtype, description="Data type of model tensor inputs"),
    "image_latents": OutputParam("image_latents", type_hint=torch.Tensor, description="Latents representing reference image"),
    "mask": OutputParam("mask", type_hint=torch.Tensor, description="Mask for inpainting"),
    "masked_image_latents": OutputParam("masked_image_latents", type_hint=torch.Tensor, description="Masked image latents for inpainting"),
    "crops_coords": OutputParam("crops_coords", type_hint=Optional[Tuple[int]], description="Crop coordinates"),
    "timesteps": OutputParam("timesteps", type_hint=torch.Tensor, description="Timesteps for inference"),
    "num_inference_steps": OutputParam("num_inference_steps", type_hint=int, description="Number of denoising steps"),
    "latent_timestep": OutputParam("latent_timestep", type_hint=torch.Tensor, description="Initial noise level timestep"),
    "add_time_ids": OutputParam("add_time_ids", type_hint=torch.Tensor, description="Time ids for conditioning"),
    "negative_add_time_ids": OutputParam("negative_add_time_ids", type_hint=torch.Tensor, description="Negative time ids"),
    "timestep_cond": OutputParam("timestep_cond", type_hint=torch.Tensor, description="Timestep conditioning for LCM"),
    "latents": OutputParam("latents", type_hint=torch.Tensor, description="Denoised latents"),
    "noise": OutputParam("noise", type_hint=torch.Tensor, description="Noise added to image latents"),
    "ip_adapter_embeds": OutputParam("ip_adapter_embeds", type_hint=List[torch.Tensor], description="Image embeddings for IP-Adapter"),
    "negative_ip_adapter_embeds": OutputParam("negative_ip_adapter_embeds", type_hint=List[torch.Tensor], description="Negative image embeddings for IP-Adapter"),
    "images": OutputParam("images", type_hint=Union[List[PIL.Image.Image], List[torch.Tensor], List[np.array]], description="Generated images")
}

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

DEFAULT_TYPE_MAPS ={
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
    "text_embeds": ["prompt_embeds"],
}


def get_group_name(name, group_params_keys=DEFAULT_PARAMS_GROUPS_KEYS):
    """
    Get the group name for a given parameter name, if not part of a group, return None
    e.g. "prompt_embeds" -> "text_embeds", "text_encoder" -> "text_encoders", "prompt" -> None
    """
    for group_name, group_keys in group_params_keys.items():
        for group_key in group_keys:
            if group_key in name:
                return group_name
    return None

class MellonNode(ConfigMixin):

    block_class = None
    config_name = "node_config.json"


    def __init__(self, category=DEFAULT_CATEGORY, label=None, input_params=None, intermediate_params=None, component_params=None, output_params=None):
        self.blocks = self.block_class()

        if label is None:
            label = self.blocks.__class__.__name__
        
        expected_inputs = [inp.name for inp in self.blocks.inputs]
        expected_intermediates = [inp.name for inp in self.blocks.intermediates_inputs]
        expected_components = [comp.name for comp in self.blocks.expected_components]
        expected_outputs = [out.name for out in self.blocks.intermediates_outputs]

        if input_params is None:
            input_params ={}
            for inp in self.blocks.inputs:
                # create a param dict for each input e.g. for prompt, param = {"prompt": {"label": "Prompt", "type": "string", "default": "a bear sitting in a chair drinking a milkshake", "display": "textarea"}  }
                param = {}
                if inp.name in DEFAULT_PARAM_MAPS:
                    # first check if it's in the default param map, if so, directly use that
                    param[inp.name] = DEFAULT_PARAM_MAPS[inp.name]
                elif inp.required: 
                    group_name = get_group_name(inp.name)
                    if group_name:
                        param = group_name
                    else:
                        # if not, check if it's in the SDXL input schema, if so, 
                        # 1. use the type hint to determine the type
                        # 2. use the default param dict for the type e.g. if "steps" is a "int" type, {"steps": {"type": "int", "default": 0, "min": 0}}
                        inp_spec = SDXL_INPUTS_SCHEMA.get(inp.name, None)
                        if inp_spec:
                            type_str = str(inp_spec.type_hint).lower()
                            for type_key, type_param in DEFAULT_TYPE_MAPS.items():
                                if type_key in type_str:
                                    param[inp.name] = type_param
                                    param[inp.name]["display"] = "input"
                                    break
                        else:
                            param = inp.name
                # add the param dict to the inp_params dict
                if param:
                    input_params[inp.name] = param

        if intermediate_params is None:
            intermediate_params = {}
            for inp in self.blocks.intermediates_inputs:
                param = {}
                if inp.name in DEFAULT_PARAM_MAPS:
                    param[inp.name] = DEFAULT_PARAM_MAPS[inp.name]
                elif inp.required:
                    group_name = get_group_name(inp.name)
                    if group_name:
                        param = group_name
                    else:
                        inp_spec = SDXL_INTERMEDIATE_INPUTS_SCHEMA.get(inp.name, None)
                        if inp_spec:
                            type_str = str(inp_spec.type_hint).lower()
                        for type_key, type_param in DEFAULT_TYPE_MAPS.items():
                            if type_key in type_str:
                                    param[inp.name] = type_param
                                    param[inp.name]["display"] = "input"
                                    break
                        else:
                            param = inp.name
                # add the param dict to the intermediate_params dict
                if param:
                    intermediate_params[inp.name] = param

        if component_params is None:
            component_params = {}
            for comp in self.blocks.expected_components:
                to_exclude = False
                for exclude_key in DEFAULT_EXCLUDE_MODEL_KEYS:
                    if exclude_key in comp.name:
                        to_exclude = True
                        break
                if to_exclude:
                    continue
                
                param = {}
                group_name = get_group_name(comp.name)
                if group_name:
                    param = group_name
                elif comp.name in DEFAULT_MODEL_KEYS:
                    param[comp.name] = {
                        "label": comp.name,
                        "type": "diffusers_auto_model",
                        "display": "input",
                    }
                else:
                    param = comp.name
                # add the param dict to the model_params dict
                if param:
                    component_params[comp.name] = param

        if output_params is None:
            output_params = {}
            if isinstance(self.blocks, SequentialPipelineBlocks):
                last_block_name = list(self.blocks.blocks.keys())[-1]
                outputs = self.blocks.blocks[last_block_name].intermediates_outputs
            else:
                outputs = self.blocks.intermediates_outputs
        
            for out in outputs:
                param = {}
                if out.name in DEFAULT_PARAM_MAPS:
                    param[out.name] = DEFAULT_PARAM_MAPS[out.name]
                    param[out.name]["display"] = "output"
                else:
                    group_name = get_group_name(out.name)
                    if group_name:
                        param = group_name
                    else:
                        param = out.name
                # add the param dict to the outputs dict
                if param:
                    output_params[out.name] = param
        
        register_dict = {
            "category": category,
            "label": label,
            "input_params": input_params,
            "intermediate_params": intermediate_params,
            "component_params": component_params,
            "output_params": output_params,
        }
        self.register_to_config(**register_dict)
        











