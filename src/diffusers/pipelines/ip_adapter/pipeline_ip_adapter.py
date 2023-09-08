from typing import Any, Dict, List, Optional, Union

import PIL
import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from ...models.attention_processor import AttnProcessor, AttnProcessor2_0
from ...utils import (
    DIFFUSERS_CACHE,
    HF_HUB_OFFLINE,
    _get_model_file,
)
from ..pipeline_utils import DiffusionPipeline
from .attention_processor import CNAttnProcessor, CNAttnProcessor2_0, IPAttnProcessor, IPAttnProcessor2_0
from .image_projection import ImageProjModel


class IPAdapterPipeline(DiffusionPipeline):
    def __init__(
        self,
        pipeline: DiffusionPipeline,
        image_projection: ImageProjModel,
        image_encoder: CLIPVisionModelWithProjection,
        image_processor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            pipeline=pipeline,
            image_encoder=image_encoder,
            image_processor=image_processor,
            image_projection=image_projection,
        )
        self._set_ip_adapter()

    def _set_ip_adapter(self):
        unet = self.pipeline.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_processor_class = (
                    AttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else AttnProcessor
                )
                attn_procs[name] = attn_processor_class()
            else:
                attn_processor_class = (
                    IPAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else IPAttnProcessor
                )
                attn_procs[name] = attn_processor_class(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0
                )
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipeline, "controlnet"):
            attn_processor_class = (
                CNAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else CNAttnProcessor
            )
            self.pipeline.controlnet.set_attn_processor(attn_processor_class())

    def load_ip_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        **kwargs,
    ):
        """
        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
        """
        # Load the main state dict first which has the LoRA layers for either of
        # UNet and text encoder or both.
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict
        ip_layers = torch.nn.ModuleList(
            [
                module if isinstance(module, nn.Module) else nn.Identity()
                for module in self.pipeline.unet.attn_processors.values()
            ]
        )
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    def set_scale(self, scale):
        for attn_processor in self.pipeline.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def _encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        (image_embeddings,) = self.image_encoder(image).image_embeds
        image_prompt_embeds = self.image_projection(image_embeddings)
        uncond_image_prompt_embeds = self.image_projection(torch.zeros_like(image_embeddings))

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        return image_prompt_embeds, uncond_image_prompt_embeds

    def to(self, *args, **kwargs):
        super(IPAdapterPipeline, self).to(*args, **kwargs)
        self.pipeline.to(*args, **kwargs)

    @torch.no_grad()
    def __call__(
        self,
        *args,
        example_image: Union[torch.FloatTensor, PIL.Image.Image],
        prompt: Union[str, List[str]] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        ip_adapter_scale: float = 1.0,
        **kwargs,
    ):
        # 0. Set IP Adapter scale
        self.set_scale(ip_adapter_scale)

        # 1. Define call parameters
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 2. Encode input image
        image_embeddings, uncond_image_embeddings = self._encode_image(example_image, device, num_images_per_prompt)

        # 3. Encode prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        prompt_embeds = torch.cat([prompt_embeds, image_embeddings], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_embeddings], dim=1)

        return self.pipeline(
            guidance_scale=guidance_scale,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            cross_attention_kwargs=cross_attention_kwargs,
            **kwargs,
        )
