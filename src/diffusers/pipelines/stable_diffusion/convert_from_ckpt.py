# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
"""Conversion script for the Stable Diffusion checkpoints."""

from contextlib import nullcontext
from io import BytesIO

import requests
import torch
import yaml
from transformers import (
    AutoFeatureExtractor,
    BertTokenizerFast,
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from ...loaders.conversion import get_conversion
from ...loaders.conversion.checkpoint import convert_component_checkpoint
from ...models import (
    AutoencoderKL,
    ControlNetModel,
    PriorTransformer,
    UNet2DConditionModel,
)
from ...schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UnCLIPScheduler,
)
from ...utils import is_accelerate_available, logging
from ...utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from ...utils.torch_utils import get_device
from ..deprecated.paint_by_example import PaintByExampleImageEncoder
from ..latent_diffusion.pipeline_latent_diffusion import LDMBertConfig, LDMBertModel
from ..pipeline_utils import DiffusionPipeline
from .safety_checker import StableDiffusionSafetyChecker
from .stable_unclip_image_normalizer import StableUnCLIPImageNormalizer


if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def create_unet_diffusers_config(original_config, image_size: int, controlnet=False):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    if controlnet:
        unet_params = original_config["model"]["params"]["control_stage_config"]["params"]
    else:
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

    if controlnet:
        config["conditioning_channels"] = unet_params["hint_channels"]
    else:
        config["out_channels"] = unet_params["out_channels"]
        config["up_block_types"] = tuple(up_block_types)

    return config


def create_vae_diffusers_config(original_config, image_size: int):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    vae_params = original_config["model"]["params"]["first_stage_config"]["params"]["ddconfig"]
    _ = original_config["model"]["params"]["first_stage_config"]["params"]["embed_dim"]

    block_out_channels = [vae_params["ch"] * mult for mult in vae_params["ch_mult"]]
    down_block_types = [
        "DownEncoderBlock2D" if image_size // 2**i not in vae_params["attn_resolutions"] else "AttnDownEncoderBlock2D"
        for i, _ in enumerate(block_out_channels)
    ]
    up_block_types = [
        "UpDecoderBlock2D" if image_size // 2**i not in vae_params["attn_resolutions"] else "AttnUpDecoderBlock2D"
        for i, _ in enumerate(block_out_channels)
    ][::-1]

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


def create_diffusers_schedular(original_config):
    schedular = DDIMScheduler(
        num_train_timesteps=original_config["model"]["params"]["timesteps"],
        beta_start=original_config["model"]["params"]["linear_start"],
        beta_end=original_config["model"]["params"]["linear_end"],
        beta_schedule="scaled_linear",
    )
    return schedular


def create_ldm_bert_config(original_config):
    bert_params = original_config["model"]["params"]["cond_stage_config"]["params"]
    config = LDMBertConfig(
        d_model=bert_params.n_embed,
        encoder_layers=bert_params.n_layer,
        encoder_ffn_dim=bert_params.n_embed * 4,
    )
    return config


def convert_ldm_unet_checkpoint(
    checkpoint, config, path=None, extract_ema=False, controlnet=False, skip_extract_state_dict=False
):
    cls = "ControlNetModel" if controlnet else "UNet2DConditionModel"
    if skip_extract_state_dict:
        return get_conversion(cls, config).to_diffusers(checkpoint)
    return convert_component_checkpoint(checkpoint, config, cls, extract_ema=extract_ema)


def convert_ldm_vae_checkpoint(checkpoint, config):
    return convert_component_checkpoint(checkpoint, config, "AutoencoderKL")


def convert_ldm_bert_checkpoint(checkpoint, config):
    state = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint
    if any(key.startswith("cond_stage_model.") for key in state):
        state = {
            key.removeprefix("cond_stage_model."): value
            for key, value in state.items()
            if key.startswith("cond_stage_model.")
        }
    model = LDMBertModel(config).eval()
    model.load_state_dict(get_conversion("LDMBertModel", config.to_dict()).to_diffusers(state), strict=True)
    return model


def convert_ldm_clip_checkpoint(checkpoint, local_files_only=False, text_encoder=None):
    if text_encoder is None:
        config_name = "openai/clip-vit-large-patch14"
        try:
            config = CLIPTextConfig.from_pretrained(config_name, local_files_only=local_files_only)
        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the configuration in the following path: 'openai/clip-vit-large-patch14'."
            )

        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            text_model = CLIPTextModel(config)
    else:
        text_model = text_encoder

    keys = list(checkpoint.keys())

    text_model_dict = {}

    remove_prefixes = ["cond_stage_model.transformer", "conditioner.embedders.0.transformer"]

    for key in keys:
        for prefix in remove_prefixes:
            if key.startswith(prefix):
                text_model_dict[key[len(prefix + ".") :]] = checkpoint[key]

    if is_accelerate_available():
        for param_name, param in text_model_dict.items():
            set_module_tensor_to_device(text_model, param_name, "cpu", value=param)
    else:
        if not (hasattr(text_model, "embeddings") and hasattr(text_model.embeddings.position_ids)):
            text_model_dict.pop("text_model.embeddings.position_ids", None)

        text_model.load_state_dict(text_model_dict)

    return text_model


def convert_paint_by_example_checkpoint(checkpoint, local_files_only=False):
    config = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14", local_files_only=local_files_only)
    model = PaintByExampleImageEncoder(config)
    state = {
        key: value
        for key, value in checkpoint.items()
        if key.startswith(("cond_stage_model.", "proj_out.")) or key == "learnable_vector"
    }
    model.load_state_dict(
        get_conversion("PaintByExampleImageEncoder", config.to_dict()).to_diffusers(state), strict=True
    )
    return model


def convert_open_clip_checkpoint(
    checkpoint,
    config_name,
    prefix="cond_stage_model.model.",
    has_projection=False,
    local_files_only=False,
    **config_kwargs,
):
    # text_model = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="text_encoder")
    # text_model = CLIPTextModelWithProjection.from_pretrained(
    #    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", projection_dim=1280
    # )
    try:
        config = CLIPTextConfig.from_pretrained(config_name, **config_kwargs, local_files_only=local_files_only)
    except Exception:
        raise ValueError(
            f"With local_files_only set to {local_files_only}, you must first locally save the configuration in the following path: '{config_name}'."
        )

    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        text_model = CLIPTextModelWithProjection(config) if has_projection else CLIPTextModel(config)

    from ...loaders.single_file_utils import convert_open_clip_checkpoint as convert_text_component

    state = convert_text_component(text_model, checkpoint, prefix=prefix)
    text_model.load_state_dict(state, strict=True, assign=True)
    return text_model


def stable_unclip_image_encoder(original_config, local_files_only=False):
    """
    Returns the image processor and clip image encoder for the img2img unclip pipeline.

    We currently know of two types of stable unclip models which separately use the clip and the openclip image
    encoders.
    """

    image_embedder_config = original_config["model"]["params"]["embedder_config"]

    sd_clip_image_embedder_class = image_embedder_config["target"]
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


def stable_unclip_image_noising_components(
    original_config, clip_stats_path: str | None = None, device: str | None = None
):
    """
    Returns the noising components for the img2img and txt2img unclip pipelines.

    Converts the stability noise augmentor into
    1. a `StableUnCLIPImageNormalizer` for holding the CLIP stats
    2. a `DDPMScheduler` for holding the noise schedule

    If the noise augmentor config specifies a clip stats path, the `clip_stats_path` must be provided.
    """
    noise_aug_config = original_config["model"]["params"]["noise_aug_config"]
    noise_aug_class = noise_aug_config["target"]
    noise_aug_class = noise_aug_class.split(".")[-1]

    if noise_aug_class == "CLIPEmbeddingNoiseAugmentation":
        noise_aug_config = noise_aug_config.params
        embedding_dim = noise_aug_config.timestep_dim
        max_noise_level = noise_aug_config.noise_schedule_config.timesteps
        beta_schedule = noise_aug_config.noise_schedule_config.beta_schedule

        image_normalizer = StableUnCLIPImageNormalizer(embedding_dim=embedding_dim)
        image_noising_scheduler = DDPMScheduler(num_train_timesteps=max_noise_level, beta_schedule=beta_schedule)

        if "clip_stats_path" in noise_aug_config:
            if clip_stats_path is None:
                raise ValueError("This stable unclip config requires a `clip_stats_path`")

            clip_mean, clip_std = torch.load(clip_stats_path, map_location=device)
            clip_mean = clip_mean[None, :]
            clip_std = clip_std[None, :]

            clip_stats_state_dict = {
                "mean": clip_mean,
                "std": clip_std,
            }

            image_normalizer.load_state_dict(clip_stats_state_dict)
    else:
        raise NotImplementedError(f"Unknown noise augmentor class: {noise_aug_class}")

    return image_normalizer, image_noising_scheduler


def convert_controlnet_checkpoint(
    checkpoint,
    original_config,
    checkpoint_path,
    image_size,
    upcast_attention,
    extract_ema,
    use_linear_projection=None,
    cross_attention_dim=None,
):
    ctrlnet_config = create_unet_diffusers_config(original_config, image_size=image_size, controlnet=True)
    ctrlnet_config["upcast_attention"] = upcast_attention

    ctrlnet_config.pop("sample_size")

    if use_linear_projection is not None:
        ctrlnet_config["use_linear_projection"] = use_linear_projection

    if cross_attention_dim is not None:
        ctrlnet_config["cross_attention_dim"] = cross_attention_dim

    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        controlnet = ControlNetModel(**ctrlnet_config)

    # Some controlnet ckpt files are distributed independently from the rest of the
    # model components i.e. https://huggingface.co/thibaud/controlnet-sd21/
    if "time_embed.0.weight" in checkpoint:
        skip_extract_state_dict = True
    else:
        skip_extract_state_dict = False

    converted_ctrl_checkpoint = convert_ldm_unet_checkpoint(
        checkpoint,
        ctrlnet_config,
        path=checkpoint_path,
        extract_ema=extract_ema,
        controlnet=True,
        skip_extract_state_dict=skip_extract_state_dict,
    )

    if is_accelerate_available():
        for param_name, param in converted_ctrl_checkpoint.items():
            set_module_tensor_to_device(controlnet, param_name, "cpu", value=param)
    else:
        controlnet.load_state_dict(converted_ctrl_checkpoint)

    return controlnet


def download_from_original_stable_diffusion_ckpt(
    checkpoint_path_or_dict: str | dict[str, torch.Tensor],
    original_config_file: str = None,
    image_size: int | None = None,
    prediction_type: str = None,
    model_type: str = None,
    extract_ema: bool = False,
    scheduler_type: str = "pndm",
    num_in_channels: int | None = None,
    upcast_attention: bool | None = None,
    device: str = None,
    from_safetensors: bool = False,
    stable_unclip: str | None = None,
    stable_unclip_prior: str | None = None,
    clip_stats_path: str | None = None,
    controlnet: bool | None = None,
    adapter: bool | None = None,
    load_safety_checker: bool = True,
    safety_checker: StableDiffusionSafetyChecker | None = None,
    feature_extractor: AutoFeatureExtractor | None = None,
    pipeline_class: DiffusionPipeline = None,
    local_files_only=False,
    vae_path=None,
    vae=None,
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,
    config_files=None,
) -> DiffusionPipeline:
    """
    Load a Stable Diffusion pipeline object from a CompVis-style `.ckpt`/`.safetensors` file and (ideally) a `.yaml`
    config file.

    Although many of the arguments can be automatically inferred, some of these rely on brittle checks against the
    global step count, which will likely fail for models that have undergone further fine-tuning. Therefore, it is
    recommended that you override the default values and/or supply an `original_config_file` wherever possible.

    Args:
        checkpoint_path_or_dict (`str` or `dict`): Path to `.ckpt` file, or the state dict.
        original_config_file (`str`):
            Path to `.yaml` config file corresponding to the original architecture. If `None`, will be automatically
            inferred by looking for a key that only exists in SD2.0 models.
        image_size (`int`, *optional*, defaults to 512):
            The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable Diffusion v2
            Base. Use 768 for Stable Diffusion v2.
        prediction_type (`str`, *optional*):
            The prediction type that the model was trained on. Use `'epsilon'` for Stable Diffusion v1.X and Stable
            Diffusion v2 Base. Use `'v_prediction'` for Stable Diffusion v2.
        num_in_channels (`int`, *optional*, defaults to None):
            The number of input channels. If `None`, it will be automatically inferred.
        scheduler_type (`str`, *optional*, defaults to 'pndm'):
            Type of scheduler to use. Should be one of `["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm",
            "ddim"]`.
        model_type (`str`, *optional*, defaults to `None`):
            The pipeline type. `None` to automatically infer, or one of `["FrozenOpenCLIPEmbedder",
            "FrozenCLIPEmbedder", "PaintByExample"]`.
        is_img2img (`bool`, *optional*, defaults to `False`):
            Whether the model should be loaded as an img2img pipeline.
        extract_ema (`bool`, *optional*, defaults to `False`): Only relevant for
            checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights or not. Defaults to
            `False`. Pass `True` to extract the EMA weights. EMA weights usually yield higher quality images for
            inference. Non-EMA weights are usually better to continue fine-tuning.
        upcast_attention (`bool`, *optional*, defaults to `None`):
            Whether the attention computation should always be upcasted. This is necessary when running stable
            diffusion 2.1.
        device (`str`, *optional*, defaults to `None`):
            The device to use. Pass `None` to determine automatically.
        from_safetensors (`str`, *optional*, defaults to `False`):
            If `checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of PyTorch.
        load_safety_checker (`bool`, *optional*, defaults to `True`):
            Whether to load the safety checker or not. Defaults to `True`.
        safety_checker (`StableDiffusionSafetyChecker`, *optional*, defaults to `None`):
            Safety checker to use. If this parameter is `None`, the function will load a new instance of
            [StableDiffusionSafetyChecker] by itself, if needed.
        feature_extractor (`AutoFeatureExtractor`, *optional*, defaults to `None`):
            Feature extractor to use. If this parameter is `None`, the function will load a new instance of
            [AutoFeatureExtractor] by itself, if needed.
        pipeline_class (`str`, *optional*, defaults to `None`):
            The pipeline class to use. Pass `None` to determine automatically.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether or not to only look at local files (i.e., do not try to download the model).
        vae (`AutoencoderKL`, *optional*, defaults to `None`):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations. If
            this parameter is `None`, the function will load a new instance of [CLIP] by itself, if needed.
        text_encoder (`CLIPTextModel`, *optional*, defaults to `None`):
            An instance of [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel)
            to use, specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
            variant. If this parameter is `None`, the function will load a new instance of [CLIP] by itself, if needed.
        tokenizer (`CLIPTokenizer`, *optional*, defaults to `None`):
            An instance of
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer)
            to use. If this parameter is `None`, the function will load a new instance of [CLIPTokenizer] by itself, if
            needed.
        config_files (`dict[str, str]`, *optional*, defaults to `None`):
            A dictionary mapping from config file names to their contents. If this parameter is `None`, the function
            will load the config files by itself, if needed. Valid keys are:
                - `v1`: Config file for Stable Diffusion v1
                - `v2`: Config file for Stable Diffusion v2
                - `xl`: Config file for Stable Diffusion XL
                - `xl_refiner`: Config file for Stable Diffusion XL Refiner
        return: A StableDiffusionPipeline object representing the passed-in `.ckpt`/`.safetensors` file.
    """

    # import pipelines here to avoid circular import error when using from_single_file method
    from diffusers import (
        LDMTextToImagePipeline,
        PaintByExamplePipeline,
        StableDiffusionControlNetPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionPipeline,
        StableDiffusionUpscalePipeline,
        StableDiffusionXLControlNetInpaintPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLPipeline,
        StableUnCLIPImg2ImgPipeline,
        StableUnCLIPPipeline,
    )

    if prediction_type == "v-prediction":
        prediction_type = "v_prediction"

    if isinstance(checkpoint_path_or_dict, str):
        if from_safetensors:
            from safetensors.torch import load_file as safe_load

            checkpoint = safe_load(checkpoint_path_or_dict, device="cpu")
        else:
            if device is None:
                device = get_device()
                checkpoint = torch.load(checkpoint_path_or_dict, map_location=device)
            else:
                checkpoint = torch.load(checkpoint_path_or_dict, map_location=device)
    elif isinstance(checkpoint_path_or_dict, dict):
        checkpoint = checkpoint_path_or_dict

    # Sometimes models don't have the global_step item
    if "global_step" in checkpoint:
        global_step = checkpoint["global_step"]
    else:
        logger.debug("global_step key not found in model")
        global_step = None

    # NOTE: this while loop isn't great but this controlnet checkpoint has one additional
    # "state_dict" key https://huggingface.co/thibaud/controlnet-canny-sd21
    while "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if original_config_file is None:
        key_name_v2_1 = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
        key_name_sd_xl_base = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias"
        key_name_sd_xl_refiner = "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias"
        is_upscale = pipeline_class == StableDiffusionUpscalePipeline

        config_url = None

        # model_type = "v1"
        if config_files is not None and "v1" in config_files:
            original_config_file = config_files["v1"]
        else:
            config_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"

        if key_name_v2_1 in checkpoint and checkpoint[key_name_v2_1].shape[-1] == 1024:
            # model_type = "v2"
            if config_files is not None and "v2" in config_files:
                original_config_file = config_files["v2"]
            else:
                config_url = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml"
            if global_step == 110000:
                # v2.1 needs to upcast attention
                upcast_attention = True
        elif key_name_sd_xl_base in checkpoint:
            # only base xl has two text embedders
            if config_files is not None and "xl" in config_files:
                original_config_file = config_files["xl"]
            else:
                config_url = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"
        elif key_name_sd_xl_refiner in checkpoint:
            # only refiner xl has embedder and one text embedders
            if config_files is not None and "xl_refiner" in config_files:
                original_config_file = config_files["xl_refiner"]
            else:
                config_url = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_refiner.yaml"

        if is_upscale:
            config_url = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/x4-upscaling.yaml"

        if config_url is not None:
            original_config_file = BytesIO(requests.get(config_url, timeout=DIFFUSERS_REQUEST_TIMEOUT).content)
        else:
            with open(original_config_file, "r") as f:
                original_config_file = f.read()
    else:
        with open(original_config_file, "r") as f:
            original_config_file = f.read()

    original_config = yaml.safe_load(original_config_file)

    # Convert the text model.
    if (
        model_type is None
        and "cond_stage_config" in original_config["model"]["params"]
        and original_config["model"]["params"]["cond_stage_config"] is not None
    ):
        model_type = original_config["model"]["params"]["cond_stage_config"]["target"].split(".")[-1]
        logger.debug(f"no `model_type` given, `model_type` inferred as: {model_type}")
    elif model_type is None and original_config["model"]["params"]["network_config"] is not None:
        if original_config["model"]["params"]["network_config"]["params"]["context_dim"] == 2048:
            model_type = "SDXL"
        else:
            model_type = "SDXL-Refiner"
        if image_size is None:
            image_size = 1024

    if pipeline_class is None:
        # Check if we have a SDXL or SD model and initialize default pipeline
        if model_type not in ["SDXL", "SDXL-Refiner"]:
            pipeline_class = StableDiffusionPipeline if not controlnet else StableDiffusionControlNetPipeline
        else:
            pipeline_class = StableDiffusionXLPipeline if model_type == "SDXL" else StableDiffusionXLImg2ImgPipeline

    if num_in_channels is None and pipeline_class in [
        StableDiffusionInpaintPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLControlNetInpaintPipeline,
    ]:
        num_in_channels = 9
    if num_in_channels is None and pipeline_class == StableDiffusionUpscalePipeline:
        num_in_channels = 7
    elif num_in_channels is None:
        num_in_channels = 4

    if "unet_config" in original_config["model"]["params"]:
        original_config["model"]["params"]["unet_config"]["params"]["in_channels"] = num_in_channels
    elif "network_config" in original_config["model"]["params"]:
        original_config["model"]["params"]["network_config"]["params"]["in_channels"] = num_in_channels

    if (
        "parameterization" in original_config["model"]["params"]
        and original_config["model"]["params"]["parameterization"] == "v"
    ):
        if prediction_type is None:
            # NOTE: For stable diffusion 2 base it is recommended to pass `prediction_type=="epsilon"`
            # as it relies on a brittle global step parameter here
            prediction_type = "epsilon" if global_step == 875000 else "v_prediction"
        if image_size is None:
            # NOTE: For stable diffusion 2 base one has to pass `image_size==512`
            # as it relies on a brittle global step parameter here
            image_size = 512 if global_step == 875000 else 768
    else:
        if prediction_type is None:
            prediction_type = "epsilon"
        if image_size is None:
            image_size = 512

    if controlnet is None and "control_stage_config" in original_config["model"]["params"]:
        path = checkpoint_path_or_dict if isinstance(checkpoint_path_or_dict, str) else ""
        controlnet = convert_controlnet_checkpoint(
            checkpoint, original_config, path, image_size, upcast_attention, extract_ema
        )

    if "timesteps" in original_config["model"]["params"]:
        num_train_timesteps = original_config["model"]["params"]["timesteps"]
    else:
        num_train_timesteps = 1000

    if model_type in ["SDXL", "SDXL-Refiner"]:
        scheduler_dict = {
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "interpolation_type": "linear",
            "num_train_timesteps": num_train_timesteps,
            "prediction_type": "epsilon",
            "sample_max_value": 1.0,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "timestep_spacing": "leading",
        }
        scheduler = EulerDiscreteScheduler.from_config(scheduler_dict)
        scheduler_type = "euler"
    else:
        if "linear_start" in original_config["model"]["params"]:
            beta_start = original_config["model"]["params"]["linear_start"]
        else:
            beta_start = 0.02

        if "linear_end" in original_config["model"]["params"]:
            beta_end = original_config["model"]["params"]["linear_end"]
        else:
            beta_end = 0.085
        scheduler = DDIMScheduler(
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            beta_start=beta_start,
            num_train_timesteps=num_train_timesteps,
            steps_offset=1,
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type=prediction_type,
        )
    # make sure scheduler works correctly with DDIM
    scheduler.register_to_config(clip_sample=False)

    if scheduler_type == "pndm":
        config = dict(scheduler.config)
        config["skip_prk_steps"] = True
        scheduler = PNDMScheduler.from_config(config)
    elif scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "heun":
        scheduler = HeunDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "euler-ancestral":
        scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "dpm":
        scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    elif scheduler_type == "ddim":
        scheduler = scheduler
    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

    if pipeline_class == StableDiffusionUpscalePipeline:
        image_size = original_config["model"]["params"]["unet_config"]["params"]["image_size"]

    # Convert the UNet2DConditionModel model.
    unet_config = create_unet_diffusers_config(original_config, image_size=image_size)
    unet_config["upcast_attention"] = upcast_attention

    path = checkpoint_path_or_dict if isinstance(checkpoint_path_or_dict, str) else ""
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
        checkpoint, unet_config, path=path, extract_ema=extract_ema
    )

    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        unet = UNet2DConditionModel(**unet_config)

    if is_accelerate_available():
        if model_type not in ["SDXL", "SDXL-Refiner"]:  # SBM Delay this.
            for param_name, param in converted_unet_checkpoint.items():
                set_module_tensor_to_device(unet, param_name, "cpu", value=param)
    else:
        unet.load_state_dict(converted_unet_checkpoint)

    # Convert the VAE model.
    if vae_path is None and vae is None:
        vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

        if (
            "model" in original_config
            and "params" in original_config["model"]
            and "scale_factor" in original_config["model"]["params"]
        ):
            vae_scaling_factor = original_config["model"]["params"]["scale_factor"]
        else:
            vae_scaling_factor = 0.18215  # default SD scaling factor

        vae_config["scaling_factor"] = vae_scaling_factor

        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            vae = AutoencoderKL(**vae_config)

        if is_accelerate_available():
            for param_name, param in converted_vae_checkpoint.items():
                set_module_tensor_to_device(vae, param_name, "cpu", value=param)
        else:
            vae.load_state_dict(converted_vae_checkpoint)
    elif vae is None:
        vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=local_files_only)

    if model_type == "FrozenOpenCLIPEmbedder":
        config_name = "stabilityai/stable-diffusion-2"
        config_kwargs = {"subfolder": "text_encoder"}

        if text_encoder is None:
            text_model = convert_open_clip_checkpoint(
                checkpoint, config_name, local_files_only=local_files_only, **config_kwargs
            )
        else:
            text_model = text_encoder

        try:
            tokenizer = CLIPTokenizer.from_pretrained(
                "stabilityai/stable-diffusion-2", subfolder="tokenizer", local_files_only=local_files_only
            )
        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the tokenizer in the following path: 'stabilityai/stable-diffusion-2'."
            )

        if stable_unclip is None:
            if controlnet:
                pipe = pipeline_class(
                    vae=vae,
                    text_encoder=text_model,
                    tokenizer=tokenizer,
                    unet=unet,
                    scheduler=scheduler,
                    controlnet=controlnet,
                    safety_checker=safety_checker,
                    feature_extractor=feature_extractor,
                )
                if hasattr(pipe, "requires_safety_checker"):
                    pipe.requires_safety_checker = False

            elif pipeline_class == StableDiffusionUpscalePipeline:
                scheduler = DDIMScheduler.from_pretrained(
                    "stabilityai/stable-diffusion-x4-upscaler", subfolder="scheduler"
                )
                low_res_scheduler = DDPMScheduler.from_pretrained(
                    "stabilityai/stable-diffusion-x4-upscaler", subfolder="low_res_scheduler"
                )

                pipe = pipeline_class(
                    vae=vae,
                    text_encoder=text_model,
                    tokenizer=tokenizer,
                    unet=unet,
                    scheduler=scheduler,
                    low_res_scheduler=low_res_scheduler,
                    safety_checker=safety_checker,
                    feature_extractor=feature_extractor,
                )

            else:
                pipe = pipeline_class(
                    vae=vae,
                    text_encoder=text_model,
                    tokenizer=tokenizer,
                    unet=unet,
                    scheduler=scheduler,
                    safety_checker=safety_checker,
                    feature_extractor=feature_extractor,
                )
                if hasattr(pipe, "requires_safety_checker"):
                    pipe.requires_safety_checker = False

        else:
            image_normalizer, image_noising_scheduler = stable_unclip_image_noising_components(
                original_config, clip_stats_path=clip_stats_path, device=device
            )

            if stable_unclip == "img2img":
                feature_extractor, image_encoder = stable_unclip_image_encoder(original_config)

                pipe = StableUnCLIPImg2ImgPipeline(
                    # image encoding components
                    feature_extractor=feature_extractor,
                    image_encoder=image_encoder,
                    # image noising components
                    image_normalizer=image_normalizer,
                    image_noising_scheduler=image_noising_scheduler,
                    # regular denoising components
                    tokenizer=tokenizer,
                    text_encoder=text_model,
                    unet=unet,
                    scheduler=scheduler,
                    # vae
                    vae=vae,
                )
            elif stable_unclip == "txt2img":
                if stable_unclip_prior is None or stable_unclip_prior == "karlo":
                    karlo_model = "kakaobrain/karlo-v1-alpha"
                    prior = PriorTransformer.from_pretrained(
                        karlo_model, subfolder="prior", local_files_only=local_files_only
                    )

                    try:
                        prior_tokenizer = CLIPTokenizer.from_pretrained(
                            "openai/clip-vit-large-patch14", local_files_only=local_files_only
                        )
                    except Exception:
                        raise ValueError(
                            f"With local_files_only set to {local_files_only}, you must first locally save the tokenizer in the following path: 'openai/clip-vit-large-patch14'."
                        )
                    prior_text_model = CLIPTextModelWithProjection.from_pretrained(
                        "openai/clip-vit-large-patch14", local_files_only=local_files_only
                    )

                    prior_scheduler = UnCLIPScheduler.from_pretrained(
                        karlo_model, subfolder="prior_scheduler", local_files_only=local_files_only
                    )
                    prior_scheduler = DDPMScheduler.from_config(prior_scheduler.config)
                else:
                    raise NotImplementedError(f"unknown prior for stable unclip model: {stable_unclip_prior}")

                pipe = StableUnCLIPPipeline(
                    # prior components
                    prior_tokenizer=prior_tokenizer,
                    prior_text_encoder=prior_text_model,
                    prior=prior,
                    prior_scheduler=prior_scheduler,
                    # image noising components
                    image_normalizer=image_normalizer,
                    image_noising_scheduler=image_noising_scheduler,
                    # regular denoising components
                    tokenizer=tokenizer,
                    text_encoder=text_model,
                    unet=unet,
                    scheduler=scheduler,
                    # vae
                    vae=vae,
                )
            else:
                raise NotImplementedError(f"unknown `stable_unclip` type: {stable_unclip}")
    elif model_type == "PaintByExample":
        vision_model = convert_paint_by_example_checkpoint(checkpoint)
        try:
            tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14", local_files_only=local_files_only
            )
        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the tokenizer in the following path: 'openai/clip-vit-large-patch14'."
            )
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", local_files_only=local_files_only
            )
        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the feature_extractor in the following path: 'CompVis/stable-diffusion-safety-checker'."
            )
        pipe = PaintByExamplePipeline(
            vae=vae,
            image_encoder=vision_model,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=feature_extractor,
        )
    elif model_type == "FrozenCLIPEmbedder":
        text_model = convert_ldm_clip_checkpoint(
            checkpoint, local_files_only=local_files_only, text_encoder=text_encoder
        )
        try:
            tokenizer = (
                CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", local_files_only=local_files_only)
                if tokenizer is None
                else tokenizer
            )
        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the tokenizer in the following path: 'openai/clip-vit-large-patch14'."
            )

        if load_safety_checker:
            safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", local_files_only=local_files_only
            )
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", local_files_only=local_files_only
            )

        if controlnet:
            pipe = pipeline_class(
                vae=vae,
                text_encoder=text_model,
                tokenizer=tokenizer,
                unet=unet,
                controlnet=controlnet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
            )
        else:
            pipe = pipeline_class(
                vae=vae,
                text_encoder=text_model,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
            )
    elif model_type in ["SDXL", "SDXL-Refiner"]:
        is_refiner = model_type == "SDXL-Refiner"

        if (is_refiner is False) and (tokenizer is None):
            try:
                tokenizer = CLIPTokenizer.from_pretrained(
                    "openai/clip-vit-large-patch14", local_files_only=local_files_only
                )
            except Exception:
                raise ValueError(
                    f"With local_files_only set to {local_files_only}, you must first locally save the tokenizer in the following path: 'openai/clip-vit-large-patch14'."
                )

        if (is_refiner is False) and (text_encoder is None):
            text_encoder = convert_ldm_clip_checkpoint(checkpoint, local_files_only=local_files_only)

        if tokenizer_2 is None:
            try:
                tokenizer_2 = CLIPTokenizer.from_pretrained(
                    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", pad_token="!", local_files_only=local_files_only
                )
            except Exception:
                raise ValueError(
                    f"With local_files_only set to {local_files_only}, you must first locally save the tokenizer in the following path: 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k' with `pad_token` set to '!'."
                )

        if text_encoder_2 is None:
            config_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
            config_kwargs = {"projection_dim": 1280}
            prefix = "conditioner.embedders.0.model." if is_refiner else "conditioner.embedders.1.model."

            text_encoder_2 = convert_open_clip_checkpoint(
                checkpoint,
                config_name,
                prefix=prefix,
                has_projection=True,
                local_files_only=local_files_only,
                **config_kwargs,
            )

        if is_accelerate_available():  # SBM Now move model to cpu.
            for param_name, param in converted_unet_checkpoint.items():
                set_module_tensor_to_device(unet, param_name, "cpu", value=param)

        if controlnet:
            pipe = pipeline_class(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                unet=unet,
                controlnet=controlnet,
                scheduler=scheduler,
                force_zeros_for_empty_prompt=True,
            )
        elif adapter:
            pipe = pipeline_class(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                unet=unet,
                adapter=adapter,
                scheduler=scheduler,
                force_zeros_for_empty_prompt=True,
            )

        else:
            pipeline_kwargs = {
                "vae": vae,
                "text_encoder": text_encoder,
                "tokenizer": tokenizer,
                "text_encoder_2": text_encoder_2,
                "tokenizer_2": tokenizer_2,
                "unet": unet,
                "scheduler": scheduler,
            }

            if (pipeline_class == StableDiffusionXLImg2ImgPipeline) or (
                pipeline_class == StableDiffusionXLInpaintPipeline
            ):
                pipeline_kwargs.update({"requires_aesthetics_score": is_refiner})

            if is_refiner:
                pipeline_kwargs.update({"force_zeros_for_empty_prompt": False})

            pipe = pipeline_class(**pipeline_kwargs)
    else:
        text_config = create_ldm_bert_config(original_config)
        text_model = convert_ldm_bert_checkpoint(checkpoint, text_config)
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", local_files_only=local_files_only)
        pipe = LDMTextToImagePipeline(vqvae=vae, bert=text_model, tokenizer=tokenizer, unet=unet, scheduler=scheduler)

    return pipe


def download_controlnet_from_original_ckpt(
    checkpoint_path: str,
    original_config_file: str,
    image_size: int = 512,
    extract_ema: bool = False,
    num_in_channels: int | None = None,
    upcast_attention: bool | None = None,
    device: str = None,
    from_safetensors: bool = False,
    use_linear_projection: bool | None = None,
    cross_attention_dim: bool | None = None,
) -> DiffusionPipeline:
    if from_safetensors:
        from safetensors import safe_open

        checkpoint = {}
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                checkpoint[key] = f.get_tensor(key)
    else:
        if device is None:
            device = get_device()
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

    # NOTE: this while loop isn't great but this controlnet checkpoint has one additional
    # "state_dict" key https://huggingface.co/thibaud/controlnet-canny-sd21
    while "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    with open(original_config_file, "r") as f:
        original_config_file = f.read()
    original_config = yaml.safe_load(original_config_file)

    if num_in_channels is not None:
        original_config["model"]["params"]["unet_config"]["params"]["in_channels"] = num_in_channels

    if "control_stage_config" not in original_config["model"]["params"]:
        raise ValueError("`control_stage_config` not present in original config")

    controlnet = convert_controlnet_checkpoint(
        checkpoint,
        original_config,
        checkpoint_path,
        image_size,
        upcast_attention,
        extract_ema,
        use_linear_projection=use_linear_projection,
        cross_attention_dim=cross_attention_dim,
    )

    return controlnet
