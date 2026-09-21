# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Resolve a component's reversible conversion from its class and Diffusers configuration."""

import importlib
import inspect

from .ace_step import ace_step_conversion
from .ace_step_conditioner import ace_step_conditioner_conversion
from .ace_step_detokenizer import ace_step_detokenizer_conversion
from .ace_step_tokenizer import ace_step_tokenizer_conversion
from .anima_conditioner import anima_conditioner_conversion
from .animatediff import animatediff_conversion
from .anyflow import anyflow_conversion
from .anyflow_far import anyflow_far_conversion
from .asymmetric_vae import asymmetric_vae_conversion
from .audioldm2_projection import audioldm2_projection_conversion
from .audioldm2_unet import audioldm2_unet_conversion
from .auraflow import auraflow_conversion
from .autoencoder_dc import autoencoder_dc_conversion
from .blip_qformer import blip_qformer_conversion
from .chroma import chroma_conversion
from .chronoedit import chronoedit_conversion
from .clap_audio import clap_audio_conversion
from .clap_text import clap_text_conversion
from .clip import clip_conversion
from .clip_vision import clip_vision_conversion
from .cogvideox import cogvideox_transformer_conversion, cogvideox_vae_conversion
from .cogview3plus import cogview3plus_conversion
from .cogview4 import cogview4_conversion
from .consistency_decoder import consistency_decoder_conversion
from .controlnet import controlnet_conversion
from .cosmos import cosmos_conversion
from .cosmos_controlnet import cosmos_controlnet_conversion
from .cosmos_vae import cosmos_vae_conversion
from .dit import dit_conversion
from .ernie_image import ernie_image_conversion
from .flux import flux_conversion
from .flux2 import flux2_conversion
from .flux2_vae import flux2_vae_conversion
from .flux_ip_adapter import flux_ip_adapter_conversion
from .hidream import hidream_conversion
from .hifigan import hifigan_conversion
from .hunyuan_dit import hunyuan_dit_conversion
from .hunyuan_dit_controlnet import hunyuan_dit_controlnet_conversion
from .hunyuan_image import hunyuan_image_conversion
from .hunyuan_image_refiner_vae import hunyuan_image_refiner_vae_conversion
from .hunyuan_image_vae import hunyuan_image_vae_conversion
from .hunyuan_video import hunyuan_video_conversion
from .hunyuan_video15 import hunyuan_video15_conversion
from .hunyuan_video15_vae import hunyuan_video15_vae_conversion
from .hunyuan_video_vae import hunyuan_video_vae_conversion
from .i2vgen_xl import i2vgen_xl_conversion
from .joy_image import joy_image_conversion
from .kandinsky3 import kandinsky3_conversion
from .ldm_unet import ldm_unet_conversion
from .ldm_vae import ldm_vae_conversion
from .longcat_audio import longcat_audio_conversion
from .longcat_audio_vae import longcat_audio_vae_conversion
from .lora import lora_conversion
from .ltx import ltx_conversion
from .ltx2 import ltx2_conversion
from .ltx2_audio_vae import ltx2_audio_vae_conversion
from .ltx2_connectors import ltx2_connectors_conversion
from .ltx2_diffusion_decoder import ltx2_diffusion_decoder_conversion
from .ltx2_duration import ltx2_duration_conversion
from .ltx2_upsampler import ltx2_upsampler_conversion
from .ltx2_vae import ltx2_vae_conversion
from .ltx2_vocoder import ltx2_vocoder_conversion
from .ltx_upsampler import ltx_upsampler_conversion
from .ltx_vae import ltx_vae_conversion
from .lumina import lumina_conversion
from .lumina2 import lumina2_conversion
from .minimax_h3 import minimax_h3_conversion
from .minimax_h3_audio_vae import minimax_h3_audio_vae_conversion
from .minimax_h3_vae import minimax_h3_vae_conversion
from .minimax_music3 import minimax_music3_conversion
from .minimax_music3_conditioner import minimax_music3_conditioner_conversion
from .minimax_music3_rvq import minimax_music3_rvq_conversion
from .minimax_music3_vocoder import minimax_music3_vocoder_conversion
from .mochi import mochi_conversion
from .mochi_vae import mochi_vae_conversion
from .motif_video import motif_video_conversion
from .omnigen import omnigen_conversion
from .oobleck import oobleck_conversion
from .ovis_image import ovis_image_conversion
from .paella import paella_conversion
from .pipeline_components import (
    if_safety_checker_conversion,
    ldm_bert_conversion,
    learned_classifier_free_conversion,
    paint_by_example_conversion,
)
from .pixart import pixart_conversion
from .prior import prior_conversion
from .prx import prx_conversion
from .qwen3 import qwen3_conversion
from .qwen_image import qwen_image_conversion
from .qwen_image_vae import qwen_image_vae_conversion
from .rae import rae_conversion
from .same import same_conversion
from .sana import sana_conversion
from .sana_controlnet import sana_controlnet_conversion
from .sana_video import sana_video_conversion
from .sd3 import sd3_conversion
from .sd3_controlnet import sd3_controlnet_conversion
from .shap_e_renderer import shap_e_renderer_conversion
from .skyreels_v2 import skyreels_v2_conversion
from .sparse_controlnet import sparse_controlnet_conversion
from .spectrogram_continuous import spectrogram_continuous_conversion
from .spectrogram_notes import spectrogram_notes_conversion
from .stable_audio import stable_audio_conversion
from .stable_audio3 import stable_audio3_conversion
from .stable_audio3_duration import stable_audio3_duration_conversion
from .stable_audio_projection import stable_audio_projection_conversion
from .stable_cascade import stable_cascade_conversion
from .svd import svd_conversion
from .svd_vae import svd_vae_conversion
from .t2i_adapter import t2i_adapter_conversion
from .t5 import t5_conversion
from .t5_film import t5_film_conversion
from .tiny_vae import tiny_vae_conversion
from .umt5 import umt5_conversion
from .unclip_text_projection import unclip_text_projection_conversion
from .unet_1d import unet_1d_conversion
from .unet_2d import unet_2d_conversion
from .unet_3d import unet_3d_conversion
from .unidiffuser import unidiffuser_conversion
from .unidiffuser_text import unidiffuser_text_conversion
from .uvit import uvit_conversion
from .versatile_text_unet import versatile_text_unet_conversion
from .vq_diffusion import vq_diffusion_conversion
from .vq_model import vq_model_conversion
from .wan import wan_conversion
from .wan_animate import wan_animate_conversion
from .wan_animate2 import wan_animate2_conversion
from .wan_vace import wan_vace_conversion
from .wan_vae import wan_vae_conversion
from .wuerstchen_decoder import wuerstchen_decoder_conversion
from .wuerstchen_prior import wuerstchen_prior_conversion
from .z_image import z_image_conversion
from .z_image_controlnet import z_image_controlnet_conversion
from .zero123_projection import zero123_projection_conversion


CONVERSION_BUILDERS = {
    "LDMBertModel": ldm_bert_conversion,
    "PaintByExampleImageEncoder": paint_by_example_conversion,
    "IFSafetyChecker": if_safety_checker_conversion,
    "LearnedClassifierFreeSamplingEmbeddings": learned_classifier_free_conversion,
    "CogVideoXTransformer3DModel": cogvideox_transformer_conversion,
    "AutoencoderKLCogVideoX": cogvideox_vae_conversion,
    "ChronoEditTransformer3DModel": chronoedit_conversion,
    "QwenImageTransformer2DModel": qwen_image_conversion,
    "MotifVideoTransformer3DModel": motif_video_conversion,
    "LoRA": lora_conversion,
    "LTXLatentUpsamplerModel": ltx_upsampler_conversion,
    "SkyReelsV2Transformer3DModel": skyreels_v2_conversion,
    "ClapModel": clap_audio_conversion,
    "ClapAudioModel": clap_audio_conversion,
    "ClapAudioModelWithProjection": clap_audio_conversion,
    "ConsistencyDecoderVAE": consistency_decoder_conversion,
    "FluxIPAdapter": flux_ip_adapter_conversion,
    "ShapERenderer": shap_e_renderer_conversion,
    "JoyImageEditTransformer3DModel": joy_image_conversion,
    "JoyImageEditPlusTransformer3DModel": joy_image_conversion,
    "CLIPVisionModel": clip_vision_conversion,
    "CLIPVisionModelWithProjection": clip_vision_conversion,
    "Transformer2DModel": vq_diffusion_conversion,
    "UniDiffuserModel": unidiffuser_conversion,
    "UniDiffuserTextDecoder": unidiffuser_text_conversion,
    "CCProjection": zero123_projection_conversion,
    "LTX2DurationHead": ltx2_duration_conversion,
    "LTX2LatentUpsamplerModel": ltx2_upsampler_conversion,
    "LTX2TextConnectors": ltx2_connectors_conversion,
    "LTX2Vocoder": ltx2_vocoder_conversion,
    "LTX2VocoderWithBWE": ltx2_vocoder_conversion,
    "LTX2VideoDiffusionDecoderModel": ltx2_diffusion_decoder_conversion,
    "WuerstchenPrior": wuerstchen_prior_conversion,
    "WuerstchenDiffNeXt": wuerstchen_decoder_conversion,
    "PaellaVQModel": paella_conversion,
    "SparseControlNetModel": sparse_controlnet_conversion,
    "T2IAdapter": t2i_adapter_conversion,
    "AsymmetricAutoencoderKL": asymmetric_vae_conversion,
    "Blip2QFormerModel": blip_qformer_conversion,
    "ContextCLIPTextModel": clip_conversion,
    "AutoencoderKLFlux2": flux2_vae_conversion,
    "UNetFlatConditionModel": versatile_text_unet_conversion,
    "UNetSpatioTemporalConditionModel": svd_conversion,
    "AutoencoderKLTemporalDecoder": svd_vae_conversion,
    "I2VGenXLUNet": i2vgen_xl_conversion,
    "UNet3DConditionModel": unet_3d_conversion,
    "UNet1DModel": unet_1d_conversion,
    "Kandinsky3UNet": kandinsky3_conversion,
    "ClapTextModelWithProjection": clap_text_conversion,
    "SpeechT5HifiGan": hifigan_conversion,
    "AudioLDM2UNet2DConditionModel": audioldm2_unet_conversion,
    "AudioLDM2ProjectionModel": audioldm2_projection_conversion,
    "UnCLIPTextProjModel": unclip_text_projection_conversion,
    "PriorTransformer": prior_conversion,
    "UNet2DModel": unet_2d_conversion,
    "SD3ControlNetModel": sd3_controlnet_conversion,
    "SanaVideoTransformer3DModel": sana_video_conversion,
    "SanaControlNetModel": sana_controlnet_conversion,
    "CosmosControlNetModel": cosmos_controlnet_conversion,
    "AutoencoderRAE": rae_conversion,
    "AutoencoderKLMiniMaxH3Audio": minimax_h3_audio_vae_conversion,
    "AutoencoderKLHunyuanImage": hunyuan_image_vae_conversion,
    "AutoencoderKLHunyuanImageRefiner": hunyuan_image_refiner_vae_conversion,
    "AutoencoderKLMiniMaxH3": minimax_h3_vae_conversion,
    "HunyuanImageTransformer2DModel": hunyuan_image_conversion,
    "MiniMaxH3Transformer3DModel": minimax_h3_conversion,
    "PRXTransformer2DModel": prx_conversion,
    "HunyuanVideo15Transformer3DModel": hunyuan_video15_conversion,
    "AutoencoderKLHunyuanVideo15": hunyuan_video15_vae_conversion,
    "AutoencoderKLCosmos": cosmos_vae_conversion,
    "AutoencoderKLHunyuanVideo": hunyuan_video_vae_conversion,
    "AutoencoderKLMochi": mochi_vae_conversion,
    "AutoencoderTiny": tiny_vae_conversion,
    "SpectrogramNotesEncoder": spectrogram_notes_conversion,
    "SpectrogramContEncoder": spectrogram_continuous_conversion,
    "T5FilmDecoder": t5_film_conversion,
    "Qwen3Model": qwen3_conversion,
    "Qwen3ForCausalLM": qwen3_conversion,
    "UMT5EncoderModel": umt5_conversion,
    "UVit2DModel": uvit_conversion,
    "VQModel": vq_model_conversion,
    "AceStepTransformer1DModel": ace_step_conversion,
    "AceStepConditionEncoder": ace_step_conditioner_conversion,
    "AceStepAudioTokenizer": ace_step_tokenizer_conversion,
    "AceStepAudioTokenDetokenizer": ace_step_detokenizer_conversion,
    "LongCatAudioDiTTransformer": longcat_audio_conversion,
    "LongCatAudioDiTVae": longcat_audio_vae_conversion,
    "AutoencoderSAME": same_conversion,
    "StableAudio3DiTModel": stable_audio3_conversion,
    "StableAudio3DurationEmbedder": stable_audio3_duration_conversion,
    "AutoencoderOobleck": oobleck_conversion,
    "StableAudioProjectionModel": stable_audio_projection_conversion,
    "MiniMaxMusic3ConditionEncoder": minimax_music3_conditioner_conversion,
    "MiniMaxMusic3Vocoder": minimax_music3_vocoder_conversion,
    "MiniMaxMusic3RVQDepthDecoder": minimax_music3_rvq_conversion,
    "StableAudioDiTModel": stable_audio_conversion,
    "MiniMaxMusic3Transformer1DModel": minimax_music3_conversion,
    "OvisImageTransformer2DModel": ovis_image_conversion,
    "AnyFlowTransformer3DModel": anyflow_conversion,
    "AnyFlowFARTransformer3DModel": anyflow_far_conversion,
    "AutoencoderKLQwenImage": qwen_image_vae_conversion,
    "AnimaTextConditioner": anima_conditioner_conversion,
    "OmniGenTransformer2DModel": omnigen_conversion,
    "LuminaNextDiT2DModel": lumina_conversion,
    "HunyuanDiT2DModel": hunyuan_dit_conversion,
    "HunyuanDiT2DControlNetModel": hunyuan_dit_controlnet_conversion,
    "DiTTransformer2DModel": dit_conversion,
    "PixArtTransformer2DModel": pixart_conversion,
    "CogView3PlusTransformer2DModel": cogview3plus_conversion,
    "CogView4Transformer2DModel": cogview4_conversion,
    "WanAnimateTransformer3DModel": wan_animate_conversion,
    "WanVACETransformer3DModel": wan_vace_conversion,
    "WanAnimate2Transformer3DModel": wan_animate2_conversion,
    "ZImageControlNetModel": z_image_controlnet_conversion,
    "AutoencoderKLLTX2Audio": ltx2_audio_vae_conversion,
    "CLIPTextModel": clip_conversion,
    "CLIPTextModelWithProjection": clip_conversion,
    "T5EncoderModel": t5_conversion,
    "AutoencoderDC": autoencoder_dc_conversion,
    "StableCascadeUNet": stable_cascade_conversion,
    "ErnieImageTransformer2DModel": ernie_image_conversion,
    "HiDreamImageTransformer2DModel": hidream_conversion,
    "MotionAdapter": animatediff_conversion,
    "UNet2DConditionModel": ldm_unet_conversion,
    "ControlNetModel": controlnet_conversion,
    "AutoencoderKL": ldm_vae_conversion,
    "AutoencoderKLWan": wan_vae_conversion,
    "AutoencoderKLLTXVideo": ltx_vae_conversion,
    "AutoencoderKLLTX2Video": ltx2_vae_conversion,
    "LTX2VideoTransformer3DModel": ltx2_conversion,
    "CosmosTransformer3DModel": cosmos_conversion,
    "LTXVideoTransformer3DModel": ltx_conversion,
    "WanTransformer3DModel": wan_conversion,
    "ZImageTransformer2DModel": z_image_conversion,
    "AuraFlowTransformer2DModel": auraflow_conversion,
    "Lumina2Transformer2DModel": lumina2_conversion,
    "HunyuanVideoTransformer3DModel": hunyuan_video_conversion,
    "SanaTransformer2DModel": sana_conversion,
    "FluxTransformer2DModel": flux_conversion,
    "Flux2Transformer2DModel": flux2_conversion,
    "ChromaTransformer2DModel": chroma_conversion,
    "SD3Transformer2DModel": sd3_conversion,
    "MochiTransformer3DModel": mochi_conversion,
}


CONVERSION_FORMATS = {
    "ConsistencyDecoderVAE": ("consistency_decoder", "consistency_decoder_jit"),
    "CLIPTextModel": ("clip", "openclip"),
    "CLIPTextModelWithProjection": ("clip", "openclip"),
    "ContextCLIPTextModel": ("clip", "openclip"),
    "CLIPVisionModel": ("clip", "openclip"),
    "CLIPVisionModelWithProjection": ("clip", "openclip"),
    "CosmosTransformer3DModel": ("cosmos1", "cosmos2"),
    "CogView4Transformer2DModel": ("cogview4", "megatron"),
    "UNet1DModel": ("diffuser_rl", "diffuser_rl_legacy"),
    "UNet2DModel": ("ddpm", "ldm", "consistency", "ncsnpp"),
    "UNet2DConditionModel": ("ldm", "versatile_image"),
    "HunyuanImageTransformer2DModel": ("hunyuan_image_fused", "hunyuan_image_split"),
    "LoRA": ("kohya", "diffusers", "diffusers_old", "peft", "animatediff"),
    "LTX2VideoDiffusionDecoderModel": ("ltx2_diffusion_decoder", "ltx2_diffusion_decoder_gated"),
    "MiniMaxH3Transformer3DModel": ("minimax_h3", "minimax_h3_shards"),
    "PriorTransformer": ("shap_e", "unclip", "kandinsky"),
    "PRXTransformer2DModel": ("prx", "prx_weight_norm"),
    "AutoencoderKLTemporalDecoder": ("svd", "temporal_vae"),
    "AutoencoderKLHunyuanImage": ("hunyuan_image_vae", "hunyuan_image_vae_2d"),
}


def get_conversion(model_class: str, config: dict):
    """Build a component conversion using its Diffusers config, without constructing a model.

    Each definition returns `Conversion(mapping=..., rules=...)`. Constructor defaults fill omitted config fields; no
    original checkpoint or previous import is required. File wrappers, prefixes, and auxiliary state are separate.
    """
    original_format = config.get("original_format")
    if original_format is not None and original_format not in CONVERSION_FORMATS.get(model_class, ()):
        raise ValueError(f"Unsupported original_format {original_format!r} for {model_class}.")
    if model_class == "LoRA":
        return lora_conversion(config)
    if model_class == "FluxIPAdapter":
        return flux_ip_adapter_conversion(config)
    if model_class == "CCProjection":
        return zero123_projection_conversion(config)
    if model_class == "Transformer2DModel":
        if config.get("norm_type") == "ada_norm_zero":
            model_class = "DiTTransformer2DModel"
        elif config.get("norm_type") == "ada_norm_single":
            model_class = "PixArtTransformer2DModel"
    if model_class == "CogVideoXTransformer3DModel":
        return cogvideox_transformer_conversion(config)
    if model_class == "AutoencoderKLCogVideoX":
        return cogvideox_vae_conversion(config)
    if model_class not in CONVERSION_BUILDERS:
        raise ValueError(f"No reversible conversion registered for {model_class}.")
    if model_class in ("LDMBertModel", "PaintByExampleImageEncoder", "IFSafetyChecker"):
        if model_class == "LDMBertModel":
            module = importlib.import_module("diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion")
            resolved_config = module.LDMBertConfig(**config).to_dict()
        else:
            module = importlib.import_module("transformers")
            config_class = (
                module.CLIPVisionConfig if model_class == "PaintByExampleImageEncoder" else module.CLIPConfig
            )
            resolved_config = config_class(**config).to_dict()
            resolved_config["transformers_version"] = module.__version__
        resolved_config.update(config)
        return CONVERSION_BUILDERS[model_class](resolved_config)
    if model_class in (
        "CLIPVisionModel",
        "CLIPVisionModelWithProjection",
        "CLIPTextModel",
        "CLIPTextModelWithProjection",
        "ContextCLIPTextModel",
        "Blip2QFormerModel",
        "T5EncoderModel",
        "UMT5EncoderModel",
        "Qwen3Model",
        "Qwen3ForCausalLM",
        "ClapModel",
        "ClapAudioModel",
        "ClapAudioModelWithProjection",
        "ClapTextModelWithProjection",
        "SpeechT5HifiGan",
    ):
        transformers = importlib.import_module("transformers")
        config_name = {
            "CLIPVisionModel": "CLIPVisionConfig",
            "CLIPVisionModelWithProjection": "CLIPVisionConfig",
            "T5EncoderModel": "T5Config",
            "Blip2QFormerModel": "Blip2Config",
            "UMT5EncoderModel": "UMT5Config",
            "Qwen3Model": "Qwen3Config",
            "Qwen3ForCausalLM": "Qwen3Config",
            "ClapModel": "ClapConfig",
            "ClapAudioModel": "ClapAudioConfig",
            "ClapAudioModelWithProjection": "ClapAudioConfig",
            "ClapTextModelWithProjection": "ClapTextConfig",
            "SpeechT5HifiGan": "SpeechT5HifiGanConfig",
        }.get(model_class, "CLIPTextConfig")
        config_class = getattr(transformers, config_name)
        resolved_config = config_class(**config).to_dict()
        resolved_config.update(config)
        resolved_config["_class_name"] = model_class
        return CONVERSION_BUILDERS[model_class](resolved_config)
    internal_modules = {
        "LearnedClassifierFreeSamplingEmbeddings": "diffusers.pipelines.deprecated.vq_diffusion.pipeline_vq_diffusion",
        "LTXLatentUpsamplerModel": "diffusers.pipelines.ltx.modeling_latent_upsampler",
        "ShapERenderer": "diffusers.pipelines.shap_e.renderer",
        "UniDiffuserModel": "diffusers.pipelines.deprecated.unidiffuser.modeling_uvit",
        "UniDiffuserTextDecoder": "diffusers.pipelines.deprecated.unidiffuser.modeling_text_decoder",
        "LTX2TextConnectors": "diffusers.pipelines.ltx2.connectors",
        "LTX2Vocoder": "diffusers.pipelines.ltx2.vocoder",
        "LTX2VocoderWithBWE": "diffusers.pipelines.ltx2.vocoder",
        "LTX2DurationHead": "diffusers.pipelines.ltx2.duration_head",
        "LTX2LatentUpsamplerModel": "diffusers.pipelines.ltx2.latent_upsampler",
        "WuerstchenPrior": "diffusers.pipelines.deprecated.wuerstchen.modeling_wuerstchen_prior",
        "WuerstchenDiffNeXt": "diffusers.pipelines.deprecated.wuerstchen.modeling_wuerstchen_diffnext",
        "PaellaVQModel": "diffusers.pipelines.deprecated.wuerstchen.modeling_paella_vq_model",
        "UNetFlatConditionModel": "diffusers.pipelines.deprecated.versatile_diffusion.modeling_text_unet",
        "UnCLIPTextProjModel": "diffusers.pipelines.deprecated.unclip.text_proj",
        "SpectrogramNotesEncoder": "diffusers.pipelines.deprecated.spectrogram_diffusion.notes_encoder",
        "SpectrogramContEncoder": "diffusers.pipelines.deprecated.spectrogram_diffusion.continuous_encoder",
    }
    cls = getattr(importlib.import_module(internal_modules.get(model_class, "diffusers")), model_class)
    resolved_config = {
        name: parameter.default
        for name, parameter in inspect.signature(cls.__init__).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    resolved_config.update(config)
    if model_class == "AutoencoderRAE" and not resolved_config.get("transformers_version"):
        resolved_config["transformers_version"] = importlib.import_module("transformers").__version__
    resolved_config["_class_name"] = model_class
    return CONVERSION_BUILDERS[model_class](resolved_config)
