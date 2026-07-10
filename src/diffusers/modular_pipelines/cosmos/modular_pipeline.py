import torch

from ...pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipeline, CosmosSafetyChecker
from ..modular_pipeline import ModularPipeline


class Cosmos3OmniModularPipeline(ModularPipeline):
    """
    A ModularPipeline for Cosmos 3 omni generation.
    """

    default_blocks_name = "Cosmos3OmniBlocks"

    duration_template = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
    image_resolution_template = "This image is of {height}x{width} resolution."
    video_resolution_template = "This video is of {height}x{width} resolution."
    inverse_duration_template = "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS."
    inverse_image_resolution_template = "This image is not of {height}x{width} resolution."
    inverse_video_resolution_template = "This video is not of {height}x{width} resolution."

    @property
    def vae_scale_factor_spatial(self):
        if getattr(self, "vae", None) is not None:
            return int(self.vae.config.scale_factor_spatial)
        return 16

    @property
    def vae_scale_factor_temporal(self):
        if getattr(self, "vae", None) is not None:
            return int(self.vae.config.scale_factor_temporal)
        return 4

    @property
    def num_channels_latents(self):
        if getattr(self, "transformer", None) is not None:
            return int(self.transformer.config.latent_channel)
        return 48

    @property
    def sound_sampling_rate(self):
        if getattr(self, "sound_tokenizer", None) is not None:
            return int(self.sound_tokenizer.config.sampling_rate)
        return 48000

    @property
    def sound_hop_size(self):
        if getattr(self, "sound_tokenizer", None) is not None:
            return int(self.sound_tokenizer._hop_size)
        return 1920

    @property
    def _vae_latents_mean(self):
        return torch.tensor(self.vae.config.latents_mean, dtype=self.vae.dtype)

    @property
    def _vae_latents_inv_std(self):
        return 1.0 / torch.tensor(self.vae.config.latents_std, dtype=self.vae.dtype)

    @property
    def llm_special_tokens(self):
        if getattr(self, "text_tokenizer", None) is None:
            return None
        return {
            "start_of_generation": self.text_tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            "eos_token_id": self.text_tokenizer.eos_token_id,
        }

    def enable_safety_checker(self, safety_checker=None):
        if safety_checker is not None:
            self.safety_checker = safety_checker
        elif getattr(self, "safety_checker", None) is None:
            self.safety_checker = CosmosSafetyChecker()
        self._is_safety_checker_enabled = True

    def disable_safety_checker(self):
        self._is_safety_checker_enabled = False

    @property
    def requires_safety_checker(self):
        return getattr(self, "_is_safety_checker_enabled", True)

    def _encode_video(self, x):
        return Cosmos3OmniPipeline._encode_video(self, x)

    def decode_sound(self, latent):
        return Cosmos3OmniPipeline.decode_sound(self, latent)

    def _prepare_text_segment(self, input_ids, device):
        return Cosmos3OmniPipeline._prepare_text_segment(self, input_ids, device)

    def _prepare_vision_segment(self, *args, **kwargs):
        return Cosmos3OmniPipeline._prepare_vision_segment(self, *args, **kwargs)

    def _prepare_sound_segment(self, *args, **kwargs):
        return Cosmos3OmniPipeline._prepare_sound_segment(self, *args, **kwargs)

    def _prepare_action_segment(self, *args, **kwargs):
        return Cosmos3OmniPipeline._prepare_action_segment(self, *args, **kwargs)

    def _prepare_action_video_conditioning(self, *args, **kwargs):
        return Cosmos3OmniPipeline._prepare_action_video_conditioning(self, *args, **kwargs)

    def _remove_action_video_padding_from_latent(self, *args, **kwargs):
        return Cosmos3OmniPipeline._remove_action_video_padding_from_latent(self, *args, **kwargs)

    @staticmethod
    def _build_action_json_prompt(*args, **kwargs):
        return Cosmos3OmniPipeline._build_action_json_prompt(*args, **kwargs)

    def tokenize_prompt(self, *args, **kwargs):
        return Cosmos3OmniPipeline.tokenize_prompt(self, *args, **kwargs)

    @staticmethod
    def _mask_velocity_predictions(*args, **kwargs):
        return Cosmos3OmniPipeline._mask_velocity_predictions(*args, **kwargs)

    def _apply_video_safety_check(self, *args, **kwargs):
        return Cosmos3OmniPipeline._apply_video_safety_check(self, *args, **kwargs)
