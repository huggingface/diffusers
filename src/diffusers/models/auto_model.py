# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from collections import OrderedDict

from huggingface_hub.utils import validate_hf_hub_args

from ..configuration_utils import ConfigMixin
from ..pipelines.audioldm2.modeling_audioldm2 import AudioLDM2ProjectionModel, AudioLDM2UNet2DConditionModel
from ..pipelines.deepfloyd_if.watermark import IFWatermarker
from ..pipelines.flux.modeling_flux import ReduxImageEncoder
from ..pipelines.shap_e.renderer import MLPNeRSTFModel, ShapEParamsProjModel, ShapERenderer
from ..pipelines.stable_audio.modeling_stable_audio import StableAudioProjectionModel
from ..pipelines.stable_diffusion.clip_image_project_model import CLIPImageProjection
from ..pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer
from ..pipelines.unclip.text_proj import UnCLIPTextProjModel
from ..pipelines.unidiffuser.modeling_text_decoder import UniDiffuserTextDecoder
from ..pipelines.unidiffuser.modeling_uvit import UniDiffuserModel, UTransformer2DModel
from ..pipelines.wuerstchen.modeling_paella_vq_model import PaellaVQModel
from ..pipelines.wuerstchen.modeling_wuerstchen_diffnext import WuerstchenDiffNeXt
from ..pipelines.wuerstchen.modeling_wuerstchen_prior import WuerstchenPrior
from .adapter import MultiAdapter, T2IAdapter
from .autoencoders.autoencoder_asym_kl import AsymmetricAutoencoderKL
from .autoencoders.autoencoder_dc import AutoencoderDC
from .autoencoders.autoencoder_kl import AutoencoderKL
from .autoencoders.autoencoder_kl_allegro import AutoencoderKLAllegro
from .autoencoders.autoencoder_kl_cogvideox import AutoencoderKLCogVideoX
from .autoencoders.autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo
from .autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo
from .autoencoders.autoencoder_kl_magvit import AutoencoderKLMagvit
from .autoencoders.autoencoder_kl_mochi import AutoencoderKLMochi
from .autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from .autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from .autoencoders.autoencoder_oobleck import AutoencoderOobleck
from .autoencoders.autoencoder_tiny import AutoencoderTiny
from .autoencoders.consistency_decoder_vae import ConsistencyDecoderVAE
from .autoencoders.vq_model import VQModel
from .controlnets.controlnet import ControlNetModel
from .controlnets.controlnet_flux import FluxControlNetModel, FluxMultiControlNetModel
from .controlnets.controlnet_hunyuan import HunyuanDiT2DControlNetModel, HunyuanDiT2DMultiControlNetModel
from .controlnets.controlnet_sd3 import SD3ControlNetModel, SD3MultiControlNetModel
from .controlnets.controlnet_sparsectrl import SparseControlNetModel
from .controlnets.controlnet_union import ControlNetUnionModel
from .controlnets.controlnet_xs import ControlNetXSAdapter, UNetControlNetXSModel
from .transformers.auraflow_transformer_2d import AuraFlowTransformer2DModel
from .transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from .transformers.consisid_transformer_3d import ConsisIDTransformer3DModel
from .transformers.dit_transformer_2d import DiTTransformer2DModel
from .transformers.hunyuan_transformer_2d import HunyuanDiT2DModel
from .transformers.latte_transformer_3d import LatteTransformer3DModel
from .transformers.lumina_nextdit2d import LuminaNextDiT2DModel
from .transformers.pixart_transformer_2d import PixArtTransformer2DModel
from .transformers.prior_transformer import PriorTransformer
from .transformers.sana_transformer import SanaTransformer2DModel
from .transformers.stable_audio_transformer import StableAudioDiTModel
from .transformers.t5_film_transformer import T5FilmDecoder
from .transformers.transformer_allegro import AllegroTransformer3DModel
from .transformers.transformer_cogview3plus import CogView3PlusTransformer2DModel
from .transformers.transformer_cogview4 import CogView4Transformer2DModel
from .transformers.transformer_easyanimate import EasyAnimateTransformer3DModel
from .transformers.transformer_flux import FluxTransformer2DModel
from .transformers.transformer_hunyuan_video import HunyuanVideoTransformer3DModel
from .transformers.transformer_ltx import LTXVideoTransformer3DModel
from .transformers.transformer_lumina2 import Lumina2Transformer2DModel
from .transformers.transformer_mochi import MochiTransformer3DModel
from .transformers.transformer_omnigen import OmniGenTransformer2DModel
from .transformers.transformer_sd3 import SD3Transformer2DModel
from .transformers.transformer_temporal import TransformerTemporalModel
from .transformers.transformer_wan import WanTransformer3DModel
from .unets.unet_1d import UNet1DModel
from .unets.unet_2d import UNet2DModel
from .unets.unet_2d_condition import UNet2DConditionModel
from .unets.unet_3d_condition import UNet3DConditionModel
from .unets.unet_i2vgen_xl import I2VGenXLUNet
from .unets.unet_kandinsky3 import Kandinsky3UNet
from .unets.unet_motion_model import MotionAdapter, UNetMotionModel
from .unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from .unets.unet_stable_cascade import StableCascadeUNet
from .unets.uvit_2d import UVit2DModel


AUTO_MODEL_MAPPING = OrderedDict(
    [
        ("multi-adapter", MultiAdapter),
        ("t2i-adapter", T2IAdapter),
        ("asym-autoencoder-kl", AsymmetricAutoencoderKL),
        ("autoencoder-dc", AutoencoderDC),
        ("autoencoder-kl-allegro", AutoencoderKLAllegro),
        ("autoencoder-kl-cogvideox", AutoencoderKLCogVideoX),
        ("autoencoder-kl-hunyuan-video", AutoencoderKLHunyuanVideo),
        ("autoencoder-kl-ltx", AutoencoderKLLTXVideo),
        ("autoencoder-kl-magvit", AutoencoderKLMagvit),
        ("autoencoder-kl-mochi", AutoencoderKLMochi),
        ("autoencoder-kl-temporal-decoder", AutoencoderKLTemporalDecoder),
        ("autoencoder-kl-wan", AutoencoderKLWan),
        ("autoencoder-kl", AutoencoderKL),
        ("autoencoder-oobleck", AutoencoderOobleck),
        ("autoencoder-tiny", AutoencoderTiny),
        ("consistency-decoder-vae", ConsistencyDecoderVAE),
        ("vq-model", VQModel),
        ("controlnet-flux", FluxControlNetModel),
        ("controlnet-flux-multi", FluxMultiControlNetModel),
        ("controlnet-hunyuan", HunyuanDiT2DControlNetModel),
        ("controlnet-hunyuan-multi", HunyuanDiT2DMultiControlNetModel),
        ("controlnet-sd3", SD3ControlNetModel),
        ("controlnet-sd3-multi", SD3MultiControlNetModel),
        ("controlnet-sparse", SparseControlNetModel),
        ("controlnet-union", ControlNetUnionModel),
        ("controlnet-xs-adapter", ControlNetXSAdapter),
        ("controlnet-xs", UNetControlNetXSModel),
        ("controlnet", ControlNetModel),
        ("auraflow-transformer-2d", AuraFlowTransformer2DModel),
        ("cogvideox-transformer-3d", CogVideoXTransformer3DModel),
        ("consisid-transformer-3d", ConsisIDTransformer3DModel),
        ("dit-transformer-2d", DiTTransformer2DModel),
        ("hunyuan-transformer-2d", HunyuanDiT2DModel),
        ("latte-transformer-3d", LatteTransformer3DModel),
        ("lumina-nextdit2d", LuminaNextDiT2DModel),
        ("pixart-transformer-2d", PixArtTransformer2DModel),
        ("prior-transformer", PriorTransformer),
        ("sana-transformer", SanaTransformer2DModel),
        ("stable-audio-transformer", StableAudioDiTModel),
        ("t5-film-decoder", T5FilmDecoder),
        ("allegro-transformer-3d", AllegroTransformer3DModel),
        ("cogview3plus-transformer-2d", CogView3PlusTransformer2DModel),
        ("cogview4-transformer-2d", CogView4Transformer2DModel),
        ("easyanimate-transformer-3d", EasyAnimateTransformer3DModel),
        ("flux-transformer-2d", FluxTransformer2DModel),
        ("hunyuan-video-transformer-3d", HunyuanVideoTransformer3DModel),
        ("ltx-video-transformer-3d", LTXVideoTransformer3DModel),
        ("lumina2-transformer-2d", Lumina2Transformer2DModel),
        ("mochi-transformer-3d", MochiTransformer3DModel),
        ("omnigen-transformer-2d", OmniGenTransformer2DModel),
        ("sd3-transformer-2d", SD3Transformer2DModel),
        ("transformer-temporal", TransformerTemporalModel),
        ("wan-transformer-3d", WanTransformer3DModel),
        ("unet-1d", UNet1DModel),
        ("unet-2d", UNet2DModel),
        ("unet-2d-condition", UNet2DConditionModel),
        ("unet-3d-condition", UNet3DConditionModel),
        ("i2vgen-xl-unet", I2VGenXLUNet),
        ("kandinsky3-unet", Kandinsky3UNet),
        ("motion-adapter", MotionAdapter),
        ("unet-motion", UNetMotionModel),
        ("unet-spatio-temporal", UNetSpatioTemporalConditionModel),
        ("stable-cascade-unet", StableCascadeUNet),
        ("uvit-2d", UVit2DModel),
        ("audioldm2-projection", AudioLDM2ProjectionModel),
        ("audioldm2-unet-2d", AudioLDM2UNet2DConditionModel),
        ("if-watermarker", IFWatermarker),
        ("redux-image-encoder", ReduxImageEncoder),
        ("mlp-nerstf", MLPNeRSTFModel),
        ("shap-e-params-proj", ShapEParamsProjModel),
        ("shap-e-renderer", ShapERenderer),
        ("stable-audio-projection", StableAudioProjectionModel),
        ("clip-image-projection", CLIPImageProjection),
        ("stable-unclip-image-normalizer", StableUnCLIPImageNormalizer),
        ("unclip-text-proj", UnCLIPTextProjModel),
        ("unidiffuser-text-decoder", UniDiffuserTextDecoder),
        ("utransformer-2d", UTransformer2DModel),
        ("unidiffuser", UniDiffuserModel),
        ("paella-vq", PaellaVQModel),
        ("wuerstchen-diffnext", WuerstchenDiffNeXt),
        ("wuerstchen-prior", WuerstchenPrior),
    ]
)


SUPPORTED_TASKS_MAPPINGS = [AUTO_MODEL_MAPPING]


def _get_task_class(mapping, model_class_name, throw_error_if_not_exist: bool = True):
    def get_model(model_class_name):
        for task_mapping in SUPPORTED_TASKS_MAPPINGS:
            for model_name, model in task_mapping.items():
                if model.__name__ == model_class_name:
                    return model_name

    model_name = get_model(model_class_name)

    if model_name is not None:
        task_class = mapping.get(model_name, None)
        if task_class is not None:
            return task_class

    if throw_error_if_not_exist:
        raise ValueError(f"AutoModel can't find a model linked to {model_class_name} for {model_name}")


class AutoModel(ConfigMixin):
    config_name = "config.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_pipe(pipeline)` methods."
        )

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        r"""
        Instantiate a pretrained PyTorch model from a pretrained model configuration.

        The model is set in evaluation mode - `model.eval()` - by default, and dropout modules are deactivated. To
        train the model, set it back in training mode with `model.train()`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`~ModelMixin.save_pretrained`].

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info (`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device. Defaults to `None`, meaning that the model will be loaded on CPU.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if `device_map` contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            variant (`str`, *optional*):
                Load weights from a specified `variant` filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the `safetensors` weights are downloaded if they're available **and** if the
                `safetensors` library is installed. If set to `True`, the model is forcibly loaded from `safetensors`
                weights. If set to `False`, `safetensors` weights are not loaded.
            disable_mmap ('bool', *optional*, defaults to 'False'):
                Whether to disable mmap when loading a Safetensors model. This option can perform better when the model
                is on a network mount or hard drive, which may not handle the seeky-ness of mmap very well.

        <Tip>

        To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with
        `huggingface-cli login`. You can also activate the special
        ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use this method in a
        firewalled environment.

        </Tip>

        Example:

        ```py
        from diffusers import AutoModel

        unet = AutoModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
        ```

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```bash
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)

        load_config_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "token": token,
            "local_files_only": local_files_only,
            "revision": revision,
            "subfolder": subfolder,
        }

        config = cls.load_config(pretrained_model_or_path, **load_config_kwargs)
        orig_class_name = config["_class_name"]

        model_cls = _get_task_class(AUTO_MODEL_MAPPING, orig_class_name)

        kwargs = {**load_config_kwargs, **kwargs}
        return model_cls.from_pretrained(pretrained_model_or_path, **kwargs)
