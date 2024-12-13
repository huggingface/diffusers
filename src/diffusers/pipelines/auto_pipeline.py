# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
from ..loaders.single_file_utils import (
    get_keyword_types,
    infer_diffusers_model_type,
    load_single_file_checkpoint,
)
from ..utils import is_sentencepiece_available
from .animatediff import AnimateDiffPipeline, AnimateDiffSDXLPipeline
from .aura_flow import AuraFlowPipeline
from .cogview3 import CogView3PlusPipeline
from .controlnet import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
)
from .deepfloyd_if import IFImg2ImgPipeline, IFInpaintingPipeline, IFPipeline
from .flux import (
    FluxControlNetImg2ImgPipeline,
    FluxControlNetInpaintPipeline,
    FluxControlNetPipeline,
    FluxImg2ImgPipeline,
    FluxInpaintPipeline,
    FluxPipeline,
)
from .hunyuandit import HunyuanDiTPipeline
from .kandinsky import (
    KandinskyCombinedPipeline,
    KandinskyImg2ImgCombinedPipeline,
    KandinskyImg2ImgPipeline,
    KandinskyInpaintCombinedPipeline,
    KandinskyInpaintPipeline,
    KandinskyPipeline,
)
from .kandinsky2_2 import (
    KandinskyV22CombinedPipeline,
    KandinskyV22Img2ImgCombinedPipeline,
    KandinskyV22Img2ImgPipeline,
    KandinskyV22InpaintCombinedPipeline,
    KandinskyV22InpaintPipeline,
    KandinskyV22Pipeline,
)
from .kandinsky3 import Kandinsky3Img2ImgPipeline, Kandinsky3Pipeline
from .latent_consistency_models import (
    LatentConsistencyModelImg2ImgPipeline,
    LatentConsistencyModelPipeline,
)
from .lumina import LuminaText2ImgPipeline
from .pag import (
    HunyuanDiTPAGPipeline,
    PixArtSigmaPAGPipeline,
    StableDiffusion3PAGImg2ImgPipeline,
    StableDiffusion3PAGPipeline,
    StableDiffusionControlNetPAGInpaintPipeline,
    StableDiffusionControlNetPAGPipeline,
    StableDiffusionPAGImg2ImgPipeline,
    StableDiffusionPAGInpaintPipeline,
    StableDiffusionPAGPipeline,
    StableDiffusionXLControlNetPAGImg2ImgPipeline,
    StableDiffusionXLControlNetPAGPipeline,
    StableDiffusionXLPAGImg2ImgPipeline,
    StableDiffusionXLPAGInpaintPipeline,
    StableDiffusionXLPAGPipeline,
)
from .pixart_alpha import PixArtAlphaPipeline, PixArtSigmaPipeline
from .stable_cascade import StableCascadeCombinedPipeline, StableCascadeDecoderPipeline
from .stable_diffusion import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
)
from .stable_diffusion_3 import (
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3InpaintPipeline,
    StableDiffusion3Pipeline,
)
from .stable_diffusion_xl import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from .wuerstchen import WuerstchenCombinedPipeline, WuerstchenDecoderPipeline


AUTO_TEXT2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", StableDiffusionPipeline),
        ("stable-diffusion-xl", StableDiffusionXLPipeline),
        ("stable-diffusion-3", StableDiffusion3Pipeline),
        ("stable-diffusion-3-pag", StableDiffusion3PAGPipeline),
        ("if", IFPipeline),
        ("hunyuan", HunyuanDiTPipeline),
        ("hunyuan-pag", HunyuanDiTPAGPipeline),
        ("kandinsky", KandinskyCombinedPipeline),
        ("kandinsky22", KandinskyV22CombinedPipeline),
        ("kandinsky3", Kandinsky3Pipeline),
        ("stable-diffusion-controlnet", StableDiffusionControlNetPipeline),
        ("stable-diffusion-xl-controlnet", StableDiffusionXLControlNetPipeline),
        ("wuerstchen", WuerstchenCombinedPipeline),
        ("cascade", StableCascadeCombinedPipeline),
        ("lcm", LatentConsistencyModelPipeline),
        ("pixart-alpha", PixArtAlphaPipeline),
        ("pixart-sigma", PixArtSigmaPipeline),
        ("stable-diffusion-pag", StableDiffusionPAGPipeline),
        ("stable-diffusion-controlnet-pag", StableDiffusionControlNetPAGPipeline),
        ("stable-diffusion-xl-pag", StableDiffusionXLPAGPipeline),
        ("stable-diffusion-xl-controlnet-pag", StableDiffusionXLControlNetPAGPipeline),
        ("pixart-sigma-pag", PixArtSigmaPAGPipeline),
        ("auraflow", AuraFlowPipeline),
        ("flux", FluxPipeline),
        ("flux-controlnet", FluxControlNetPipeline),
        ("lumina", LuminaText2ImgPipeline),
        ("cogview3", CogView3PlusPipeline),
    ]
)

AUTO_IMAGE2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", StableDiffusionImg2ImgPipeline),
        ("stable-diffusion-xl", StableDiffusionXLImg2ImgPipeline),
        ("stable-diffusion-3", StableDiffusion3Img2ImgPipeline),
        ("stable-diffusion-3-pag", StableDiffusion3PAGImg2ImgPipeline),
        ("if", IFImg2ImgPipeline),
        ("kandinsky", KandinskyImg2ImgCombinedPipeline),
        ("kandinsky22", KandinskyV22Img2ImgCombinedPipeline),
        ("kandinsky3", Kandinsky3Img2ImgPipeline),
        ("stable-diffusion-controlnet", StableDiffusionControlNetImg2ImgPipeline),
        ("stable-diffusion-pag", StableDiffusionPAGImg2ImgPipeline),
        ("stable-diffusion-xl-controlnet", StableDiffusionXLControlNetImg2ImgPipeline),
        ("stable-diffusion-xl-pag", StableDiffusionXLPAGImg2ImgPipeline),
        ("stable-diffusion-xl-controlnet-pag", StableDiffusionXLControlNetPAGImg2ImgPipeline),
        ("lcm", LatentConsistencyModelImg2ImgPipeline),
        ("flux", FluxImg2ImgPipeline),
        ("flux-controlnet", FluxControlNetImg2ImgPipeline),
    ]
)

AUTO_INPAINT_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", StableDiffusionInpaintPipeline),
        ("stable-diffusion-xl", StableDiffusionXLInpaintPipeline),
        ("stable-diffusion-3", StableDiffusion3InpaintPipeline),
        ("if", IFInpaintingPipeline),
        ("kandinsky", KandinskyInpaintCombinedPipeline),
        ("kandinsky22", KandinskyV22InpaintCombinedPipeline),
        ("stable-diffusion-controlnet", StableDiffusionControlNetInpaintPipeline),
        ("stable-diffusion-controlnet-pag", StableDiffusionControlNetPAGInpaintPipeline),
        ("stable-diffusion-xl-controlnet", StableDiffusionXLControlNetInpaintPipeline),
        ("stable-diffusion-xl-pag", StableDiffusionXLPAGInpaintPipeline),
        ("flux", FluxInpaintPipeline),
        ("flux-controlnet", FluxControlNetInpaintPipeline),
        ("stable-diffusion-pag", StableDiffusionPAGInpaintPipeline),
    ]
)

_AUTO_TEXT2IMAGE_DECODER_PIPELINES_MAPPING = OrderedDict(
    [
        ("kandinsky", KandinskyPipeline),
        ("kandinsky22", KandinskyV22Pipeline),
        ("wuerstchen", WuerstchenDecoderPipeline),
        ("cascade", StableCascadeDecoderPipeline),
    ]
)
_AUTO_IMAGE2IMAGE_DECODER_PIPELINES_MAPPING = OrderedDict(
    [
        ("kandinsky", KandinskyImg2ImgPipeline),
        ("kandinsky22", KandinskyV22Img2ImgPipeline),
    ]
)
_AUTO_INPAINT_DECODER_PIPELINES_MAPPING = OrderedDict(
    [
        ("kandinsky", KandinskyInpaintPipeline),
        ("kandinsky22", KandinskyV22InpaintPipeline),
    ]
)

SINGLE_FILE_CHECKPOINT_TEXT2IMAGE_PIPELINE_MAPPING = OrderedDict(
    [
        # Animated Diffusion Pipelines
        ("animatediff_rgb", AnimateDiffPipeline),
        ("animatediff_scribble", AnimateDiffPipeline),
        ("animatediff_sdxl_beta", AnimateDiffSDXLPipeline),
        ("animatediff_v1", AnimateDiffPipeline),
        ("animatediff_v2", AnimateDiffPipeline),
        ("animatediff_v3", AnimateDiffPipeline),

        # Cascade Stages
        ("stable_cascade_stage_b", None),
        ("stable_cascade_stage_b_lite", None),
        ("stable_cascade_stage_c", None),
        ("stable_cascade_stage_c_lite", None),

        # ControlNet Pipelines
        ("controlnet", StableDiffusionControlNetPipeline),
        ("controlnet_xl", StableDiffusionXLControlNetPipeline),
        ("controlnet_xl_large", StableDiffusionXLControlNetPipeline),
        ("controlnet_xl_mid", StableDiffusionXLControlNetPipeline),
        ("controlnet_xl_small", StableDiffusionXLControlNetPipeline),

        # Flux Pipelines
        ("flux_dev", FluxPipeline),
        ("flux_schnell", FluxPipeline),

        # Stable Diffusion Pipelines
        ("sd3", StableDiffusion3Pipeline),
        ("sd35_large", StableDiffusion3Pipeline),
        ("v1", StableDiffusionPipeline),
        ("v2", StableDiffusionPipeline),

        # Stable Diffusion XL Pipelines
        ("playground-v2-5", StableDiffusionXLPipeline),
        ("xl_base", StableDiffusionXLPipeline),
        ("xl_refiner", StableDiffusionXLPipeline),

        # Inpainting Pipelines
        ("inpainting", None),
        ("inpainting_v2", None),
        ("xl_inpaint", None),

        # Others
        ("upscale", StableDiffusionUpscalePipeline),
    ]
)

SINGLE_FILE_CHECKPOINT_IMAGE2IMAGE_PIPELINE_MAPPING = OrderedDict(
    [
        # Animated Diffusion Pipelines
        ("animatediff_rgb", AnimateDiffPipeline),
        ("animatediff_scribble", AnimateDiffPipeline),
        ("animatediff_sdxl_beta", AnimateDiffSDXLPipeline),
        ("animatediff_v1", AnimateDiffPipeline),
        ("animatediff_v2", AnimateDiffPipeline),
        ("animatediff_v3", AnimateDiffPipeline),

        # Cascade Stages
        ("stable_cascade_stage_b", None),
        ("stable_cascade_stage_b_lite", None),
        ("stable_cascade_stage_c", None),
        ("stable_cascade_stage_c_lite", None),

        # ControlNet Pipelines
        ("controlnet", StableDiffusionControlNetImg2ImgPipeline),
        ("controlnet_xl", StableDiffusionXLControlNetImg2ImgPipeline),
        ("controlnet_xl_large", StableDiffusionXLControlNetImg2ImgPipeline),
        ("controlnet_xl_mid", StableDiffusionXLControlNetImg2ImgPipeline),
        ("controlnet_xl_small", StableDiffusionXLControlNetImg2ImgPipeline),

        # Flux Pipelines
        ("flux_dev", FluxImg2ImgPipeline),
        ("flux_schnell", FluxImg2ImgPipeline),

        # Stable Diffusion Pipelines
        ("sd3", StableDiffusion3Img2ImgPipeline),
        ("sd35_large", StableDiffusion3Img2ImgPipeline),
        ("v1", StableDiffusionImg2ImgPipeline),
        ("v2", StableDiffusionImg2ImgPipeline),

        # Stable Diffusion XL Pipelines
        ("playground-v2-5", StableDiffusionXLImg2ImgPipeline),
        ("xl_base", StableDiffusionXLImg2ImgPipeline),
        ("xl_refiner", StableDiffusionXLImg2ImgPipeline),

        # Inpainting Pipelines
        ("inpainting", None),
        ("inpainting_v2", None),
        ("xl_inpaint", None),

        # Others
        ("upscale", StableDiffusionUpscalePipeline),
    ]
)

SINGLE_FILE_CHECKPOINT_INPAINT_PIPELINE_MAPPING = OrderedDict(
    [
        # Animated Diffusion Pipelines
        ("animatediff_rgb", None),
        ("animatediff_scribble", None),
        ("animatediff_sdxl_beta", None),
        ("animatediff_v1", None),
        ("animatediff_v2", None),
        ("animatediff_v3", None),

        # Cascade Stages
        ("stable_cascade_stage_b", None),
        ("stable_cascade_stage_b_lite", None),
        ("stable_cascade_stage_c", None),
        ("stable_cascade_stage_c_lite", None),

        # ControlNet Pipelines
        ("controlnet", None),
        ("controlnet_xl", None),
        ("controlnet_xl_large", None),
        ("controlnet_xl_mid", None),
        ("controlnet_xl_small", None),

        # Flux Pipelines
        ("flux_dev", None),
        ("flux_schnell", None),

        # Stable Diffusion Pipelines
        ("sd3", None),
        ("sd35_large", None),
        ("v1", None),
        ("v2", None),

        # Stable Diffusion XL Pipelines
        ("playground-v2-5", None),
        ("xl_base", None),
        ("xl_refiner", None),

        # Inpainting Pipelines
        ("inpainting", StableDiffusionInpaintPipeline),
        ("inpainting_v2", StableDiffusionInpaintPipeline),
        ("xl_inpaint", StableDiffusionXLInpaintPipeline),

        # Others
        ("upscale", StableDiffusionUpscalePipeline),
    ]
)

if is_sentencepiece_available():
    from .kolors import KolorsImg2ImgPipeline, KolorsPipeline
    from .pag import KolorsPAGPipeline

    AUTO_TEXT2IMAGE_PIPELINES_MAPPING["kolors"] = KolorsPipeline
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING["kolors-pag"] = KolorsPAGPipeline
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["kolors"] = KolorsImg2ImgPipeline

SUPPORTED_TASKS_MAPPINGS = [
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
    AUTO_INPAINT_PIPELINES_MAPPING,
    _AUTO_TEXT2IMAGE_DECODER_PIPELINES_MAPPING,
    _AUTO_IMAGE2IMAGE_DECODER_PIPELINES_MAPPING,
    _AUTO_INPAINT_DECODER_PIPELINES_MAPPING,
]


def _get_connected_pipeline(pipeline_cls):
    # for now connected pipelines can only be loaded from decoder pipelines, such as kandinsky-community/kandinsky-2-2-decoder
    if pipeline_cls in _AUTO_TEXT2IMAGE_DECODER_PIPELINES_MAPPING.values():
        return _get_task_class(
            AUTO_TEXT2IMAGE_PIPELINES_MAPPING, pipeline_cls.__name__, throw_error_if_not_exist=False
        )
    if pipeline_cls in _AUTO_IMAGE2IMAGE_DECODER_PIPELINES_MAPPING.values():
        return _get_task_class(
            AUTO_IMAGE2IMAGE_PIPELINES_MAPPING, pipeline_cls.__name__, throw_error_if_not_exist=False
        )
    if pipeline_cls in _AUTO_INPAINT_DECODER_PIPELINES_MAPPING.values():
        return _get_task_class(AUTO_INPAINT_PIPELINES_MAPPING, pipeline_cls.__name__, throw_error_if_not_exist=False)


def _get_task_class(mapping, pipeline_class_name, throw_error_if_not_exist: bool = True):
    def get_model(pipeline_class_name):
        for task_mapping in SUPPORTED_TASKS_MAPPINGS:
            for model_name, pipeline in task_mapping.items():
                if pipeline.__name__ == pipeline_class_name:
                    return model_name

    model_name = get_model(pipeline_class_name)

    if model_name is not None:
        task_class = mapping.get(model_name, None)
        if task_class is not None:
            return task_class

    if throw_error_if_not_exist:
        raise ValueError(f"AutoPipeline can't find a pipeline linked to {pipeline_class_name} for {model_name}")


# Copied from diffusers.examples.model_search.pipeline_easy.get_keyword_types
@validate_hf_hub_args
def auto_load_single_checkpoint(pretrained_model_or_path, pipeline_mapping, **kwargs):
    r"""
    Instantiate a [`DiffusionPipeline`] from pretrained pipeline weights saved in the `.ckpt` or `.safetensors`
    format. The pipeline is set in evaluation mode (`model.eval()`) by default.

    Parameters:
        pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
            Can be either:
                - A link to the `.ckpt` file (for example
                  `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                - A path to a *file* containing all pipeline weights.
        pipeline_mapping (dict):
            A dictionary mapping model types to their corresponding pipeline classes.
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Override the default `torch.dtype` and load the model with another dtype.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        cache_dir (`Union[str, os.PathLike]`, *optional*):
            Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
            is not used.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether to only load local model weights and configuration files or not. If set to `True`, the model
            won't be downloaded from the Hub.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
            `diffusers-cli login` (stored in `~/.huggingface`) is used.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
            allowed by Git.
        original_config_file (`str`, *optional*):
            The path to the original config file that was used to train the model. If not provided, the config file
            will be inferred from the checkpoint file.
        config (`str`, *optional*):
            Can be either:
                - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                  hosted on the Hub.
                - A path to a *directory* (for example `./my_pipeline_directory/`) containing the pipeline
                  component configs in Diffusers format.
        checkpoint (`dict`, *optional*):
            The loaded state dictionary of the model.
        kwargs (remaining dictionary of keyword arguments, *optional*):
            Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
            class). The overwritten components are passed directly to the pipelines `__init__` method. See example
            below for more information.
    """

    # Load the checkpoint from the provided link or path
    checkpoint = load_single_file_checkpoint(pretrained_model_or_path)

    # Infer the model type from the loaded checkpoint
    model_type = infer_diffusers_model_type(checkpoint)

    # Get the corresponding pipeline class from the pipeline mapping
    pipeline_class = pipeline_mapping.get(model_type, None)

    # For tasks not supported by this pipeline
    if pipeline_class is None:
        raise ValueError(
            f"{model_type} is not supported in this pipeline."
            "For `Text2Image`, please use `AutoPipelineForText2Image.from_pretrained`, "
            "for `Image2Image` , please use `AutoPipelineForImage2Image.from_pretrained`, "
            "and `inpaint` is only supported in `AutoPipelineForInpainting.from_pretrained`"
        )

    else:
        # Instantiate and return the pipeline with the loaded checkpoint and any additional kwargs
        return pipeline_class.from_auto(pretrained_model_or_path, **kwargs)


class AutoPipelineForText2Image(ConfigMixin):
    r"""

    [`AutoPipelineForText2Image`] is a generic pipeline class that instantiates a text-to-image pipeline class. The
    specific underlying pipeline class is automatically selected from either the
    [`~AutoPipelineForText2Image.from_pretrained`] or [`~AutoPipelineForText2Image.from_pipe`] methods.

    This class cannot be instantiated using `__init__()` (throws an error).

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.

    """

    config_name = "model_index.json"

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
        Instantiates a text-to-image Pytorch diffusion pipeline from pretrained pipeline weight.

        The from_pretrained() method takes care of returning the correct pipeline class instance by:
            1. Detect the pipeline class of the pretrained_model_or_path based on the _class_name property of its
               config object
            2. Find the text-to-image pipeline linked to the pipeline class using pattern matching on pipeline class
               name.

        If a `controlnet` argument is passed, it will instantiate a [`StableDiffusionControlNetPipeline`] object.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```

        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForText2Image

        >>> pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> image = pipeline(prompt).images[0]
        ```
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        load_config_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "token": token,
            "local_files_only": local_files_only,
            "revision": revision,
        }

        config = cls.load_config(pretrained_model_or_path, **load_config_kwargs)
        orig_class_name = config["_class_name"]

        if "controlnet" in kwargs:
            orig_class_name = config["_class_name"].replace("Pipeline", "ControlNetPipeline")
        if "enable_pag" in kwargs:
            enable_pag = kwargs.pop("enable_pag")
            if enable_pag:
                orig_class_name = orig_class_name.replace("Pipeline", "PAGPipeline")

        text_2_image_cls = _get_task_class(AUTO_TEXT2IMAGE_PIPELINES_MAPPING, orig_class_name)

        kwargs = {**load_config_kwargs, **kwargs}
        return text_2_image_cls.from_pretrained(pretrained_model_or_path, **kwargs)

    @classmethod
    def from_pipe(cls, pipeline, **kwargs):
        r"""
        Instantiates a text-to-image Pytorch diffusion pipeline from another instantiated diffusion pipeline class.

        The from_pipe() method takes care of returning the correct pipeline class instance by finding the text-to-image
        pipeline linked to the pipeline class using pattern matching on pipeline class name.

        All the modules the pipeline contains will be used to initialize the new pipeline without reallocating
        additional memory.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pipeline (`DiffusionPipeline`):
                an instantiated `DiffusionPipeline` object

        ```py
        >>> from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

        >>> pipe_i2i = AutoPipelineForImage2Image.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", requires_safety_checker=False
        ... )

        >>> pipe_t2i = AutoPipelineForText2Image.from_pipe(pipe_i2i)
        >>> image = pipe_t2i(prompt).images[0]
        ```
        """

        original_config = dict(pipeline.config)
        original_cls_name = pipeline.__class__.__name__

        # derive the pipeline class to instantiate
        text_2_image_cls = _get_task_class(AUTO_TEXT2IMAGE_PIPELINES_MAPPING, original_cls_name)

        if "controlnet" in kwargs:
            if kwargs["controlnet"] is not None:
                to_replace = "PAGPipeline" if "PAG" in text_2_image_cls.__name__ else "Pipeline"
                text_2_image_cls = _get_task_class(
                    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
                    text_2_image_cls.__name__.replace("ControlNet", "").replace(to_replace, "ControlNet" + to_replace),
                )
            else:
                text_2_image_cls = _get_task_class(
                    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
                    text_2_image_cls.__name__.replace("ControlNet", ""),
                )

        if "enable_pag" in kwargs:
            enable_pag = kwargs.pop("enable_pag")
            if enable_pag:
                text_2_image_cls = _get_task_class(
                    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
                    text_2_image_cls.__name__.replace("PAG", "").replace("Pipeline", "PAGPipeline"),
                )
            else:
                text_2_image_cls = _get_task_class(
                    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
                    text_2_image_cls.__name__.replace("PAG", ""),
                )

        # define expected module and optional kwargs given the pipeline signature
        expected_modules, optional_kwargs = text_2_image_cls._get_signature_keys(text_2_image_cls)

        pretrained_model_name_or_path = original_config.pop("_name_or_path", None)

        # allow users pass modules in `kwargs` to override the original pipeline's components
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        original_class_obj = {
            k: pipeline.components[k]
            for k, v in pipeline.components.items()
            if k in expected_modules and k not in passed_class_obj
        }

        # allow users pass optional kwargs to override the original pipelines config attribute
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}
        original_pipe_kwargs = {
            k: original_config[k]
            for k, v in original_config.items()
            if k in optional_kwargs and k not in passed_pipe_kwargs
        }

        # config that were not expected by original pipeline is stored as private attribute
        # we will pass them as optional arguments if they can be accepted by the pipeline
        additional_pipe_kwargs = [
            k[1:]
            for k in original_config.keys()
            if k.startswith("_") and k[1:] in optional_kwargs and k[1:] not in passed_pipe_kwargs
        ]
        for k in additional_pipe_kwargs:
            original_pipe_kwargs[k] = original_config.pop(f"_{k}")

        text_2_image_kwargs = {**passed_class_obj, **original_class_obj, **passed_pipe_kwargs, **original_pipe_kwargs}

        # store unused config as private attribute
        unused_original_config = {
            f"{'' if k.startswith('_') else '_'}{k}": original_config[k]
            for k, v in original_config.items()
            if k not in text_2_image_kwargs
        }

        missing_modules = set(expected_modules) - set(pipeline._optional_components) - set(text_2_image_kwargs.keys())

        if len(missing_modules) > 0:
            raise ValueError(
                f"Pipeline {text_2_image_cls} expected {expected_modules}, but only {set(list(passed_class_obj.keys()) + list(original_class_obj.keys()))} were passed"
            )

        model = text_2_image_cls(**text_2_image_kwargs)
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        model.register_to_config(**unused_original_config)

        return model

    @classmethod
    def from_auto(cls, pretrained_model_or_path, **kwargs):
        r"""
        Instantiate a PyTorch diffusion pipeline from pretrained pipeline weights.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```

        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            custom_pipeline (`str`, *optional*):

                <Tip warning={true}>

                🧪 This is an experimental feature and may change in the future.

                </Tip>

                Can be either:

                    - A string, the *repo id* (for example `hf-internal-testing/diffusers-dummy-pipeline`) of a custom
                      pipeline hosted on the Hub. The repository must contain a file called pipeline.py that defines
                      the custom pipeline.
                    - A string, the *file name* of a community pipeline hosted on GitHub under
                      [Community](https://github.com/huggingface/diffusers/tree/main/examples/community). Valid file
                      names must match the file name and not the pipeline script (`clip_guided_stable_diffusion`
                      instead of `clip_guided_stable_diffusion.py`). Community pipelines are always loaded from the
                      current main branch of GitHub.
                    - A path to a directory (`./my_pipeline_directory/`) containing a custom pipeline. The directory
                      must contain a file called `pipeline.py` that defines the custom pipeline.

                For more information on how to load and create custom pipelines, please have a look at [Loading and
                Adding Custom
                Pipelines](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview)
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. Defaults to the latest stable 🤗 Diffusers
                version.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            use_onnx (`bool`, *optional*, defaults to `None`):
                If set to `True`, ONNX weights will always be downloaded if present. If set to `False`, ONNX weights
                will never be downloaded. By default `use_onnx` defaults to the `_is_onnx` class attribute which is
                `False` for non-ONNX pipelines and `True` for ONNX pipelines. ONNX weights include both files ending
                with `.onnx` and `.pb`.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Example_1:
        ```py
        >>> from diffusers import StableDiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = StableDiffusionPipeline.from_auto("stabilityai/stable-diffusion-2-1").to("cuda")
        ```

        Example_2:
        ```py
        >>> from diffusers import StableDiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = StableDiffusionPipeline.from_auto(
        ...     "https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.safetensors"
        ... ).to("cuda")
        ```
        """

        # Get the keyword types for the provided model path
        hf_model_status = get_keyword_types(pretrained_model_or_path)

        # Check if it is loadable.
        if hf_model_status["loading_method"] is not None:
            if hf_model_status["checkpoint_format"] == "single_file":
                # For single file checkpoint
                return auto_load_single_checkpoint(
                    pretrained_model_or_path=pretrained_model_or_path,
                    pipeline_mapping=SINGLE_FILE_CHECKPOINT_TEXT2IMAGE_PIPELINE_MAPPING,
                    **kwargs,
                )
            else:
                # Load the pipeline using the standard from_pretrained method if not a single file checkpoint
                return cls.from_pretrained(pretrained_model_or_path, **kwargs)
        else:
            # Exception handling when loading is not possible
            raise ValueError(f"Invalid path or URL: {pretrained_model_or_path}")


class AutoPipelineForImage2Image(ConfigMixin):
    r"""

    [`AutoPipelineForImage2Image`] is a generic pipeline class that instantiates an image-to-image pipeline class. The
    specific underlying pipeline class is automatically selected from either the
    [`~AutoPipelineForImage2Image.from_pretrained`] or [`~AutoPipelineForImage2Image.from_pipe`] methods.

    This class cannot be instantiated using `__init__()` (throws an error).

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.

    """

    config_name = "model_index.json"

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
        Instantiates a image-to-image Pytorch diffusion pipeline from pretrained pipeline weight.

        The from_pretrained() method takes care of returning the correct pipeline class instance by:
            1. Detect the pipeline class of the pretrained_model_or_path based on the _class_name property of its
               config object
            2. Find the image-to-image pipeline linked to the pipeline class using pattern matching on pipeline class
               name.

        If a `controlnet` argument is passed, it will instantiate a [`StableDiffusionControlNetImg2ImgPipeline`]
        object.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```

        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForImage2Image

        >>> pipeline = AutoPipelineForImage2Image.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> image = pipeline(prompt, image).images[0]
        ```
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        load_config_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "token": token,
            "local_files_only": local_files_only,
            "revision": revision,
        }

        config = cls.load_config(pretrained_model_or_path, **load_config_kwargs)
        orig_class_name = config["_class_name"]

        # the `orig_class_name` can be:
        # `- *Pipeline` (for regular text-to-image checkpoint)
        # `- *Img2ImgPipeline` (for refiner checkpoint)
        to_replace = "Img2ImgPipeline" if "Img2Img" in config["_class_name"] else "Pipeline"

        if "controlnet" in kwargs:
            orig_class_name = orig_class_name.replace(to_replace, "ControlNet" + to_replace)
        if "enable_pag" in kwargs:
            enable_pag = kwargs.pop("enable_pag")
            if enable_pag:
                orig_class_name = orig_class_name.replace(to_replace, "PAG" + to_replace)

        image_2_image_cls = _get_task_class(AUTO_IMAGE2IMAGE_PIPELINES_MAPPING, orig_class_name)

        kwargs = {**load_config_kwargs, **kwargs}
        return image_2_image_cls.from_pretrained(pretrained_model_or_path, **kwargs)

    @classmethod
    def from_pipe(cls, pipeline, **kwargs):
        r"""
        Instantiates a image-to-image Pytorch diffusion pipeline from another instantiated diffusion pipeline class.

        The from_pipe() method takes care of returning the correct pipeline class instance by finding the
        image-to-image pipeline linked to the pipeline class using pattern matching on pipeline class name.

        All the modules the pipeline contains will be used to initialize the new pipeline without reallocating
        additional memory.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pipeline (`DiffusionPipeline`):
                an instantiated `DiffusionPipeline` object

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

        >>> pipe_t2i = AutoPipelineForText2Image.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", requires_safety_checker=False
        ... )

        >>> pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i)
        >>> image = pipe_i2i(prompt, image).images[0]
        ```
        """

        original_config = dict(pipeline.config)
        original_cls_name = pipeline.__class__.__name__

        # derive the pipeline class to instantiate
        image_2_image_cls = _get_task_class(AUTO_IMAGE2IMAGE_PIPELINES_MAPPING, original_cls_name)

        if "controlnet" in kwargs:
            if kwargs["controlnet"] is not None:
                to_replace = "Img2ImgPipeline"
                if "PAG" in image_2_image_cls.__name__:
                    to_replace = "PAG" + to_replace
                image_2_image_cls = _get_task_class(
                    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
                    image_2_image_cls.__name__.replace("ControlNet", "").replace(
                        to_replace, "ControlNet" + to_replace
                    ),
                )
            else:
                image_2_image_cls = _get_task_class(
                    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
                    image_2_image_cls.__name__.replace("ControlNet", ""),
                )

        if "enable_pag" in kwargs:
            enable_pag = kwargs.pop("enable_pag")
            if enable_pag:
                image_2_image_cls = _get_task_class(
                    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
                    image_2_image_cls.__name__.replace("PAG", "").replace("Img2ImgPipeline", "PAGImg2ImgPipeline"),
                )
            else:
                image_2_image_cls = _get_task_class(
                    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
                    image_2_image_cls.__name__.replace("PAG", ""),
                )

        # define expected module and optional kwargs given the pipeline signature
        expected_modules, optional_kwargs = image_2_image_cls._get_signature_keys(image_2_image_cls)

        pretrained_model_name_or_path = original_config.pop("_name_or_path", None)

        # allow users pass modules in `kwargs` to override the original pipeline's components
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        original_class_obj = {
            k: pipeline.components[k]
            for k, v in pipeline.components.items()
            if k in expected_modules and k not in passed_class_obj
        }

        # allow users pass optional kwargs to override the original pipelines config attribute
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}
        original_pipe_kwargs = {
            k: original_config[k]
            for k, v in original_config.items()
            if k in optional_kwargs and k not in passed_pipe_kwargs
        }

        # config attribute that were not expected by original pipeline is stored as its private attribute
        # we will pass them as optional arguments if they can be accepted by the pipeline
        additional_pipe_kwargs = [
            k[1:]
            for k in original_config.keys()
            if k.startswith("_") and k[1:] in optional_kwargs and k[1:] not in passed_pipe_kwargs
        ]
        for k in additional_pipe_kwargs:
            original_pipe_kwargs[k] = original_config.pop(f"_{k}")

        image_2_image_kwargs = {**passed_class_obj, **original_class_obj, **passed_pipe_kwargs, **original_pipe_kwargs}

        # store unused config as private attribute
        unused_original_config = {
            f"{'' if k.startswith('_') else '_'}{k}": original_config[k]
            for k, v in original_config.items()
            if k not in image_2_image_kwargs
        }

        missing_modules = set(expected_modules) - set(pipeline._optional_components) - set(image_2_image_kwargs.keys())

        if len(missing_modules) > 0:
            raise ValueError(
                f"Pipeline {image_2_image_cls} expected {expected_modules}, but only {set(list(passed_class_obj.keys()) + list(original_class_obj.keys()))} were passed"
            )

        model = image_2_image_cls(**image_2_image_kwargs)
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        model.register_to_config(**unused_original_config)

        return model

    @classmethod
    def from_auto(cls, pretrained_model_or_path, **kwargs):
        r"""
        Instantiate a PyTorch diffusion pipeline from pretrained pipeline weights.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```

        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            custom_pipeline (`str`, *optional*):

                <Tip warning={true}>

                🧪 This is an experimental feature and may change in the future.

                </Tip>

                Can be either:

                    - A string, the *repo id* (for example `hf-internal-testing/diffusers-dummy-pipeline`) of a custom
                      pipeline hosted on the Hub. The repository must contain a file called pipeline.py that defines
                      the custom pipeline.
                    - A string, the *file name* of a community pipeline hosted on GitHub under
                      [Community](https://github.com/huggingface/diffusers/tree/main/examples/community). Valid file
                      names must match the file name and not the pipeline script (`clip_guided_stable_diffusion`
                      instead of `clip_guided_stable_diffusion.py`). Community pipelines are always loaded from the
                      current main branch of GitHub.
                    - A path to a directory (`./my_pipeline_directory/`) containing a custom pipeline. The directory
                      must contain a file called `pipeline.py` that defines the custom pipeline.

                For more information on how to load and create custom pipelines, please have a look at [Loading and
                Adding Custom
                Pipelines](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview)
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. Defaults to the latest stable 🤗 Diffusers
                version.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            use_onnx (`bool`, *optional*, defaults to `None`):
                If set to `True`, ONNX weights will always be downloaded if present. If set to `False`, ONNX weights
                will never be downloaded. By default `use_onnx` defaults to the `_is_onnx` class attribute which is
                `False` for non-ONNX pipelines and `True` for ONNX pipelines. ONNX weights include both files ending
                with `.onnx` and `.pb`.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Example_1:
        ```py
        >>> from diffusers import StableDiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = StableDiffusionPipeline.from_auto("stabilityai/stable-diffusion-2-1").to("cuda")
        ```

        Example_2:
        ```py
        >>> from diffusers import StableDiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = StableDiffusionPipeline.from_auto(
        ...     "https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.safetensors"
        ... ).to("cuda")
        ```
        """

        # Get the keyword types for the provided model path
        hf_model_status = get_keyword_types(pretrained_model_or_path)

        # Check if it is loadable.
        if hf_model_status["loading_method"] is not None:
            if hf_model_status["checkpoint_format"] == "single_file":
                # For single file checkpoint
                return auto_load_single_checkpoint(
                    pretrained_model_or_path=pretrained_model_or_path,
                    pipeline_mapping=SINGLE_FILE_CHECKPOINT_IMAGE2IMAGE_PIPELINE_MAPPING,
                    **kwargs,
                )
            else:
                # Load the pipeline using the standard from_pretrained method if not a single file checkpoint
                return cls.from_pretrained(pretrained_model_or_path, **kwargs)
        else:
            # Exception handling when loading is not possible
            raise ValueError(f"Invalid path or URL: {pretrained_model_or_path}")


class AutoPipelineForInpainting(ConfigMixin):
    r"""

    [`AutoPipelineForInpainting`] is a generic pipeline class that instantiates an inpainting pipeline class. The
    specific underlying pipeline class is automatically selected from either the
    [`~AutoPipelineForInpainting.from_pretrained`] or [`~AutoPipelineForInpainting.from_pipe`] methods.

    This class cannot be instantiated using `__init__()` (throws an error).

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.

    """

    config_name = "model_index.json"

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
        Instantiates a inpainting Pytorch diffusion pipeline from pretrained pipeline weight.

        The from_pretrained() method takes care of returning the correct pipeline class instance by:
            1. Detect the pipeline class of the pretrained_model_or_path based on the _class_name property of its
               config object
            2. Find the inpainting pipeline linked to the pipeline class using pattern matching on pipeline class name.

        If a `controlnet` argument is passed, it will instantiate a [`StableDiffusionControlNetInpaintPipeline`]
        object.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```

        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForInpainting

        >>> pipeline = AutoPipelineForInpainting.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> image = pipeline(prompt, image=init_image, mask_image=mask_image).images[0]
        ```
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        load_config_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "token": token,
            "local_files_only": local_files_only,
            "revision": revision,
        }

        config = cls.load_config(pretrained_model_or_path, **load_config_kwargs)
        orig_class_name = config["_class_name"]

        # The `orig_class_name`` can be:
        # `- *InpaintPipeline` (for inpaint-specific checkpoint)
        #  - or *Pipeline (for regular text-to-image checkpoint)
        to_replace = "InpaintPipeline" if "Inpaint" in config["_class_name"] else "Pipeline"

        if "controlnet" in kwargs:
            orig_class_name = orig_class_name.replace(to_replace, "ControlNet" + to_replace)
        if "enable_pag" in kwargs:
            enable_pag = kwargs.pop("enable_pag")
            if enable_pag:
                orig_class_name = orig_class_name.replace(to_replace, "PAG" + to_replace)
        inpainting_cls = _get_task_class(AUTO_INPAINT_PIPELINES_MAPPING, orig_class_name)

        kwargs = {**load_config_kwargs, **kwargs}
        return inpainting_cls.from_pretrained(pretrained_model_or_path, **kwargs)

    @classmethod
    def from_pipe(cls, pipeline, **kwargs):
        r"""
        Instantiates a inpainting Pytorch diffusion pipeline from another instantiated diffusion pipeline class.

        The from_pipe() method takes care of returning the correct pipeline class instance by finding the inpainting
        pipeline linked to the pipeline class using pattern matching on pipeline class name.

        All the modules the pipeline class contain will be used to initialize the new pipeline without reallocating
        additional memory.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pipeline (`DiffusionPipeline`):
                an instantiated `DiffusionPipeline` object

        Examples:

        ```py
        >>> from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting

        >>> pipe_t2i = AutoPipelineForText2Image.from_pretrained(
        ...     "DeepFloyd/IF-I-XL-v1.0", requires_safety_checker=False
        ... )

        >>> pipe_inpaint = AutoPipelineForInpainting.from_pipe(pipe_t2i)
        >>> image = pipe_inpaint(prompt, image=init_image, mask_image=mask_image).images[0]
        ```
        """
        original_config = dict(pipeline.config)
        original_cls_name = pipeline.__class__.__name__

        # derive the pipeline class to instantiate
        inpainting_cls = _get_task_class(AUTO_INPAINT_PIPELINES_MAPPING, original_cls_name)

        if "controlnet" in kwargs:
            if kwargs["controlnet"] is not None:
                inpainting_cls = _get_task_class(
                    AUTO_INPAINT_PIPELINES_MAPPING,
                    inpainting_cls.__name__.replace("ControlNet", "").replace(
                        "InpaintPipeline", "ControlNetInpaintPipeline"
                    ),
                )
            else:
                inpainting_cls = _get_task_class(
                    AUTO_INPAINT_PIPELINES_MAPPING,
                    inpainting_cls.__name__.replace("ControlNetInpaintPipeline", "InpaintPipeline"),
                )

        if "enable_pag" in kwargs:
            enable_pag = kwargs.pop("enable_pag")
            if enable_pag:
                inpainting_cls = _get_task_class(
                    AUTO_INPAINT_PIPELINES_MAPPING,
                    inpainting_cls.__name__.replace("PAG", "").replace("InpaintPipeline", "PAGInpaintPipeline"),
                )
            else:
                inpainting_cls = _get_task_class(
                    AUTO_INPAINT_PIPELINES_MAPPING,
                    inpainting_cls.__name__.replace("PAGInpaintPipeline", "InpaintPipeline"),
                )

        # define expected module and optional kwargs given the pipeline signature
        expected_modules, optional_kwargs = inpainting_cls._get_signature_keys(inpainting_cls)

        pretrained_model_name_or_path = original_config.pop("_name_or_path", None)

        # allow users pass modules in `kwargs` to override the original pipeline's components
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        original_class_obj = {
            k: pipeline.components[k]
            for k, v in pipeline.components.items()
            if k in expected_modules and k not in passed_class_obj
        }

        # allow users pass optional kwargs to override the original pipelines config attribute
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}
        original_pipe_kwargs = {
            k: original_config[k]
            for k, v in original_config.items()
            if k in optional_kwargs and k not in passed_pipe_kwargs
        }

        # config that were not expected by original pipeline is stored as private attribute
        # we will pass them as optional arguments if they can be accepted by the pipeline
        additional_pipe_kwargs = [
            k[1:]
            for k in original_config.keys()
            if k.startswith("_") and k[1:] in optional_kwargs and k[1:] not in passed_pipe_kwargs
        ]
        for k in additional_pipe_kwargs:
            original_pipe_kwargs[k] = original_config.pop(f"_{k}")

        inpainting_kwargs = {**passed_class_obj, **original_class_obj, **passed_pipe_kwargs, **original_pipe_kwargs}

        # store unused config as private attribute
        unused_original_config = {
            f"{'' if k.startswith('_') else '_'}{k}": original_config[k]
            for k, v in original_config.items()
            if k not in inpainting_kwargs
        }

        missing_modules = set(expected_modules) - set(pipeline._optional_components) - set(inpainting_kwargs.keys())

        if len(missing_modules) > 0:
            raise ValueError(
                f"Pipeline {inpainting_cls} expected {expected_modules}, but only {set(list(passed_class_obj.keys()) + list(original_class_obj.keys()))} were passed"
            )

        model = inpainting_cls(**inpainting_kwargs)
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        model.register_to_config(**unused_original_config)

        return model

    @classmethod
    def from_auto(cls, pretrained_model_or_path, **kwargs):
        r"""
        Instantiate a PyTorch diffusion pipeline from pretrained pipeline weights.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```

        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            custom_pipeline (`str`, *optional*):

                <Tip warning={true}>

                🧪 This is an experimental feature and may change in the future.

                </Tip>

                Can be either:

                    - A string, the *repo id* (for example `hf-internal-testing/diffusers-dummy-pipeline`) of a custom
                      pipeline hosted on the Hub. The repository must contain a file called pipeline.py that defines
                      the custom pipeline.
                    - A string, the *file name* of a community pipeline hosted on GitHub under
                      [Community](https://github.com/huggingface/diffusers/tree/main/examples/community). Valid file
                      names must match the file name and not the pipeline script (`clip_guided_stable_diffusion`
                      instead of `clip_guided_stable_diffusion.py`). Community pipelines are always loaded from the
                      current main branch of GitHub.
                    - A path to a directory (`./my_pipeline_directory/`) containing a custom pipeline. The directory
                      must contain a file called `pipeline.py` that defines the custom pipeline.

                For more information on how to load and create custom pipelines, please have a look at [Loading and
                Adding Custom
                Pipelines](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview)
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. Defaults to the latest stable 🤗 Diffusers
                version.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            use_onnx (`bool`, *optional*, defaults to `None`):
                If set to `True`, ONNX weights will always be downloaded if present. If set to `False`, ONNX weights
                will never be downloaded. By default `use_onnx` defaults to the `_is_onnx` class attribute which is
                `False` for non-ONNX pipelines and `True` for ONNX pipelines. ONNX weights include both files ending
                with `.onnx` and `.pb`.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Example_1:
        ```py
        >>> from diffusers import StableDiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = StableDiffusionPipeline.from_auto("stabilityai/stable-diffusion-2-1").to("cuda")
        ```

        Example_2:
        ```py
        >>> from diffusers import StableDiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = StableDiffusionPipeline.from_auto(
        ...     "https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.safetensors"
        ... ).to("cuda")
        ```
        """

        # Get the keyword types for the provided model path
        hf_model_status = get_keyword_types(pretrained_model_or_path)

        # Check if it is loadable.
        if hf_model_status["loading_method"] is not None:
            if hf_model_status["checkpoint_format"] == "single_file":
                # For single file checkpoint
                return auto_load_single_checkpoint(
                    pretrained_model_or_path=pretrained_model_or_path,
                    pipeline_mapping=SINGLE_FILE_CHECKPOINT_INPAINT_PIPELINE_MAPPING,
                    **kwargs,
                )
            else:
                # Load the pipeline using the standard from_pretrained method if not a single file checkpoint
                return cls.from_pretrained(pretrained_model_or_path, **kwargs)
        else:
            # Exception handling when loading is not possible
            raise ValueError(f"Invalid path or URL: {pretrained_model_or_path}")
