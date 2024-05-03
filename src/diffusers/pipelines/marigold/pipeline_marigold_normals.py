# Copyright 2024 Marigold authors, PRS ETH Zurich. All rights reserved.
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
# --------------------------------------------------------------------------
# More information and citation instructions are available on the
# Marigold project website: https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from ...image_processor import PipelineImageInput
from ...models import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from ...schedulers import (
    DDIMScheduler,
    LCMScheduler,
)
from ...utils import (
    BaseOutput,
    deprecate,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = f"""
Examples:
```py
>>> import requests
>>> from diffusers import MarigoldNormalsPipeline
>>> from PIL import Image

>>> pipe = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-lcm-v0-1")
>>> pipe = pipe.to("cuda")

>>> image = Image.open(requests.get("https://marigoldmonodepth.github.io/images/einstein.jpg", stream=True).raw)
>>> depth = pipe(image, preset="fast", output_visualization=True)

>>> depth.visualization.save("einstein_normals.png")
```
"""


def resize_maybe_antialias(image: torch.Tensor, size: Tuple[int, int], mode: str, is_aa: bool = None) -> torch.Tensor:
    assert image.dim() == 4 and isinstance(is_aa, bool)

    antialias = is_aa and mode in ("bilinear", "bicubic")
    image = F.interpolate(image, size, mode=mode, antialias=antialias)

    return image


def resize_to_max_edge(image: torch.Tensor, max_edge_sz: int, mode: str) -> torch.Tensor:
    assert image.dim() == 4

    h, w = image.shape[-2:]
    max_orig = max(h, w)
    new_h = h * max_edge_sz // max_orig
    new_w = w * max_edge_sz // max_orig

    if new_h == 0 or new_w == 0:
        raise ValueError(f"Extreme aspect ratio of the input image: [{w} x {h}]")

    image = resize_maybe_antialias(image, (new_h, new_w), mode, is_aa=True)

    return image


def pad_image(image: torch.Tensor, align: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    assert image.dim() == 4

    h, w = image.shape[-2:]
    ph, pw = -h % align, -w % align

    image = F.pad(image, (0, pw, 0, ph), mode="replicate")

    return image, (ph, pw)


def unpad_image(image: torch.Tensor, padding: Tuple[int, int]) -> torch.Tensor:
    assert image.dim() == 4

    ph, pw = padding
    uh = None if ph == 0 else -ph
    uw = None if pw == 0 else -pw

    image = image[:, :, :uh, :uw]

    return image


def load_image_canonical(image: Union[torch.Tensor, np.ndarray, Image.Image]) -> Tuple[torch.Tensor, int]:
    if isinstance(image, Image.Image):
        image = np.array(image)

    input_dtype_max = None
    if isinstance(image, np.ndarray):
        if np.issubdtype(image.dtype, np.integer) and not np.issubdtype(image.dtype, np.unsignedinteger):
            raise ValueError(f"Input image dtype={image.dtype} cannot be a signed integer.")
        if np.issubdtype(image.dtype, np.complexfloating):
            raise ValueError(f"Input image dtype={image.dtype} cannot be complex.")
        if np.issubdtype(image.dtype, bool):
            raise ValueError(f"Input image dtype={image.dtype} cannot be boolean.")
        if np.issubdtype(image.dtype, np.unsignedinteger):
            input_dtype_max = np.iinfo(image.dtype).max
            image = image.astype(np.float32)  # because torch does not have unsigned dtypes beyond torch.uint8
        image = torch.from_numpy(image)

    if torch.is_tensor(image) and not torch.is_floating_point(image) and input_dtype_max is None:
        if image.dtype != torch.uint8:
            raise ValueError(f"Image dtype={image.dtype} is not supported.")
        input_dtype_max = 255

    if image.dim() == 2:
        image = image.unsqueeze(0).repeat(3, 1, 1)  # [3,H,W]
    elif image.dim() == 3:
        if image.shape[2] in (1, 3):
            image = image.permute(2, 0, 1)  # [1|3,H,W]
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)  # [3,H,W]
        if image.shape[0] != 3:
            raise ValueError(f"Input image is not 1- or 3-channel: {image.shape}.")
    else:
        raise ValueError("Input image is not a 2- or 3-dimensional tensor.")

    image = image.unsqueeze(0)  # [1,3,H,W]

    return image, input_dtype_max


def check_image_values_range(image: torch.FloatTensor) -> None:
    assert torch.is_floating_point(image)

    val_min = image.min().item()
    val_max = image.max().item()

    if val_min < -1.0 or val_max > 1.0:
        raise ValueError("Input image data is partially outside of the [-1,1] range.")
    if val_min >= 0.0:
        logger.warning(
            "Input image data is entirely in the [0,1] range; expecting [-1,1]. "
            "This could be an issue with normalization"
        )


def validate_vae(vae: torch.nn.Module) -> None:
    if len(vae.config.down_block_types) != 4 or len(vae.config.up_block_types) != 4:
        raise ValueError(f"Unexpected VAE down_block_types or up_block_types; {vae.config}")
    if vae.config.in_channels != 3 or vae.config.out_channels != 3:
        raise ValueError(f"Unexpected VAE input-output space size; {vae.config}")
    if vae.config.latent_channels != 4:
        raise ValueError(f"Unexpected VAE latent space size; {vae.config}")


def find_batch_size(ensemble_size: int, resolution: int, dtype: torch.dtype) -> int:
    bs_search_table = [
        # tested on A100-PCIE-80GB
        {"res": 768, "total_vram": 79, "bs": 36, "dtype": torch.float32},
        {"res": 1024, "total_vram": 79, "bs": 20, "dtype": torch.float32},
        # tested on A100-PCIE-40GB
        {"res": 768, "total_vram": 39, "bs": 18, "dtype": torch.float32},
        {"res": 1024, "total_vram": 39, "bs": 10, "dtype": torch.float32},
        {"res": 768, "total_vram": 39, "bs": 30, "dtype": torch.float16},
        {"res": 1024, "total_vram": 39, "bs": 15, "dtype": torch.float16},
        # tested on RTX3090, RTX4090
        {"res": 512, "total_vram": 23, "bs": 22, "dtype": torch.float32},
        {"res": 768, "total_vram": 23, "bs": 9, "dtype": torch.float32},
        {"res": 1024, "total_vram": 23, "bs": 5, "dtype": torch.float32},
        {"res": 512, "total_vram": 23, "bs": 40, "dtype": torch.float16},
        {"res": 768, "total_vram": 23, "bs": 18, "dtype": torch.float16},
        {"res": 1024, "total_vram": 23, "bs": 10, "dtype": torch.float16},
        # tested on GTX1080Ti
        {"res": 512, "total_vram": 10, "bs": 6, "dtype": torch.float32},
        {"res": 768, "total_vram": 10, "bs": 2, "dtype": torch.float32},
        {"res": 1024, "total_vram": 10, "bs": 1, "dtype": torch.float32},
        {"res": 512, "total_vram": 10, "bs": 10, "dtype": torch.float16},
        {"res": 768, "total_vram": 10, "bs": 5, "dtype": torch.float16},
        {"res": 1024, "total_vram": 10, "bs": 3, "dtype": torch.float16},
    ]

    if not torch.cuda.is_available():
        logger.info("No GPU available, using batch_size=1")
        return 1

    total_vram = torch.cuda.mem_get_info()[1] / 1024.0**3
    filtered_bs_search_table = [s for s in bs_search_table if s["dtype"] == dtype]
    for settings in sorted(
        filtered_bs_search_table,
        key=lambda k: (k["res"], -k["total_vram"]),
    ):
        if resolution <= settings["res"] and total_vram >= settings["total_vram"]:
            bs = settings["bs"]
            if bs > ensemble_size:
                bs = ensemble_size
            elif (ensemble_size + 1) // 2 < bs < ensemble_size:
                bs = (ensemble_size + 1) // 2
            return bs

    logger.info("Falling back to batch_size=1; feel free to set it manually to a higher value.")

    return 1


def normalize_normals(normals: torch.FloatTensor, eps: float = 1e-6) -> torch.FloatTensor:
    assert normals.dim() == 4

    norm = torch.norm(normals, dim=1, keepdim=True)
    normals /= norm.clamp(min=eps)

    return normals


def ensemble_normals(
    normals: torch.FloatTensor, output_uncertainty: bool, reduction: str = "closest"
) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
    assert normals.dim() == 4
    assert reduction in ("closest", "mean")

    E, C, H, W = normals.shape
    assert C == 3

    mean_normals = normals.mean(dim=0, keepdim=True)  # [1,3,H,W]
    mean_normals = normalize_normals(mean_normals)  # [1,3,H,W]

    sim_cos = (mean_normals * normals).sum(dim=1, keepdim=True)  # [E,1,H,W]

    uncertainty = None
    if output_uncertainty:
        uncertainty = sim_cos.arccos()  # [E,1,H,W]
        uncertainty = uncertainty.mean(dim=0, keepdim=True) / math.pi  # [1,1,H,W]

    if reduction == "mean":
        return mean_normals, uncertainty  # [1,3,H,W], [1,1,H,W]

    closest_indices = sim_cos.argmax(dim=0, keepdim=True)  # [1,1,H,W]
    closest_indices = closest_indices.repeat(1, 3, 1, 1)  # [1,3,H,W]
    closest_normals = torch.gather(normals, 0, closest_indices)

    return closest_normals, uncertainty  # [1,3,H,W], [1,1,H,W]


def visualize_normals(
    normals: torch.FloatTensor,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
) -> Image.Image:
    assert normals.dim() == 4 and normals.shape[:2] == (1, 3)

    visualization = normals.squeeze(0)  # [3,H,W]
    visualization = visualization.permute(1, 2, 0)  # [H,W,3]

    if any((flip_x, flip_y, flip_z)):
        flip_vec = [
            (-1) ** flip_x,
            (-1) ** flip_y,
            (-1) ** flip_z,
        ]
        visualization *= torch.tensor(flip_vec, dtype=normals.dtype, device=normals.device)

    visualization = (visualization + 1.0) * 0.5
    visualization = (visualization * 255).to(dtype=torch.uint8, device="cpu").numpy()
    visualization = Image.fromarray(visualization)

    return visualization


@dataclass
class MarigoldNormalsOutput(BaseOutput):
    """
    Output class for Marigold monocular normals prediction pipeline.

    Args:
        prediction (`np.ndarray`, `torch.FloatTensor`, `List[np.ndarray]`, or `List[torch.FloatTensor]`):
            Predicted normals, with values in the range of [-1, 1].
        visualization (`None`, `PIL.Image.Image`, or List[PIL.Image.Image]):
            Colorized prediction for visualization.
        uncertainty (`None`, `np.ndarray`, torch.FloatTensor, or a `List` of them):
            Uncertainty map computed from the ensemble.
        latent (`None`, `torch.FloatTensor`, or `List[torch.FloatTensor]`):
            Latent features corresponding to the ensemble predictions.
        config (`None` or `dict`):
            A set of configuration parameters used for this prediction. This can be useful for bookkeeping and
            understanding the values resolved automatically, such as batch_size.
    """

    prediction: Union[
        np.ndarray,
        torch.FloatTensor,
        List[np.ndarray],
        List[torch.FloatTensor],
    ]
    visualization: Union[
        None,
        Image.Image,
        List[Image.Image],
    ]
    uncertainty: Union[
        None,
        np.ndarray,
        torch.FloatTensor,
        List[np.ndarray],
        List[torch.FloatTensor],
    ]
    latent: Union[
        None,
        torch.FloatTensor,
        List[torch.FloatTensor],
    ]
    config: Dict[str, Any] = None


class MarigoldNormalsPipeline(DiffusionPipeline):
    """
    Pipeline for monocular normals estimation using the Marigold method: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the normals latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and predictions to and from latent
            representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: DDIMScheduler,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        use_full_z_range: bool = True,
    ):
        super().__init__()

        validate_vae(vae)

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            use_full_z_range=use_full_z_range,
        )

        self.latent_size_scale = 8
        self.latent_space_size = self.vae.config.latent_channels
        self.latent_scaling_factor = self.vae.config.scaling_factor
        self.optimal_processing_resolution = 768
        self.use_full_z_range = use_full_z_range

        self.empty_text_embedding = None

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        input_image: PipelineImageInput,
        preset: Optional[str] = None,
        denoising_steps: Optional[int] = None,
        ensemble_size: Optional[int] = None,
        processing_resolution: Optional[int] = None,
        match_input_resolution: bool = True,
        resample_method_input: str = "bilinear",
        resample_method_output: str = "bilinear",
        batch_size: int = 0,
        save_memory: bool = False,
        check_input: bool = True,
        ensembling_kwargs: Optional[Dict[str, Any]] = None,
        input_latent: Optional[Union[torch.FloatTensor, List[torch.FloatTensor]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_prediction_format: str = "np",
        output_visualization: bool = True,
        output_visualization_kwargs: Optional[Dict[str, Any]] = None,
        output_uncertainty: bool = True,
        output_latent: bool = False,
        **kwargs,
    ) -> MarigoldNormalsOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`,
                    or `List[torch.Tensor]`):
                Input image or images.
            preset (`str`, *optional*, defaults to `None`):
                A preset string, overriding subsets of the other parameters: `denoising_steps`, `ensemble_size`, or
                `processing_resolution`. The default value `None` results in no preset applied.
                - `"fast"`: This setting should be used for fast inference, which may not be the most accurate. Example
                  usage scenario: content creation, conditioning of other generative models and pipelines.
                - `"precise"`: This setting should be used to get the most precise results. Example usage scenario: 3D
                  reconstruction, or when `"fast"` didn't produce desired results.
                - `"paper"`: This setting should be used to obtain results for numerical comparisons with other
                  methods. Example usage scenario: benchmarking, quantitative comparisons, report or paper preparation.
                  NB: Ensure reproducibility by seeding inference using the `generator` parameter.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
            ensemble_size (`int`, *optional*, defaults to `None`):
                Number of ensemble predictions. The default value `None` results in automatic selection. Recommended
                values are 5 and higher for better precision, or 1 for faster inference.
            processing_resolution (`int`, *optional*, defaults to None):
                Effective processing resolution. When set to `0`, matches the larger input image dimension. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` results in automatic selection.
            match_input_resolution (`bool`, *optional*, defaults to `True`):
                When enabled, the output prediction is resized to match the input dimensions. When disabled, the longer
                side of the output will equal to `processing_resolution`.
            resample_method_input: (`str`, *optional*, defaults to `"bilinear"`):
                Resampling method used to resize input images to `processing_resolution`. The accepted values are:
                `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.
            resample_method_output: (`str`, *optional*, defaults to `"bilinear"`):
                Resampling method used to resize output predictions to match the input resolution. The accepted values
                are `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size. Smaller values save memory. The default value `0` results in automatic selection.
            save_memory (`bool`, defaults to `False`):
                Extra steps to save memory at the cost of performance.
            check_input (`bool`, defaults to `False`):
                Extra steps to validate compatibility of the inputs with the model.
            ensembling_kwargs (`dict`, *optional*, defaults to `None`)
                Extra dictionary with arguments for precise ensembling control. The following options are available:
                - reduction (`str`, *optional*, defaults to `"closest"`): Defines the ensembling function applied in
                  every pixel location, can be either "closest" or "mean".
            input_latent (`torch.Tensor`, or `List[torch.Tensor]`, *optional*, defaults to `None`):
                Latent noise tensors to replace the random initialization. These can be taken from the previous
                function call's output.
            generator (`torch.Generator`, or `List[torch.Generator]`, *optional*, defaults to `None`):
                Random number generator object to ensure reproducibility.
            output_prediction_format (`str`, *optional*, defaults to `"np"`):
                Preferred format of the output's `prediction` and the optional `uncertainty` fields. The accepted
                values are: `"np"` (numpy array) or `"pt"` (torch tensor).
            output_visualization (`bool`, *optional*, defaults to `True`):
                When enabled, the output's `visualization` field contains a PIL.Image that can be used for visual
                quality inspection.
            output_visualization_kwargs (`dict`, *optional*, defaults to `None`):
                Extra dictionary with arguments for precise visualization control. Flipping axes leads to a different
                color scheme. The following options are available:
                - flip_x (`bool`, *optional*, defaults to `False`): Flips the X axis of the normals frame of reference.
                  Default direction is right.
                - flip_y (`bool`, *optional*, defaults to `False`): Flips the Y axis of the normals frame of reference.
                  Default direction is top.
                - flip_z (`bool`, *optional*, defaults to `False`): Flips the Z axis of the normals frame of reference.
                  Default direction is facing the observer.
            output_uncertainty (`bool`, *optional*, defaults to `True`):
                When enabled, the output's `uncertainty` field contains the predictive uncertainty map, provided that
                the `ensemble_size` argument is set to a value above 2.
            output_latent (`bool`, *optional*, defaults to `False`):
                When enabled, the output's `latent` field contains the latent codes corresponding to the predictions
                within the ensemble. These codes can be saved, modified, and used for subsequent calls with the
                `input_latent` argument.

        Examples:

        Returns:
            `MarigoldNormalsOutput`: Output class instance for Marigold monocular normals prediction pipeline.
        """

        device = self._execution_device

        # deprecations
        if "processing_res" in kwargs:
            deprecation_message = (
                "`processing_res` is deprecated and it will be removed in a future version. "
                "Use `processing_resolution` instead."
            )
            deprecate("processing_res", "1.0.0", deprecation_message, standard_warn=False)
            processing_resolution = kwargs["processing_res"]
        if "match_input_res" in kwargs:
            deprecation_message = (
                "`match_input_res` is deprecated and it will be removed in a future version. "
                "Use `match_input_resolution` instead."
            )
            deprecate("match_input_res", "1.0.0", deprecation_message, standard_warn=False)
            match_input_resolution = kwargs["match_input_res"]
        if "resample_method" in kwargs:
            deprecation_message = (
                "`resample_method` is deprecated and it will be removed in a future version. "
                "Use `resample_method_input` and `resample_method_output` instead."
            )
            deprecate("resample_method", "1.0.0", deprecation_message, standard_warn=False)
            resample_method_input = kwargs["resample_method"]
            resample_method_output = kwargs["resample_method"]
        if "seed" in kwargs:
            deprecation_message = (
                "`seed` is deprecated and it will be removed in a future version. Use `generator` instead."
            )
            deprecate("seed", "1.0.0", deprecation_message, standard_warn=False)
            generator = torch.Generator(device=device)
            generator.manual_seed(kwargs["seed"])
        if "show_progress_bar" in kwargs:
            deprecation_message = (
                "`show_progress_bar` is deprecated and it will be removed in a future version. "
                "Use `set_progress_bar_config` method instead."
            )
            deprecate("show_progress_bar", "1.0.0", deprecation_message, standard_warn=False)
            self.set_progress_bar_config(disable=not kwargs["show_progress_bar"])
        if "ensemble_kwargs" in kwargs:
            deprecation_message = (
                "`ensemble_kwargs` is deprecated and it will be removed in a future version. "
                "Use `ensembling_kwargs` instead."
            )
            deprecate("ensemble_kwargs", "1.0.0", deprecation_message, standard_warn=False)
            ensembling_kwargs = kwargs["ensemble_kwargs"]

        # basic input checks
        if preset not in (None, "fast", "precise", "paper"):
            raise ValueError("`preset` can take only the following values: None, 'fast', 'precise', and 'paper'.")
        if denoising_steps is not None and denoising_steps < 1:
            raise ValueError("`denoising_steps` must be positive.")
        if ensemble_size is not None:
            if ensemble_size < 1:
                raise ValueError("`ensemble_size` must be positive.")
            if ensemble_size == 2:
                logger.warning(
                    "`ensemble_size` == 2 results are similar to no ensembling (1); "
                    "consider increasing the value to at least 3."
                )
        if processing_resolution is not None:
            if processing_resolution < 0:
                raise ValueError(
                    "`processing_resolution` must be non-negative: 0 for native resolution, "
                    "or any positive value for downsampled processing."
                )
            if processing_resolution % self.latent_size_scale != 0:
                raise ValueError(f"`processing_resolution` must be a multiple of {self.latent_size_scale}.")
        if resample_method_input not in ("nearest", "nearest-exact", "bilinear", "bicubic", "area"):
            raise ValueError(
                "`resample_method_input` takes string values compatible with PIL library: "
                "nearest, nearest-exact, bilinear, bicubic, area."
            )
        if resample_method_output not in ("nearest", "nearest-exact", "bilinear", "bicubic", "area"):
            raise ValueError(
                "`resample_method_output` takes string values compatible with PIL library: "
                "nearest, nearest-exact, bilinear, bicubic, area."
            )
        if batch_size < 0:
            raise ValueError("`batch_size` must be non-negative: 0 for auto-detection or any positive value.")
        if output_prediction_format not in ["pt", "np"]:
            raise ValueError("`output_prediction_format` must be one of `pt` or `np`.")
        if input_latent is not None and generator is not None:
            raise ValueError("`input_latent` and `generator` are cannot be used together.")
        if ensembling_kwargs is not None and not isinstance(ensembling_kwargs, dict):
            raise ValueError("`ensembling_kwargs` must be a dictionary.")
        if output_visualization_kwargs is not None and not isinstance(output_visualization_kwargs, dict):
            raise ValueError("`output_visualization_kwargs` must be a dictionary.")

        # memory saving hints
        if save_memory:
            logger.warning(
                f"`save_memory` is currently not implemented. Consider setting `batch_size` to the minimum (1), "
                f"ensure that `processing_resolution` is set to the recommended default value "
                f"({self.optimal_processing_resolution}). Further memory savings can be achieved with pipeline "
                f"offloading. Refer to https://huggingface.co/docs/diffusers/optimization/memory for more information."
            )

        def preset_override(new_denoising_steps: int, new_ensemble_size: int, new_processing_resolution: int):
            nonlocal denoising_steps, ensemble_size, processing_resolution
            if denoising_steps is not None:
                logger.warning(
                    f"Overriding `denoising_steps`={denoising_steps} due to preset {preset} with "
                    f"value {new_denoising_steps}."
                )
            if ensemble_size is not None:
                logger.warning(
                    f"Overriding `ensemble_size`={ensemble_size} due to preset {preset} with value {new_ensemble_size}."
                )
            if processing_resolution is not None:
                logger.warning(
                    f"Overriding `processing_resolution`={processing_resolution} due to preset {preset} with "
                    f"value {new_processing_resolution}."
                )
            denoising_steps = new_denoising_steps
            ensemble_size = new_ensemble_size
            processing_resolution = new_processing_resolution

        def maybe_override(new_denoising_steps: int, new_ensemble_size: int, new_processing_resolution: int):
            nonlocal denoising_steps, ensemble_size, processing_resolution
            if denoising_steps is None:
                denoising_steps = new_denoising_steps
            if ensemble_size is None:
                ensemble_size = new_ensemble_size
            if processing_resolution is None:
                processing_resolution = new_processing_resolution

        # presets logic
        if preset == "paper" and generator is None:
            raise ValueError('`preset` value `"paper"` requires `generator` to be set to ensure reproducibility.')
        if isinstance(self.scheduler, DDIMScheduler):
            scheduler = "DDIMScheduler"
            if preset == "fast":
                preset_override(10, 1, self.optimal_processing_resolution)
            elif preset == "precise":
                preset_override(10, 10, self.optimal_processing_resolution)
            elif preset == "paper":
                preset_override(50, 10, self.optimal_processing_resolution)
            else:
                # closest to fast
                maybe_override(10, 1, self.optimal_processing_resolution)
            assert denoising_steps is not None and denoising_steps > 0
            if denoising_steps < 10:
                logger.warning(
                    f"Detected `denoising_steps`={denoising_steps} with DDIMScheduler; at least 10 is recommended. "
                    f"Consider using the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            scheduler = "LCMScheduler"
            if preset == "fast":
                preset_override(1, 1, self.optimal_processing_resolution)
            elif preset == "precise":
                preset_override(4, 5, self.optimal_processing_resolution)
            elif preset == "paper":
                preset_override(4, 10, self.optimal_processing_resolution)
            else:
                # closest to fast
                maybe_override(1, 1, self.optimal_processing_resolution)
            assert denoising_steps is not None and denoising_steps > 0
            if not (1 <= denoising_steps <= 4):
                logger.warning(
                    f"Detected `denoising_steps`={denoising_steps} with LCMScheduler; "
                    f"recommended value is between 1 and 4."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}.")
        assert ensemble_size > 0
        assert processing_resolution >= 0  # 0 for native

        # input checks
        input_image_stacked = False
        if isinstance(input_image, np.ndarray) or torch.is_tensor(input_image):
            if input_image.ndim < 2 or input_image.ndim > 4:
                raise ValueError(f"Unsupported number of dimension in the input image: {input_image.ndim}.")
            if input_image.ndim == 4:
                input_image = [input_image[i] for i in range(input_image.shape[0])]
                input_image_stacked = True
            elif input_image.ndim in (2, 3):
                input_image = [input_image]
            else:
                assert False
        if isinstance(input_image, Image.Image):
            input_image = [input_image]
        if not isinstance(input_image, list) and not isinstance(input_image, tuple):
            raise ValueError(f"Unsupported input image type: {type(input_image)}.")
        num_images = len(input_image)

        # latent checks
        input_latent_stacked = False
        if input_latent is not None:
            if torch.is_tensor(input_latent):
                if input_latent.ndim == 5:
                    input_latent = [input_latent[i] for i in range(input_latent.shape[0])]
                    input_latent_stacked = True
                elif input_latent.ndim == 4:
                    input_latent = [input_latent]
                else:
                    raise ValueError(f"Unsupported number of dimension in the input latent: {input_latent.ndim}.")
            if isinstance(input_latent, list) and not isinstance(input_latent, tuple):
                if not all(torch.is_tensor(k) for k in input_latent):
                    raise ValueError("Input latent must be a torch.FloatTensor.")
                if not all(
                    k.dim() == 4 and k.shape[0] == ensemble_size and k.shape[1] == self.latent_space_size
                    for k in input_latent
                ):
                    raise ValueError(
                        f"Input latent must be 4-dimensional with shape [E,{self.latent_space_size},h,w], "
                        f"where E is the requested ensemble_size."
                    )
                if len(input_latent) != num_images:
                    raise ValueError(
                        f"The numbers of input images ({num_images}) and latents ({len(input_latent)}) "
                        f"are not compatible."
                    )
            else:
                raise ValueError(f"Unsupported latent type: {type(input_latent)}.")
        if input_image_stacked ^ input_latent_stacked:
            logger.warning("Different stacking of input images and latents might be a sign of undesired behavior.")

        if generator is not None:
            if isinstance(generator, torch.Generator):
                generator = [generator]
            if isinstance(generator, list) and not isinstance(generator, tuple):
                if len(generator) == 1 and num_images > 1:
                    generator = generator * num_images
                if len(generator) != num_images:
                    raise ValueError(
                        f"The numbers of input images ({num_images}) and generators ({len(generator)}) "
                        f"are not compatible."
                    )
            else:
                raise ValueError(f"Unsupported generator type: {type(generator)}.")

        # Prepare the empty text embedding. In the future, remove text modules completely
        if self.empty_text_embedding is None:
            self.encode_empty_text()

        out = []
        with self.progress_bar(total=num_images * ensemble_size * denoising_steps) as progress_bar:
            for i in range(num_images):
                out.append(
                    self.process_image(
                        input_image[i],
                        input_latent[i] if input_latent is not None else None,
                        generator[i] if generator is not None else None,
                        denoising_steps,
                        ensemble_size,
                        processing_resolution,
                        match_input_resolution,
                        resample_method_input,
                        resample_method_output,
                        batch_size,
                        check_input,
                        ensembling_kwargs,
                        output_prediction_format,
                        output_visualization,
                        output_visualization_kwargs,
                        output_uncertainty,
                        output_latent,
                        progress_bar,
                    )
                )

        if len(out) == 1:
            out = out[0]
        else:
            prediction = [o.prediction for o in out]
            visualization = [o.visualization for o in out] if output_visualization else None
            uncertainty = [o.uncertainty for o in out] if output_uncertainty else None
            latent = [o.latent for o in out] if output_latent else None
            if input_image_stacked:
                if output_prediction_format == "np":
                    prediction = np.stack(prediction)
                    if uncertainty is not None:
                        uncertainty = np.stack(uncertainty)
                elif output_prediction_format == "pt":
                    prediction = torch.stack(prediction)
                    if uncertainty is not None:
                        uncertainty = torch.stack(uncertainty)
                if latent is not None:
                    latent = torch.stack(latent)
            out = MarigoldNormalsOutput(
                prediction=prediction, visualization=visualization, uncertainty=uncertainty, latent=latent
            )

        config = dict(  # noqa: C408
            scheduler=scheduler,
            preset=preset,
            denoising_steps=denoising_steps,
            ensemble_size=ensemble_size,
            processing_resolution=processing_resolution,
            match_input_resolution=match_input_resolution,
            resample_method_input=resample_method_input,
            resample_method_output=resample_method_output,
            batch_size=batch_size,
            save_memory=save_memory,
            check_input=check_input,
            ensembling_kwargs=ensembling_kwargs,
            input_latent=None if input_latent is None else "provided",
            generator=None if generator is None else "provided",
            output_prediction_format=output_prediction_format,
            output_visualization=output_visualization,
            output_visualization_kwargs=output_visualization_kwargs,
            output_uncertainty=output_uncertainty,
            output_latent=output_latent,
        )
        out.config = config
        logger.info(json.dumps(config, indent=2))

        return out

    @torch.no_grad()
    def process_image(
        self,
        input_image: Union[torch.Tensor, np.ndarray, Image.Image],
        input_latent: Optional[torch.FloatTensor],
        generator: Optional[torch.Generator],
        denoising_steps: Optional[int],
        ensemble_size: Optional[int],
        processing_resolution: Optional[int],
        match_input_resolution: bool,
        resample_method_input: str,
        resample_method_output: str,
        batch_size: int,
        check_input: bool,
        ensembling_kwargs: Optional[Dict[str, Any]],
        output_prediction_format: str,
        output_visualization: bool,
        output_visualization_kwargs: Optional[Dict[str, Any]],
        output_uncertainty: bool,
        output_latent: bool,
        progress_bar: tqdm,
    ) -> MarigoldNormalsOutput:
        assert (
            input_latent is None
            or input_latent.dim() == 4
            and input_latent.shape[:2] == (ensemble_size, self.latent_space_size)
        )

        device = self._execution_device
        dtype = self.dtype

        image, input_dtype_max = load_image_canonical(input_image)  # [1,3,H,W]
        image = image.to(device=device, dtype=dtype)

        original_resolution = image.shape[-2:]

        if input_dtype_max is not None:
            image = image * (2.0 / input_dtype_max) - 1.0
        elif check_input:
            check_image_values_range(image)

        if processing_resolution > 0:
            image = resize_to_max_edge(image, processing_resolution, resample_method_input)  # [1,3,PH,PW]

        image, padding = pad_image(image, self.latent_size_scale)  # [1,3,PPH,PPW]

        if batch_size == 0:
            batch_size = find_batch_size(ensemble_size, max(image.shape[-2:]), self.dtype)

        # Model invocation: self.vae.encoder, self.vae.quant_conv
        image_latent = self.encode_image(image)  # [1,4,h,w]

        pred_latent = self.prepare_latent(image_latent, input_latent, generator, ensemble_size)  # [E,4,h,w]

        # Model invocation: self.unet
        pred_latent = self.denoise_prediction_batched(
            image_latent, pred_latent, generator, denoising_steps, ensemble_size, batch_size, progress_bar
        )  # [E,4,h,w]

        # Model invocation: self.vae.decoder, self.vae.post_quant_conv
        prediction = self.decode_prediction_batched(pred_latent, ensemble_size, batch_size)  # [E,3,PPH,PPW]

        prediction = unpad_image(prediction, padding)  # [E,3,PH,PW]

        uncertainty = None
        if ensemble_size > 1:
            prediction, uncertainty = ensemble_normals(
                prediction, output_uncertainty, **(ensembling_kwargs or {})
            )  # [1,3,PH,PW], [1,1,PH,PW]

        if match_input_resolution:
            prediction = resize_maybe_antialias(
                prediction, original_resolution, resample_method_output, is_aa=False
            )  # [1,3,H,W]
            prediction = normalize_normals(prediction)  # [1,3,H,W]
            if uncertainty is not None and output_uncertainty:
                uncertainty = resize_maybe_antialias(
                    uncertainty, original_resolution, resample_method_output, is_aa=False
                )  # [1,1,H,W]

        visualization = None
        if output_visualization:
            visualization = visualize_normals(prediction, **(output_visualization_kwargs or {}))  # PIL.Image

        if output_prediction_format != "pt":
            assert output_prediction_format == "np"
            prediction = prediction.cpu().numpy()
            if uncertainty is not None and output_uncertainty:
                uncertainty = uncertainty.cpu().numpy()

        out = MarigoldNormalsOutput(
            prediction=prediction,
            visualization=visualization if output_visualization else None,
            uncertainty=uncertainty if output_uncertainty else None,
            latent=pred_latent if output_latent else None,
        )

        return out

    @torch.no_grad()
    def prepare_latent(
        self,
        image_latent: torch.FloatTensor,
        input_latent: Optional[torch.FloatTensor],
        generator: Optional[torch.Generator],
        ensemble_size: int,
    ) -> torch.FloatTensor:
        assert image_latent.dim() == 4 and image_latent.shape[:2] == (1, self.latent_space_size)  # [1,4,h,w]
        assert (
            input_latent is None
            or input_latent.dim() == 4
            and input_latent.shape[:2] == (ensemble_size, self.latent_space_size)
        )  # [E,4,h,w]

        if input_latent is not None and input_latent.shape[2:] != image_latent.shape[2:]:
            raise ValueError(
                f"Mismatching size between the passed latent ({input_latent.shape[2:]} and encoded image "
                f"latent ({image_latent.shape[2:]})."
            )

        device = self._execution_device
        dtype = self.dtype

        latent = input_latent  # [E,4,h,w]
        if input_latent is None:
            latent = randn_tensor(
                (ensemble_size, self.latent_space_size, image_latent.shape[2], image_latent.shape[3]),
                generator=generator,
                device=device,
                dtype=dtype,
            )

        return latent

    @torch.no_grad()
    def denoise_prediction_batched(
        self,
        image_latent: torch.FloatTensor,
        input_latent: torch.FloatTensor,
        generator: Optional[torch.Generator],
        denoising_steps: Optional[int],
        ensemble_size: Optional[int],
        batch_size: int,
        progress_bar: tqdm,
    ) -> torch.FloatTensor:
        assert input_latent.dim() == 4 and input_latent.shape[:2] == (ensemble_size, self.latent_space_size)

        out = []

        for i in range(0, ensemble_size, batch_size):
            i_end = min(i + batch_size, ensemble_size)
            latent = input_latent[i:i_end]  # [B,4,h,w]
            latent = self.denoise_prediction(image_latent, latent, denoising_steps, generator, progress_bar)
            out.append(latent)

        out = torch.cat(out, dim=0)

        return out  # [E,4,h,w]

    @torch.no_grad()
    def denoise_prediction(
        self,
        image_latent: torch.FloatTensor,
        pred_latent: torch.FloatTensor,
        denoising_steps: int,
        generator: Optional[torch.Generator],
        progress_bar: tqdm,
    ) -> torch.FloatTensor:
        assert image_latent.dim() == 4 and image_latent.shape[:2] == (1, self.latent_space_size)  # [1,4,h,w]
        assert pred_latent.dim() == 4 and pred_latent.shape[1] == self.latent_space_size  # [B,4,h,w]

        device = self._execution_device
        dtype = self.dtype
        B = pred_latent.shape[0]

        pred_latent = pred_latent.to(device=device, dtype=dtype)
        image_latent = image_latent.to(device=device, dtype=dtype).repeat(B, 1, 1, 1)  # [B,4,h,w]
        text_embedding = self.empty_text_embedding.to(device=device, dtype=dtype).repeat((B, 1, 1))  # [B,2,1024]

        self.scheduler.set_timesteps(denoising_steps, device=device)
        for t in self.scheduler.timesteps:
            latent_cat = torch.cat([image_latent, pred_latent], dim=1)  # [B,8,h,w]
            pred_noise = self.unet(latent_cat, t, encoder_hidden_states=text_embedding).sample  # [B,4,h,w]
            pred_latent = self.scheduler.step(pred_noise, t, pred_latent, generator=generator).prev_sample
            if progress_bar is not None:
                progress_bar.update(B)

        return pred_latent  # [B,4,h,w]

    @torch.no_grad()
    def decode_prediction_batched(
        self,
        pred_latent: torch.FloatTensor,
        ensemble_size: int,
        batch_size: int,
    ) -> torch.FloatTensor:
        assert pred_latent.dim() == 4 and pred_latent.shape[:2] == (ensemble_size, self.latent_space_size)  # [E,4,h,w]

        out = []
        for i in range(0, ensemble_size, batch_size):
            i_end = min(i + batch_size, ensemble_size)
            latent = pred_latent[i:i_end]
            prediction = self.decode_prediction(latent)
            out.append(prediction)

        out = torch.cat(out, dim=0)

        return out  # [E,3,H,W]

    @torch.no_grad()
    def decode_prediction(self, pred_latent: torch.FloatTensor) -> torch.FloatTensor:
        assert pred_latent.dim() == 4 and pred_latent.shape[1] == self.latent_space_size  # [B,4,h,w]

        prediction = self.decode_image(pred_latent)  # [B,3,H,W]

        prediction = torch.clip(prediction, -1.0, 1.0)

        if not self.use_full_z_range:
            prediction[:, 2, :, :] *= 0.5
            prediction[:, 2, :, :] += 0.5

        prediction = normalize_normals(prediction)  # [B,3,H,W]

        return prediction  # [B,3,H,W]

    @torch.no_grad()
    def decode_image(self, pred_latent: torch.FloatTensor) -> torch.FloatTensor:
        assert pred_latent.dim() == 4 and pred_latent.shape[1] == self.latent_space_size  # [B,4,h,w]

        pred_latent = pred_latent / self.latent_scaling_factor
        pred_latent = self.vae.post_quant_conv(pred_latent)
        prediction = self.vae.decoder(pred_latent)

        return prediction  # [B,3,H,W]

    @torch.no_grad()
    def encode_prediction(self, prediction: torch.FloatTensor, check_input: bool = True) -> torch.FloatTensor:
        assert torch.is_tensor(prediction) and torch.is_floating_point(prediction)
        assert prediction.dim() == 4 and prediction.shape[1] == 3  # [B,3,H,W]

        if check_input:
            msg = "ensure the normals vectors are unit length."
            if prediction.isnan().any().item():
                raise ValueError(f"NaN values detected, {msg}")
            if prediction.isfinite().all().item():
                raise ValueError(f"Non-finite values detected, {msg}")
            if ((prediction**2).sum(dim=1) - 1.0).abs().max().item() < 1e-3:
                raise ValueError(f"Non-unit vectors detected, {msg}")

        if not self.use_full_z_range:
            if check_input and (prediction[:, 2, :, :] < 0).any().item() < 1e-3:
                raise ValueError(
                    "Negative Z-component detected, ensure the normals vectors are represented in ray-space"
                )

            prediction = prediction.clone()
            prediction[:, 2, :, :] *= 2.0
            prediction[:, 2, :, :] -= 1.0

        latent = self.encode_image(prediction)

        return latent  # [B,4,h,w]

    @torch.no_grad()
    def encode_image(self, image: torch.FloatTensor) -> torch.FloatTensor:
        assert image.dim() == 4 and image.shape[1] == 3  # [B,3,H,W]

        h = self.vae.encoder(image)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        latent = mean * self.latent_scaling_factor

        return latent  # [B,4,h,w]

    @torch.no_grad()
    def encode_empty_text(self) -> None:
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embedding = self.text_encoder(text_input_ids)[0].to(self.dtype)  # [1,2,1024]
