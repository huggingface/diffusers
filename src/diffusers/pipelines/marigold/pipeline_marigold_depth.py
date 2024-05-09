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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import minimize
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
>>> from diffusers import MarigoldDepthPipeline
>>> from PIL import Image

>>> pipe = MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16)
>>> pipe = pipe.to("cuda")

>>> image = Image.open(requests.get("https://marigoldmonodepth.github.io/images/einstein.jpg", stream=True).raw)
>>> depth = pipe(image, output_visualization=True)

>>> depth.visualization.save("einstein_depth.png")
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


def ensemble_depth(
    depth: torch.FloatTensor,
    output_uncertainty: bool,
    reduction: str = "median",
    regularizer_strength: float = 0.02,
    max_iter: int = 2,
    tol: float = 1e-3,
    max_res: int = 1024,
) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
    assert depth.dim() == 4 and depth.shape[1] == 1
    E = depth.shape[0]

    if reduction not in ("mean", "median"):
        raise ValueError("Ensembling `reduction` can be either `'mean'` or `'median'`.")

    depth_to_align = depth.to(torch.float32)
    if max_res is not None and max(depth_to_align.shape[2:]) > max_res:
        depth_to_align = resize_to_max_edge(depth_to_align, max_res, "nearest-exact")

    def align_depth(depth: torch.FloatTensor, s: np.ndarray, t: np.ndarray) -> torch.FloatTensor:
        assert depth.dim() == 4 and isinstance(s, np.ndarray) and isinstance(t, np.ndarray) and len(s) == len(t) == E
        s = torch.from_numpy(s).to(depth).view(E, 1, 1, 1)
        t = torch.from_numpy(t).to(depth).view(E, 1, 1, 1)
        out = depth * s + t
        return out

    def ensemble(
        depth_aligned: torch.FloatTensor, return_uncertainty: bool = False
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        uncertainty = None
        if reduction == "mean":
            prediction = torch.mean(depth_aligned, dim=0, keepdim=True)
            if return_uncertainty:
                uncertainty = torch.std(depth_aligned, dim=0, keepdim=True)
        elif reduction == "median":
            prediction = torch.median(depth_aligned, dim=0, keepdim=True).values
            if return_uncertainty:
                uncertainty = torch.median(torch.abs(depth_aligned - prediction), dim=0, keepdim=True).values
        else:
            assert False
        return prediction, uncertainty

    def cost_fn(st: np.ndarray) -> float:
        cost = 0.0
        s, t = np.split(st, 2)
        depth_aligned = align_depth(depth_to_align, s, t)

        for i, j in torch.combinations(torch.arange(E)):
            diff = depth_aligned[i] - depth_aligned[j]
            cost += (diff**2).mean().sqrt().item()

        if regularizer_strength > 0:
            prediction, _ = ensemble(depth_aligned, return_uncertainty=False)
            err_near = (0.0 - prediction.min()).abs().item()
            err_far = (1.0 - prediction.max()).abs().item()
            cost += (err_near + err_far) * regularizer_strength

        return cost

    init_min = depth_to_align.reshape(E, -1).min(dim=1).values
    init_max = depth_to_align.reshape(E, -1).max(dim=1).values
    init_s = 1.0 / (init_max - init_min).clamp(min=1e-6)
    init_t = -init_s * init_min
    init_st = torch.cat((init_s, init_t)).cpu().numpy()

    res = minimize(
        cost_fn,
        init_st,
        method="BFGS",
        tol=tol,
        options={"maxiter": max_iter, "disp": False},
    )
    s, t = np.split(res.x, 2)

    depth = align_depth(depth, s, t)
    depth, uncertainty = ensemble(depth, return_uncertainty=output_uncertainty)

    depth_min = depth.min()
    depth_max = depth.max()
    depth_range = (depth_max - depth_min).clamp(min=1e-6)
    depth = (depth - depth_min) / depth_range
    if output_uncertainty:
        uncertainty /= depth_range

    return depth, uncertainty  # [1,1,H,W], [1,1,H,W]


def colormap(
    image: Union[np.ndarray, torch.Tensor],
    cmap: str = "Spectral",
    bytes: bool = False,
    force_method: Optional[str] = None,
) -> Union[np.ndarray, torch.Tensor]:
    assert force_method in (None, "matplotlib", "custom")
    if not (torch.is_tensor(image) or isinstance(image, np.ndarray)):
        raise ValueError("Argument must be a numpy array or torch tensor.")

    def method_matplotlib(image, cmap, bytes=False):
        try:
            import matplotlib
        except ImportError:
            return None

        arg_is_pt, device = torch.is_tensor(image), None
        if arg_is_pt:
            image, device = image.cpu().numpy(), image.device

        if cmap not in matplotlib.colormaps:
            raise ValueError(
                f"Unexpected color map {cmap}; available options are: {', '.join(list(matplotlib.colormaps.keys()))}"
            )

        cmap = matplotlib.colormaps[cmap]
        out = cmap(image, bytes=bytes)  # [?,4]
        out = out[..., :3]  # [?,3]

        if arg_is_pt:
            out = torch.tensor(out, device=device)

        return out

    def method_custom(image, cmap, bytes=False):
        arg_is_np = isinstance(image, np.ndarray)
        if arg_is_np:
            image = torch.tensor(image)
        if image.dtype == torch.uint8:
            image = image.float() / 255
        else:
            image = image.float()

        if cmap != "Spectral":
            raise ValueError("Only 'Spectral' color map is available without installing matplotlib.")

        _Spectral_data = (  # Copied from matplotlib/_cm.py
            (0.61960784313725492, 0.003921568627450980, 0.25882352941176473),  # 0.0 -> [0]
            (0.83529411764705885, 0.24313725490196078, 0.30980392156862746),
            (0.95686274509803926, 0.42745098039215684, 0.2627450980392157),
            (0.99215686274509807, 0.68235294117647061, 0.38039215686274508),
            (0.99607843137254903, 0.8784313725490196, 0.54509803921568623),
            (1.0, 1.0, 0.74901960784313726),
            (0.90196078431372551, 0.96078431372549022, 0.59607843137254901),
            (0.6705882352941176, 0.8666666666666667, 0.64313725490196083),
            (0.4, 0.76078431372549016, 0.6470588235294118),
            (0.19607843137254902, 0.53333333333333333, 0.74117647058823533),
            (0.36862745098039218, 0.30980392156862746, 0.63529411764705879),  # 1.0 -> [K-1]
        )

        cmap = torch.tensor(_Spectral_data, dtype=torch.float, device=image.device)  # [K,3]
        K = cmap.shape[0]

        pos = image.clamp(min=0, max=1) * (K - 1)
        left = pos.long()
        right = (left + 1).clamp(max=K - 1)

        d = (pos - left.float()).unsqueeze(-1)
        left_colors = cmap[left]
        right_colors = cmap[right]

        out = (1 - d) * left_colors + d * right_colors

        if bytes:
            out = (out * 255).to(torch.uint8)

        if arg_is_np:
            out = out.numpy()

        return out

    if force_method is None and torch.is_tensor(image) and cmap == "Spectral":
        return method_custom(image, cmap, bytes)

    out = None
    if force_method != "custom":
        out = method_matplotlib(image, cmap, bytes)

    if force_method == "matplotlib" and out is None:
        raise ImportError("matplotlib is not installed")

    if out is None:
        out = method_custom(image, cmap, bytes)

    return out


def visualize_depth(
    depth: torch.FloatTensor,
    color_map: str = "Spectral",
    vis_min: float = 0.0,
    vis_max: float = 1.0,
):
    assert depth.dim() == 4 and depth.shape[:2] == (1, 1)

    if vis_max <= vis_min:
        raise ValueError(f"Invalid depth range for colorization: [{vis_min}, {vis_max}]")

    if vis_min != 0.0 or vis_max != 1.0:
        depth = (depth - vis_min) / (vis_max - vis_min)
    depth = depth.squeeze(1).squeeze(0)  # [H,W]

    visualization = colormap(depth, cmap=color_map, bytes=True)  # [H,W,3]
    visualization = Image.fromarray(visualization.cpu().numpy())

    return visualization


@dataclass
class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        prediction (`PIL.Image.Image`, `np.ndarray`, `torch.FloatTensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`,
                or `List[torch.FloatTensor]`):
            Predicted depth, with values in the range of [0, 1].
        visualization (`None`, `PIL.Image.Image`, or List[PIL.Image.Image]):
            Colorized prediction for visualization.
        uncertainty (`None`, `np.ndarray`, torch.FloatTensor, or a `List` of them):
            Uncertainty map computed from the ensemble.
        latent (`None`, `torch.FloatTensor`, or `List[torch.FloatTensor]`):
            Latent features corresponding to the ensemble predictions.
    """

    prediction: Union[
        Image.Image,
        np.ndarray,
        torch.FloatTensor,
        List[Image.Image],
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


class MarigoldDepthPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using the Marigold method: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and predictions to and from latent
            representations.
        scheduler (`DDIMScheduler` or `LCMScheduler`):
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
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.latent_size_scale = 8
        self.latent_space_size = self.vae.config.latent_channels
        self.latent_scaling_factor = self.vae.config.scaling_factor
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embedding = None

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        input_image: PipelineImageInput,
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 1,
        processing_resolution: Optional[int] = None,
        match_input_resolution: bool = True,
        resample_method_input: str = "bilinear",
        resample_method_output: str = "bilinear",
        batch_size: int = 0,
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
    ) -> MarigoldDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`,
                    or `List[torch.Tensor]`):
                Input image or images.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
            ensemble_size (`int`, defaults to `1`):
                Number of ensemble predictions. Recommended values are 5 and higher for better precision, or 1 for
                faster inference.
            processing_resolution (`int`, *optional*, defaults to None):
                Effective processing resolution. When set to `0`, matches the larger input image dimension. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
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
                Inference batch size. Smaller values save memory. The default value `0` sets it to the value of
                `ensemble_size`.
            check_input (`bool`, defaults to `False`):
                Extra steps to validate compatibility of the inputs with the model.
            ensembling_kwargs (`dict`, *optional*, defaults to `None`)
                Extra dictionary with arguments for precise ensembling control. The following options are available: -
                - reduction (`str`, *optional*, defaults to `"median"`): Defines the ensembling function applied in
                  every pixel location, can be either "median" or "mean".
                - regularizer_strength (`float`, *optional*, defaults to `0.02`): Strength of the regularizer term that
                  pulls the solution to the unit range.
                - max_iter (`int`, *optional*, defaults to `2`): Number of numerical optimizer function evaluations.
                - tol (`float`, *optional*, defaults to `1e-3`): Numerical tolerance of the optimizer.
                - max_res (`int`, *optional*, defaults to `None`): Resolution at which the search of scale and shift
                  parameters is performed; `None` matches the `processing_resolution`.
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
                Extra dictionary with arguments for precise visualization control. The following options are available:
                - color_map (`str`, *optional*, defaults to `"Spectral"`): Color map used to convert a single-channel
                  depth prediction into colored representation.
                - vis_min (`float`, *optional*, defaults to `0.0`): Minimum value of the visualized depth range.
                - vis_max (`float`, *optional*, defaults to `1.0`): Maximum value of the visualized depth range.
            output_uncertainty (`bool`, *optional*, defaults to `True`):
                When enabled, the output's `uncertainty` field contains the predictive uncertainty map, provided that
                the `ensemble_size` argument is set to a value above 2.
            output_latent (`bool`, *optional*, defaults to `False`):
                When enabled, the output's `latent` field contains the latent codes corresponding to the predictions
                within the ensemble. These codes can be saved, modified, and used for subsequent calls with the
                `input_latent` argument.

        Examples:

        Returns:
            `MarigoldDepthOutput`: Output class instance for Marigold monocular depth prediction pipeline.
        """

        device = self._execution_device

        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_resolution is None:
            processing_resolution = self.default_processing_resolution
        if batch_size == 0:
            batch_size = ensemble_size

        # basic input checks
        if denoising_steps is None:
            raise ValueError("`denoising_steps` is not specified and could not be resolved from the model config.")
        if denoising_steps < 1:
            raise ValueError("`denoising_steps` must be positive.")
        if ensemble_size < 1:
            raise ValueError("`ensemble_size` must be positive.")
        if ensemble_size == 2:
            logger.warning(
                "`ensemble_size` == 2 results are similar to no ensembling (1); "
                "consider increasing the value to at least 3."
            )
        if processing_resolution is None:
            raise ValueError(
                "`processing_resolution` is not specified and could not be resolved from the model config."
            )
        if processing_resolution < 0:
            raise ValueError(
                "`processing_resolution` must be non-negative: 0 for native resolution, or any positive value for "
                "downsampled processing."
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
            raise ValueError("`batch_size` must be non-negative: 0 sets it equal to `ensemble_size`.")
        if output_prediction_format not in ["pt", "np", "pil"]:
            raise ValueError("`output_prediction_format` must be one of `pt`, `np`, or `pil`.")
        if input_latent is not None and generator is not None:
            raise ValueError("`input_latent` and `generator` are cannot be used together.")
        if ensembling_kwargs is not None and not isinstance(ensembling_kwargs, dict):
            raise ValueError("`ensembling_kwargs` must be a dictionary.")
        if output_visualization_kwargs is not None and not isinstance(output_visualization_kwargs, dict):
            raise ValueError("`output_visualization_kwargs` must be a dictionary.")

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
            out = MarigoldDepthOutput(
                prediction=prediction, visualization=visualization, uncertainty=uncertainty, latent=latent
            )

        return out

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
    ) -> MarigoldDepthOutput:
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
            prediction, uncertainty = ensemble_depth(
                prediction, output_uncertainty, **(ensembling_kwargs or {})
            )  # [1,1,PH,PW], [1,1,PH,PW]

        if match_input_resolution:
            prediction = resize_maybe_antialias(
                prediction, original_resolution, resample_method_output, is_aa=False
            )  # [1,1,H,W]
            if uncertainty is not None and output_uncertainty:
                uncertainty = resize_maybe_antialias(
                    uncertainty, original_resolution, resample_method_output, is_aa=False
                )  # [1,1,H,W]

        visualization = None
        if output_visualization:
            visualization = visualize_depth(prediction, **(output_visualization_kwargs or {}))  # PIL.Image

        if output_prediction_format != "pt":
            assert output_prediction_format == "np"
            prediction = prediction.cpu().numpy()
            if uncertainty is not None and output_uncertainty:
                uncertainty = uncertainty.cpu().numpy()

        out = MarigoldDepthOutput(
            prediction=prediction,
            visualization=visualization if output_visualization else None,
            uncertainty=uncertainty if output_uncertainty else None,
            latent=pred_latent if output_latent else None,
        )

        return out

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

        return out  # [E,1,H,W]

    def decode_prediction(self, pred_latent: torch.FloatTensor) -> torch.FloatTensor:
        assert pred_latent.dim() == 4 and pred_latent.shape[1] == self.latent_space_size  # [B,4,h,w]

        prediction = self.decode_image(pred_latent)  # [B,3,H,W]

        prediction = prediction.mean(dim=1, keepdim=True)  # [B,1,H,W]
        prediction = torch.clip(prediction, -1.0, 1.0)  # [B,1,H,W]
        prediction = (prediction + 1.0) / 2.0

        return prediction  # [B,1,H,W]

    def decode_image(self, pred_latent: torch.FloatTensor) -> torch.FloatTensor:
        assert pred_latent.dim() == 4 and pred_latent.shape[1] == self.latent_space_size  # [B,4,h,w]

        pred_latent = pred_latent / self.latent_scaling_factor
        pred_latent = self.vae.post_quant_conv(pred_latent)
        prediction = self.vae.decoder(pred_latent)

        return prediction  # [B,3,H,W]

    def encode_prediction(self, prediction: torch.FloatTensor, check_input: bool = True) -> torch.FloatTensor:
        assert torch.is_tensor(prediction) and torch.is_floating_point(prediction)
        assert prediction.dim() == 4 and prediction.shape[1] == 1  # [B,1,H,W]

        if check_input:
            msg = "ensure the depth values are within [0,1] interval, corresponding to the near and far planes."
            if prediction.isnan().any().item():
                raise ValueError(f"NaN values detected, {msg}")
            if prediction.isfinite().all().item():
                raise ValueError(f"Non-finite values detected, {msg}")
            if (prediction > 1.0).any().item() or (prediction < 0.0).any().item():
                raise ValueError(f"Values outside of the expected range detected, {msg}")

        prediction = prediction * 2 - 1  # [B,1,H,W]
        prediction = prediction.repeat(1, 3, 1, 1)  # [B,3,H,W]
        latent = self.encode_image(prediction)

        return latent  # [B,4,h,w]

    def encode_image(self, image: torch.FloatTensor) -> torch.FloatTensor:
        assert image.dim() == 4 and image.shape[1] == 3  # [B,3,H,W]

        h = self.vae.encoder(image)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        latent = mean * self.latent_scaling_factor

        return latent  # [B,4,h,w]

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
