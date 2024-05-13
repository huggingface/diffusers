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
from transformers import CLIPTextModel, CLIPTokenizer

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
from ...utils.export_utils import visualize_depth
from ...utils.import_utils import is_scipy_available
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
Examples:
```py
>>> import requests
>>> from diffusers import MarigoldDepthPipeline
>>> from PIL import Image

>>> pipe = MarigoldDepthPipeline.from_pretrained(
...     "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
... )
>>> pipe = pipe.to("cuda")

>>> image = Image.open(requests.get("https://marigoldmonodepth.github.io/images/einstein.jpg", stream=True).raw)
>>> depth = pipe(image, output_visualization=True)

>>> depth.visualization.save("einstein_depth.png")
```
"""


def resize_antialias(image: torch.Tensor, size: Tuple[int, int], mode: str, is_aa: bool = None) -> torch.Tensor:
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

    image = resize_antialias(image, (new_h, new_w), mode, is_aa=True)

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
        image = image.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)  # [1,3,H,W]
    elif image.dim() == 3:
        if image.shape[2] in (1, 3):
            image = image.permute(2, 0, 1)  # [1|3,H,W]
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)  # [3,H,W]
        if image.shape[0] != 3:
            raise ValueError(f"Input image is not 1- or 3-channel: {image.shape}.")
        image = image.unsqueeze(0)  # [1,3,H,W]
    elif image.dim() != 4:
        raise ValueError("Input image is not a 2-, 3-, or 4-dimensional tensor.")

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

    if is_scipy_available():
        import scipy
    else:
        raise ImportError("Make sure to install scipy if you want to use ensembling.")

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

    res = scipy.optimize.minimize(
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


@dataclass
class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        prediction (`PIL.Image.Image`, `np.ndarray`, `torch.FloatTensor`):
            Predicted depth, with values in the range [0, 65535] (`PIL.Image.Image`) or [0, 1] otherwise. For types
            `np.ndarray` or `torch.FloatTensor`, the shape is always $numimages \times 1 \times height \times width$.
        visualization (`None` or List[PIL.Image.Image]):
            Colorized predictions for visualization.
        uncertainty (`None`, `np.ndarray`, `torch.FloatTensor`):
            Uncertainty maps computed from the ensemble. The shape is $numimages \times 1 \times height \times width$.
        latent (`None`, `torch.FloatTensor`):
            Latent features corresponding to the predictions. The shape is $numimages * numensemble \times 4 \times
            latentheight \times latentwidth$.
    """

    prediction: Union[Image.Image, np.ndarray, torch.FloatTensor]
    visualization: Union[None, Image.Image, List[Image.Image]]
    uncertainty: Union[None, np.ndarray, torch.FloatTensor]
    latent: Union[None, torch.FloatTensor]


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

    model_cpu_offload_seq = "text_encoder->vae.encoder->unet->vae.decoder"

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

    def check_inputs(
        self,
        image: Union[Image.Image, np.ndarray, torch.FloatTensor],
        num_inference_steps: int,
        ensemble_size: int,
        processing_resolution: int,
        resample_method_input: str,
        resample_method_output: str,
        batch_size: int,
        ensembling_kwargs: Optional[Dict[str, Any]],
        latents: Optional[torch.FloatTensor],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        output_prediction_format: str,
        output_visualization_kwargs: Optional[Dict[str, Any]],
    ) -> None:
        if num_inference_steps is None:
            raise ValueError("`num_inference_steps` is not specified and could not be resolved from the model config.")
        if num_inference_steps < 1:
            raise ValueError("`num_inference_steps` must be positive.")
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
        if batch_size < 1:
            raise ValueError("`batch_size` must be positive.")
        if output_prediction_format not in ["pt", "np", "pil"]:
            raise ValueError("`output_prediction_format` must be one of `pt`, `np`, or `pil`.")
        if latents is not None and generator is not None:
            raise ValueError("`latents` and `generator` cannot be used together.")
        if ensembling_kwargs is not None and not isinstance(ensembling_kwargs, dict):
            raise ValueError("`ensembling_kwargs` must be a dictionary.")
        if output_visualization_kwargs is not None and not isinstance(output_visualization_kwargs, dict):
            raise ValueError("`output_visualization_kwargs` must be a dictionary.")

        # image checks
        num_images = 1
        if isinstance(image, np.ndarray) or torch.is_tensor(image):
            H, W = image.shape[-2:]
            if image.ndim not in (2, 3, 4):
                raise ValueError(f"`image` has unsupported dimension or shape: {image.shape}.")
            if image.ndim == 4:
                num_images = image.shape[0]
        elif isinstance(image, Image.Image):
            W, H = image.size
        else:
            raise ValueError(f"Unsupported `image` type: {type(image)}.")

        if num_images > 1 and output_prediction_format == "pil":
            raise ValueError("`output_prediction_format='pil'` is not supported when passing multiple input images.")

        # latents checks
        if latents is not None:
            if not torch.is_tensor(latents):
                raise ValueError("`latents` must be a torch.FloatTensor.")
            if not latents.dim() != 4:
                raise ValueError(f"`latents` has unsupported dimensions or shape: {latents.shape}.")

            if processing_resolution > 0:
                max_orig = max(H, W)
                new_H = H * processing_resolution // max_orig
                new_W = W * processing_resolution // max_orig
                if new_H == 0 or new_W == 0:
                    raise ValueError(f"Extreme aspect ratio of the input image: [{W} x {H}]")
                W, H = new_W, new_H
            w = (W + self.latent_size_scale - 1) // self.latent_size_scale
            h = (H + self.latent_size_scale - 1) // self.latent_size_scale
            shape_expected = (num_images * ensemble_size, self.latent_space_size, h, w)

            if latents.shape != shape_expected:
                raise ValueError(f"`latents` has unexpected shape={latents.shape} expected={shape_expected}.")

        # generator checks
        if generator is not None:
            device = self._execution_device
            if isinstance(generator, torch.Generator):
                if generator.device != device:
                    raise ValueError("`generator` device differs from the pipeline's device.")
            elif isinstance(generator, list):
                if len(generator) != num_images * ensemble_size:
                    raise ValueError(
                        "The number generators must match the total number of ensemble members for all input images."
                    )
                if not all(g.device == device for g in generator):
                    raise ValueError("At least one of the `generator` devices differs from the pipeline's device.")
            else:
                raise ValueError(f"Unsupported generator type: {type(generator)}.")

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, torch.FloatTensor],
        num_inference_steps: Optional[int] = None,
        ensemble_size: int = 1,
        processing_resolution: Optional[int] = None,
        match_input_resolution: bool = True,
        resample_method_input: str = "bilinear",
        resample_method_output: str = "bilinear",
        batch_size: int = 1,
        check_input: bool = True,
        ensembling_kwargs: Optional[Dict[str, Any]] = None,
        latents: Optional[Union[torch.FloatTensor, List[torch.FloatTensor]]] = None,
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
            image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`):
                Input image or stacked images.
            num_inference_steps (`int`, *optional*, defaults to `None`):
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
            batch_size (`int`, *optional*, defaults to `1`):
                Batch size; only matters when setting `ensemble_size` or passing a tensor of images.
            check_input (`bool`, defaults to `False`):
                Extra steps to validate compatibility of the inputs with the model.
            ensembling_kwargs (`dict`, *optional*, defaults to `None`)
                Extra dictionary with arguments for precise ensembling control. The following options are available:
                - reduction (`str`, *optional*, defaults to `"median"`): Defines the ensembling function applied in
                  every pixel location, can be either `"median"` or `"mean"`.
                - regularizer_strength (`float`, *optional*, defaults to `0.02`): Strength of the regularizer term that
                  pulls the solution to the unit range.
                - max_iter (`int`, *optional*, defaults to `2`): Number of numerical optimizer function evaluations.
                - tol (`float`, *optional*, defaults to `1e-3`): Numerical tolerance of the optimizer.
                - max_res (`int`, *optional*, defaults to `None`): Resolution at which the search of scale and shift
                  parameters is performed; `None` matches the `processing_resolution`.
            latents (`torch.Tensor`, or `List[torch.Tensor]`, *optional*, defaults to `None`):
                Latent noise tensors to replace the random initialization. These can be taken from the previous
                function call's output.
            generator (`torch.Generator`, or `List[torch.Generator]`, *optional*, defaults to `None`):
                Random number generator object to ensure reproducibility.
            output_prediction_format (`str`, *optional*, defaults to `"np"`):
                Preferred format of the output's `prediction` and the optional `uncertainty` fields. The accepted
                values are: `"pil'` (`PIL.Image.Image`), `"np"` (numpy array) or `"pt"` (torch tensor).
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
                `latents` argument.

        Examples:

        Returns:
            `MarigoldDepthOutput`: Output class instance for Marigold monocular depth prediction pipeline.
        """

        # 0. Resolving variables
        device = self._execution_device
        dtype = self.dtype

        num_images = 1
        if (isinstance(image, np.ndarray) or torch.is_tensor(image)) and image.ndim == 4:
            num_images = image.shape[0]

        if num_inference_steps is None:
            num_inference_steps = self.default_denoising_steps
        if processing_resolution is None:
            processing_resolution = self.default_processing_resolution

        # 1. Checking inputs
        self.check_inputs(
            image,
            num_inference_steps,
            ensemble_size,
            processing_resolution,
            resample_method_input,
            resample_method_output,
            batch_size,
            ensembling_kwargs,
            latents,
            generator,
            output_prediction_format,
            output_visualization_kwargs,
        )

        # 2. Prepare empty text conditioning. Model invocation: self.tokenizer, self.text_encoder
        if self.empty_text_embedding is None:
            self.encode_empty_text()

        # 3. Preprocessing input image
        image, input_dtype_max = load_image_canonical(image)  # [N,3,H,W]

        image = image.to(device=device, dtype=dtype)

        original_resolution = image.shape[-2:]

        if input_dtype_max is not None:
            image = image * (2.0 / input_dtype_max) - 1.0
        elif check_input:
            check_image_values_range(image)

        if processing_resolution > 0:
            image = resize_to_max_edge(image, processing_resolution, resample_method_input)  # [N,3,PH,PW]

        image, padding = pad_image(image, self.latent_size_scale)  # [N,3,PPH,PPW]

        # 4. Encode input image into latent space. Model invocation: self.vae.encoder
        image_latent, pred_latent = self.prepare_latent(
            image, latents, generator, ensemble_size, batch_size
        )  # [N*E,4,h,w], [N*E,4,h,w]

        del image

        batch_empty_text_embedding = self.empty_text_embedding.to(device=device, dtype=dtype).repeat(
            batch_size, 1, 1
        )  # [B,1024,2]

        # 5. Denoising loop. Model invocation: self.unet
        clean_latent = []

        with self.progress_bar(total=num_images * ensemble_size * num_inference_steps) as progress_bar:
            for i in range(0, num_images * ensemble_size, batch_size):
                batch_image_latent = image_latent[i : i + batch_size]  # [B,4,h,w]
                batch_pred_latent = pred_latent[i : i + batch_size]  # [B,4,h,w]
                B = batch_image_latent.shape[0]

                batch_text_embedding = batch_empty_text_embedding[:B]  # [B,2,1024]

                self.scheduler.set_timesteps(num_inference_steps, device=device)

                for t in self.scheduler.timesteps:
                    batch_latent = torch.cat([batch_image_latent, batch_pred_latent], dim=1)  # [B,8,h,w]
                    noise = self.unet(batch_latent, t, encoder_hidden_states=batch_text_embedding).sample  # [B,4,h,w]
                    batch_pred_latent = self.scheduler.step(
                        noise, t, batch_pred_latent, generator=generator
                    ).prev_sample  # [B,4,h,w]
                    progress_bar.update(B)

                clean_latent.append(batch_pred_latent)

                del batch_image_latent, batch_pred_latent, batch_text_embedding, batch_latent, noise

        pred_latent = torch.cat(clean_latent, dim=0)  # [N*E,4,h,w]

        del clean_latent, image_latent, batch_empty_text_embedding

        # 6. Decode prediction from latent into pixel space. Model invocation: self.vae.decoder
        prediction = torch.cat(
            [
                self.decode_prediction(pred_latent[i : i + batch_size])
                for i in range(0, pred_latent.shape[0], batch_size)
            ],
            dim=0,
        )  # [N*E,1,PPH,PPW]

        if not output_latent:
            pred_latent = None

        # 7. Postprocess predictions
        prediction = unpad_image(prediction, padding)  # [N*E,1,PH,PW]

        uncertainty = None
        if ensemble_size > 1:
            prediction = prediction.reshape(num_images, ensemble_size, *prediction.shape[1:])  # [N,E,1,PH,PW]
            prediction = [
                ensemble_depth(prediction[i], output_uncertainty, **(ensembling_kwargs or {}))
                for i in range(num_images)
            ]  # [ [[1,1,PH,PW], [1,1,PH,PW]], ... ]
            prediction, uncertainty = zip(*prediction)  # [[1,1,PH,PW], ... ], [[1,1,PH,PW], ... ]
            prediction = torch.cat(prediction, dim=0)  # [N,1,PH,PW]
            uncertainty = torch.cat(uncertainty, dim=0)  # [N,1,PH,PW]

        if match_input_resolution:
            prediction = resize_antialias(
                prediction, original_resolution, resample_method_output, is_aa=False
            )  # [1,1,H,W]
            if uncertainty is not None and output_uncertainty:
                uncertainty = resize_antialias(
                    uncertainty, original_resolution, resample_method_output, is_aa=False
                )  # [1,1,H,W]

        visualization = None
        if output_visualization:
            visualization = [
                visualize_depth(prediction[i].squeeze(0), **(output_visualization_kwargs or {}))
                for i in range(num_images)
            ]  # [PIL.Image, ...]

        if output_prediction_format != "pt":
            prediction = prediction.cpu().numpy()
            if uncertainty is not None and output_uncertainty:
                uncertainty = uncertainty.cpu().numpy()
            if output_prediction_format == "pil":
                prediction = prediction.squeeze(0).squeeze(0)
                prediction = (prediction * 65535).astype(np.uint16)
                prediction = Image.fromarray(prediction, mode="I;16")

        out = MarigoldDepthOutput(
            prediction=prediction,
            visualization=visualization,
            uncertainty=uncertainty,
            latent=pred_latent,
        )

        return out

    def prepare_latent(
        self,
        image: torch.FloatTensor,
        latents: Optional[torch.FloatTensor],
        generator: Optional[torch.Generator],
        ensemble_size: int,
        batch_size: int,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        image_latent = torch.cat(
            [self.encode_image(image[i : i + batch_size]) for i in range(0, image.shape[0], batch_size)], dim=0
        )  # [N,4,h,w]
        image_latent = image_latent.repeat_interleave(ensemble_size, dim=0)  # [N*E,4,h,w]

        pred_latent = latents
        if pred_latent is None:
            pred_latent = randn_tensor(
                image_latent.shape,
                generator=generator,
                device=image_latent.device,
                dtype=image_latent.dtype,
            )  # [N*E,4,h,w]

        return image_latent, pred_latent

    def decode_prediction(self, pred_latent: torch.FloatTensor) -> torch.FloatTensor:
        assert pred_latent.dim() == 4 and pred_latent.shape[1] == self.latent_space_size  # [B,4,h,w]

        prediction = self.vae.decode(pred_latent / self.vae.config.scaling_factor, return_dict=False)[0]  # [B,3,H,W]

        prediction = prediction.mean(dim=1, keepdim=True)  # [B,1,H,W]
        prediction = torch.clip(prediction, -1.0, 1.0)  # [B,1,H,W]
        prediction = (prediction + 1.0) / 2.0

        return prediction  # [B,1,H,W]

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
