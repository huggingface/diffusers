# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
# Copyright 2024-2025 The HuggingFace Team. All rights reserved.
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
# Marigold project website: https://marigoldcomputervision.github.io
# --------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
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
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .marigold_image_processing import MarigoldImageProcessor


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
Examples:
```py
>>> import diffusers
>>> import torch

>>> pipe = diffusers.MarigoldIntrinsicsPipeline.from_pretrained(
...     "prs-eth/marigold-iid-appearance-v1-1", variant="fp16", torch_dtype=torch.float16
... ).to("cuda")

>>> image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
>>> intrinsics = pipe(image)

>>> vis = pipe.image_processor.visualize_intrinsics(intrinsics.prediction, pipe.target_properties)
>>> vis[0]["albedo"].save("einstein_albedo.png")
>>> vis[0]["roughness"].save("einstein_roughness.png")
>>> vis[0]["metallicity"].save("einstein_metallicity.png")
```
```py
>>> import diffusers
>>> import torch

>>> pipe = diffusers.MarigoldIntrinsicsPipeline.from_pretrained(
...     "prs-eth/marigold-iid-lighting-v1-1", variant="fp16", torch_dtype=torch.float16
... ).to("cuda")

>>> image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
>>> intrinsics = pipe(image)

>>> vis = pipe.image_processor.visualize_intrinsics(intrinsics.prediction, pipe.target_properties)
>>> vis[0]["albedo"].save("einstein_albedo.png")
>>> vis[0]["shading"].save("einstein_shading.png")
>>> vis[0]["residual"].save("einstein_residual.png")
```
"""


@dataclass
class MarigoldIntrinsicsOutput(BaseOutput):
    """
    Output class for Marigold Intrinsic Image Decomposition pipeline.

    Args:
        prediction (`np.ndarray`, `torch.Tensor`):
            Predicted image intrinsics with values in the range [0, 1]. The shape is `(numimages * numtargets) × 3 ×
            height × width` for `torch.Tensor` or `(numimages * numtargets) × height × width × 3` for `np.ndarray`,
            where `numtargets` corresponds to the number of predicted target modalities of the intrinsic image
            decomposition.
        uncertainty (`None`, `np.ndarray`, `torch.Tensor`):
            Uncertainty maps computed from the ensemble, with values in the range [0, 1]. The shape is `(numimages *
            numtargets) × 3 × height × width` for `torch.Tensor` or `(numimages * numtargets) × height × width × 3` for
            `np.ndarray`.
        latent (`None`, `torch.Tensor`):
            Latent features corresponding to the predictions, compatible with the `latents` argument of the pipeline.
            The shape is `(numimages * numensemble) × (numtargets * 4) × latentheight × latentwidth`.
    """

    prediction: Union[np.ndarray, torch.Tensor]
    uncertainty: Union[None, np.ndarray, torch.Tensor]
    latent: Union[None, torch.Tensor]


class MarigoldIntrinsicsPipeline(DiffusionPipeline):
    """
    Pipeline for Intrinsic Image Decomposition (IID) using the Marigold method:
    https://marigoldcomputervision.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the targets latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and predictions to and from latent
            representations.
        scheduler (`DDIMScheduler` or `LCMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        prediction_type (`str`, *optional*):
            Type of predictions made by the model.
        target_properties (`Dict[str, Any]`, *optional*):
            Properties of the predicted modalities, such as `target_names`, a `List[str]` used to define the number,
            order and names of the predicted modalities, and any other metadata that may be required to interpret the
            predictions.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    supported_prediction_types = ("intrinsics",)

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        prediction_type: Optional[str] = None,
        target_properties: Optional[Dict[str, Any]] = None,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()

        if prediction_type not in self.supported_prediction_types:
            logger.warning(
                f"Potentially unsupported `prediction_type='{prediction_type}'`; values supported by the pipeline: "
                f"{self.supported_prediction_types}."
            )

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            prediction_type=prediction_type,
            target_properties=target_properties,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8

        self.target_properties = target_properties
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embedding = None

        self.image_processor = MarigoldImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @property
    def n_targets(self):
        return self.unet.config.out_channels // self.vae.config.latent_channels

    def check_inputs(
        self,
        image: PipelineImageInput,
        num_inference_steps: int,
        ensemble_size: int,
        processing_resolution: int,
        resample_method_input: str,
        resample_method_output: str,
        batch_size: int,
        ensembling_kwargs: Optional[Dict[str, Any]],
        latents: Optional[torch.Tensor],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        output_type: str,
        output_uncertainty: bool,
    ) -> int:
        actual_vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        if actual_vae_scale_factor != self.vae_scale_factor:
            raise ValueError(
                f"`vae_scale_factor` computed at initialization ({self.vae_scale_factor}) differs from the actual one ({actual_vae_scale_factor})."
            )
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
        if ensemble_size == 1 and output_uncertainty:
            raise ValueError(
                "Computing uncertainty by setting `output_uncertainty=True` also requires setting `ensemble_size` "
                "greater than 1."
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
        if processing_resolution % self.vae_scale_factor != 0:
            raise ValueError(f"`processing_resolution` must be a multiple of {self.vae_scale_factor}.")
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
        if output_type not in ["pt", "np"]:
            raise ValueError("`output_type` must be one of `pt` or `np`.")
        if latents is not None and generator is not None:
            raise ValueError("`latents` and `generator` cannot be used together.")
        if ensembling_kwargs is not None:
            if not isinstance(ensembling_kwargs, dict):
                raise ValueError("`ensembling_kwargs` must be a dictionary.")
            if "reduction" in ensembling_kwargs and ensembling_kwargs["reduction"] not in ("median", "mean"):
                raise ValueError("`ensembling_kwargs['reduction']` can be either `'median'` or `'mean'`.")

        # image checks
        num_images = 0
        W, H = None, None
        if not isinstance(image, list):
            image = [image]
        for i, img in enumerate(image):
            if isinstance(img, np.ndarray) or torch.is_tensor(img):
                if img.ndim not in (2, 3, 4):
                    raise ValueError(f"`image[{i}]` has unsupported dimensions or shape: {img.shape}.")
                H_i, W_i = img.shape[-2:]
                N_i = 1
                if img.ndim == 4:
                    N_i = img.shape[0]
            elif isinstance(img, Image.Image):
                W_i, H_i = img.size
                N_i = 1
            else:
                raise ValueError(f"Unsupported `image[{i}]` type: {type(img)}.")
            if W is None:
                W, H = W_i, H_i
            elif (W, H) != (W_i, H_i):
                raise ValueError(
                    f"Input `image[{i}]` has incompatible dimensions {(W_i, H_i)} with the previous images {(W, H)}"
                )
            num_images += N_i

        # latents checks
        if latents is not None:
            if not torch.is_tensor(latents):
                raise ValueError("`latents` must be a torch.Tensor.")
            if latents.dim() != 4:
                raise ValueError(f"`latents` has unsupported dimensions or shape: {latents.shape}.")

            if processing_resolution > 0:
                max_orig = max(H, W)
                new_H = H * processing_resolution // max_orig
                new_W = W * processing_resolution // max_orig
                if new_H == 0 or new_W == 0:
                    raise ValueError(f"Extreme aspect ratio of the input image: [{W} x {H}]")
                W, H = new_W, new_H
            w = (W + self.vae_scale_factor - 1) // self.vae_scale_factor
            h = (H + self.vae_scale_factor - 1) // self.vae_scale_factor
            shape_expected = (num_images * ensemble_size, self.unet.config.out_channels, h, w)

            if latents.shape != shape_expected:
                raise ValueError(f"`latents` has unexpected shape={latents.shape} expected={shape_expected}.")

        # generator checks
        if generator is not None:
            if isinstance(generator, list):
                if len(generator) != num_images * ensemble_size:
                    raise ValueError(
                        "The number of generators must match the total number of ensemble members for all input images."
                    )
                if not all(g.device.type == generator[0].device.type for g in generator):
                    raise ValueError("`generator` device placement is not consistent in the list.")
            elif not isinstance(generator, torch.Generator):
                raise ValueError(f"Unsupported generator type: {type(generator)}.")

        return num_images

    @torch.compiler.disable
    def progress_bar(self, iterable=None, total=None, desc=None, leave=True):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        progress_bar_config = dict(**self._progress_bar_config)
        progress_bar_config["desc"] = progress_bar_config.get("desc", desc)
        progress_bar_config["leave"] = progress_bar_config.get("leave", leave)
        if iterable is not None:
            return tqdm(iterable, **progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput,
        num_inference_steps: Optional[int] = None,
        ensemble_size: int = 1,
        processing_resolution: Optional[int] = None,
        match_input_resolution: bool = True,
        resample_method_input: str = "bilinear",
        resample_method_output: str = "bilinear",
        batch_size: int = 1,
        ensembling_kwargs: Optional[Dict[str, Any]] = None,
        latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "np",
        output_uncertainty: bool = False,
        output_latent: bool = False,
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline.

        Args:
            image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`),
                `List[torch.Tensor]`: An input image or images used as an input for the intrinsic decomposition task.
                For arrays and tensors, the expected value range is between `[0, 1]`. Passing a batch of images is
                possible by providing a four-dimensional array or a tensor. Additionally, a list of images of two- or
                three-dimensional arrays or tensors can be passed. In the latter case, all list elements must have the
                same width and height.
            num_inference_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection.
            ensemble_size (`int`, defaults to `1`):
                Number of ensemble predictions. Higher values result in measurable improvements and visual degradation.
            processing_resolution (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, matches the larger input image dimension. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_resolution (`bool`, *optional*, defaults to `True`):
                When enabled, the output prediction is resized to match the input dimensions. When disabled, the longer
                side of the output will equal to `processing_resolution`.
            resample_method_input (`str`, *optional*, defaults to `"bilinear"`):
                Resampling method used to resize input images to `processing_resolution`. The accepted values are:
                `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.
            resample_method_output (`str`, *optional*, defaults to `"bilinear"`):
                Resampling method used to resize output predictions to match the input resolution. The accepted values
                are `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.
            batch_size (`int`, *optional*, defaults to `1`):
                Batch size; only matters when setting `ensemble_size` or passing a tensor of images.
            ensembling_kwargs (`dict`, *optional*, defaults to `None`)
                Extra dictionary with arguments for precise ensembling control. The following options are available:
                - reduction (`str`, *optional*, defaults to `"median"`): Defines the ensembling function applied in
                  every pixel location, can be either `"median"` or `"mean"`.
            latents (`torch.Tensor`, *optional*, defaults to `None`):
                Latent noise tensors to replace the random initialization. These can be taken from the previous
                function call's output.
            generator (`torch.Generator`, or `List[torch.Generator]`, *optional*, defaults to `None`):
                Random number generator object to ensure reproducibility.
            output_type (`str`, *optional*, defaults to `"np"`):
                Preferred format of the output's `prediction` and the optional `uncertainty` fields. The accepted
                values are: `"np"` (numpy array) or `"pt"` (torch tensor).
            output_uncertainty (`bool`, *optional*, defaults to `False`):
                When enabled, the output's `uncertainty` field contains the predictive uncertainty map, provided that
                the `ensemble_size` argument is set to a value above 2.
            output_latent (`bool`, *optional*, defaults to `False`):
                When enabled, the output's `latent` field contains the latent codes corresponding to the predictions
                within the ensemble. These codes can be saved, modified, and used for subsequent calls with the
                `latents` argument.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.marigold.MarigoldIntrinsicsOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~pipelines.marigold.MarigoldIntrinsicsOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.marigold.MarigoldIntrinsicsOutput`] is returned, otherwise a
                `tuple` is returned where the first element is the prediction, the second element is the uncertainty
                (or `None`), and the third is the latent (or `None`).
        """

        # 0. Resolving variables.
        device = self._execution_device
        dtype = self.dtype

        # Model-specific optimal default values leading to fast and reasonable results.
        if num_inference_steps is None:
            num_inference_steps = self.default_denoising_steps
        if processing_resolution is None:
            processing_resolution = self.default_processing_resolution

        # 1. Check inputs.
        num_images = self.check_inputs(
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
            output_type,
            output_uncertainty,
        )

        # 2. Prepare empty text conditioning.
        # Model invocation: self.tokenizer, self.text_encoder.
        if self.empty_text_embedding is None:
            prompt = ""
            text_inputs = self.tokenizer(
                prompt,
                padding="do_not_pad",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            self.empty_text_embedding = self.text_encoder(text_input_ids)[0]  # [1,2,1024]

        # 3. Preprocess input images. This function loads input image or images of compatible dimensions `(H, W)`,
        # optionally downsamples them to the `processing_resolution` `(PH, PW)`, where
        # `max(PH, PW) == processing_resolution`, and pads the dimensions to `(PPH, PPW)` such that these values are
        # divisible by the latent space downscaling factor (typically 8 in Stable Diffusion). The default value `None`
        # of `processing_resolution` resolves to the optimal value from the model config. It is a recommended mode of
        # operation and leads to the most reasonable results. Using the native image resolution or any other processing
        # resolution can lead to loss of either fine details or global context in the output predictions.
        image, padding, original_resolution = self.image_processor.preprocess(
            image, processing_resolution, resample_method_input, device, dtype
        )  # [N,3,PPH,PPW]

        # 4. Encode input image into latent space. At this step, each of the `N` input images is represented with `E`
        # ensemble members. Each ensemble member is an independent diffused prediction, just initialized independently.
        # Latents of each such predictions across all input images and all ensemble members are represented in the
        # `pred_latent` variable. The variable `image_latent` contains each input image encoded into latent space and
        # replicated `E` times. The variable `pred_latent` contains latents initialization, where the latent space is
        # replicated `T` times relative to the single latent space of `image_latent`, where `T` is the number of the
        # predicted targets. The latents can be either generated (see `generator` to ensure reproducibility), or passed
        # explicitly via the `latents` argument. The latter can be set outside the pipeline code. This behavior can be
        # achieved by setting the `output_latent` argument to `True`. The latent space dimensions are `(h, w)`. Encoding
        # into latent space happens in batches of size `batch_size`.
        # Model invocation: self.vae.encoder.
        image_latent, pred_latent = self.prepare_latents(
            image, latents, generator, ensemble_size, batch_size
        )  # [N*E,4,h,w], [N*E,T*4,h,w]

        del image

        batch_empty_text_embedding = self.empty_text_embedding.to(device=device, dtype=dtype).repeat(
            batch_size, 1, 1
        )  # [B,1024,2]

        # 5. Process the denoising loop. All `N * E` latents are processed sequentially in batches of size `batch_size`.
        # The unet model takes concatenated latent spaces of the input image and the predicted modality as an input, and
        # outputs noise for the predicted modality's latent space. The number of denoising diffusion steps is defined by
        # `num_inference_steps`. It is either set directly, or resolves to the optimal value specific to the loaded
        # model.
        # Model invocation: self.unet.
        pred_latents = []

        for i in self.progress_bar(
            range(0, num_images * ensemble_size, batch_size), leave=True, desc="Marigold predictions..."
        ):
            batch_image_latent = image_latent[i : i + batch_size]  # [B,4,h,w]
            batch_pred_latent = pred_latent[i : i + batch_size]  # [B,T*4,h,w]
            effective_batch_size = batch_image_latent.shape[0]
            text = batch_empty_text_embedding[:effective_batch_size]  # [B,2,1024]

            self.scheduler.set_timesteps(num_inference_steps, device=device)
            for t in self.progress_bar(self.scheduler.timesteps, leave=False, desc="Diffusion steps..."):
                batch_latent = torch.cat([batch_image_latent, batch_pred_latent], dim=1)  # [B,(1+T)*4,h,w]
                noise = self.unet(batch_latent, t, encoder_hidden_states=text, return_dict=False)[0]  # [B,T*4,h,w]
                batch_pred_latent = self.scheduler.step(
                    noise, t, batch_pred_latent, generator=generator
                ).prev_sample  # [B,T*4,h,w]

                if XLA_AVAILABLE:
                    xm.mark_step()

            pred_latents.append(batch_pred_latent)

        pred_latent = torch.cat(pred_latents, dim=0)  # [N*E,T*4,h,w]

        del (
            pred_latents,
            image_latent,
            batch_empty_text_embedding,
            batch_image_latent,
            batch_pred_latent,
            text,
            batch_latent,
            noise,
        )

        # 6. Decode predictions from latent into pixel space. The resulting `N * E` predictions have shape `(PPH, PPW)`,
        # which requires slight postprocessing. Decoding into pixel space happens in batches of size `batch_size`.
        # Model invocation: self.vae.decoder.
        pred_latent_for_decoding = pred_latent.reshape(
            num_images * ensemble_size * self.n_targets, self.vae.config.latent_channels, *pred_latent.shape[2:]
        )  # [N*E*T,4,PPH,PPW]
        prediction = torch.cat(
            [
                self.decode_prediction(pred_latent_for_decoding[i : i + batch_size])
                for i in range(0, pred_latent_for_decoding.shape[0], batch_size)
            ],
            dim=0,
        )  # [N*E*T,3,PPH,PPW]

        del pred_latent_for_decoding
        if not output_latent:
            pred_latent = None

        # 7. Remove padding. The output shape is (PH, PW).
        prediction = self.image_processor.unpad_image(prediction, padding)  # [N*E*T,3,PH,PW]

        # 8. Ensemble and compute uncertainty (when `output_uncertainty` is set). This code treats each of the `N*T`
        # groups of `E` ensemble predictions independently. For each group it computes an ensembled prediction of shape
        # `(PH, PW)` and an optional uncertainty map of the same dimensions. After computing this pair of outputs for
        # each group independently, it stacks them respectively into batches of `N*T` almost final predictions and
        # uncertainty maps.
        uncertainty = None
        if ensemble_size > 1:
            prediction = prediction.reshape(
                num_images, ensemble_size, self.n_targets, *prediction.shape[1:]
            )  # [N,E,T,3,PH,PW]
            prediction = [
                self.ensemble_intrinsics(prediction[i], output_uncertainty, **(ensembling_kwargs or {}))
                for i in range(num_images)
            ]  # [ [[T,3,PH,PW], [T,3,PH,PW]], ... ]
            prediction, uncertainty = zip(*prediction)  # [[T,3,PH,PW], ... ], [[T,3,PH,PW], ... ]
            prediction = torch.cat(prediction, dim=0)  # [N*T,3,PH,PW]
            if output_uncertainty:
                uncertainty = torch.cat(uncertainty, dim=0)  # [N*T,3,PH,PW]
            else:
                uncertainty = None

        # 9. If `match_input_resolution` is set, the output prediction and the uncertainty are upsampled to match the
        # input resolution `(H, W)`. This step may introduce upsampling artifacts, and therefore can be disabled.
        # Depending on the downstream use-case, upsampling can be also chosen based on the tolerated artifacts by
        # setting the `resample_method_output` parameter (e.g., to `"nearest"`).
        if match_input_resolution:
            prediction = self.image_processor.resize_antialias(
                prediction, original_resolution, resample_method_output, is_aa=False
            )  # [N*T,3,H,W]
            if uncertainty is not None and output_uncertainty:
                uncertainty = self.image_processor.resize_antialias(
                    uncertainty, original_resolution, resample_method_output, is_aa=False
                )  # [N*T,1,H,W]

        # 10. Prepare the final outputs.
        if output_type == "np":
            prediction = self.image_processor.pt_to_numpy(prediction)  # [N*T,H,W,3]
            if uncertainty is not None and output_uncertainty:
                uncertainty = self.image_processor.pt_to_numpy(uncertainty)  # [N*T,H,W,3]

        # 11. Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (prediction, uncertainty, pred_latent)

        return MarigoldIntrinsicsOutput(
            prediction=prediction,
            uncertainty=uncertainty,
            latent=pred_latent,
        )

    def prepare_latents(
        self,
        image: torch.Tensor,
        latents: Optional[torch.Tensor],
        generator: Optional[torch.Generator],
        ensemble_size: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def retrieve_latents(encoder_output):
            if hasattr(encoder_output, "latent_dist"):
                return encoder_output.latent_dist.mode()
            elif hasattr(encoder_output, "latents"):
                return encoder_output.latents
            else:
                raise AttributeError("Could not access latents of provided encoder_output")

        image_latent = torch.cat(
            [
                retrieve_latents(self.vae.encode(image[i : i + batch_size]))
                for i in range(0, image.shape[0], batch_size)
            ],
            dim=0,
        )  # [N,4,h,w]
        image_latent = image_latent * self.vae.config.scaling_factor
        image_latent = image_latent.repeat_interleave(ensemble_size, dim=0)  # [N*E,4,h,w]
        N_E, C, H, W = image_latent.shape

        pred_latent = latents
        if pred_latent is None:
            pred_latent = randn_tensor(
                (N_E, self.n_targets * C, H, W),
                generator=generator,
                device=image_latent.device,
                dtype=image_latent.dtype,
            )  # [N*E,T*4,h,w]

        return image_latent, pred_latent

    def decode_prediction(self, pred_latent: torch.Tensor) -> torch.Tensor:
        if pred_latent.dim() != 4 or pred_latent.shape[1] != self.vae.config.latent_channels:
            raise ValueError(
                f"Expecting 4D tensor of shape [B,{self.vae.config.latent_channels},H,W]; got {pred_latent.shape}."
            )

        prediction = self.vae.decode(pred_latent / self.vae.config.scaling_factor, return_dict=False)[0]  # [B,3,H,W]

        prediction = torch.clip(prediction, -1.0, 1.0)  # [B,3,H,W]
        prediction = (prediction + 1.0) / 2.0

        return prediction  # [B,3,H,W]

    @staticmethod
    def ensemble_intrinsics(
        targets: torch.Tensor,
        output_uncertainty: bool = False,
        reduction: str = "median",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Ensembles the intrinsic decomposition represented by the `targets` tensor with expected shape `(B, T, 3, H,
        W)`, where B is the number of ensemble members for a given prediction of size `(H x W)`, and T is the number of
        predicted targets.

        Args:
            targets (`torch.Tensor`):
                Input ensemble of intrinsic image decomposition maps.
            output_uncertainty (`bool`, *optional*, defaults to `False`):
                Whether to output uncertainty map.
            reduction (`str`, *optional*, defaults to `"mean"`):
                Reduction method used to ensemble aligned predictions. The accepted values are: `"median"` and
                `"mean"`.

        Returns:
            A tensor of aligned and ensembled intrinsic decomposition maps with shape `(T, 3, H, W)` and optionally a
            tensor of uncertainties of shape `(T, 3, H, W)`.
        """
        if targets.dim() != 5 or targets.shape[2] != 3:
            raise ValueError(f"Expecting 4D tensor of shape [B,T,3,H,W]; got {targets.shape}.")
        if reduction not in ("median", "mean"):
            raise ValueError(f"Unrecognized reduction method: {reduction}.")

        B, T, _, H, W = targets.shape
        uncertainty = None
        if reduction == "mean":
            prediction = torch.mean(targets, dim=0)  # [T,3,H,W]
            if output_uncertainty:
                uncertainty = torch.std(targets, dim=0)  # [T,3,H,W]
        elif reduction == "median":
            prediction = torch.median(targets, dim=0, keepdim=True).values  # [1,T,3,H,W]
            if output_uncertainty:
                uncertainty = torch.abs(targets - prediction)  # [B,T,3,H,W]
                uncertainty = torch.median(uncertainty, dim=0).values  # [T,3,H,W]
            prediction = prediction.squeeze(0)  # [T,3,H,W]
        else:
            raise ValueError(f"Unrecognized reduction method: {reduction}.")
        return prediction, uncertainty
