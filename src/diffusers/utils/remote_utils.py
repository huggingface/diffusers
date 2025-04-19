# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

import io
import json
from typing import List, Literal, Optional, Union, cast

import requests

from .deprecation_utils import deprecate
from .import_utils import is_safetensors_available, is_torch_available


if is_torch_available():
    import torch

    from ..image_processor import VaeImageProcessor
    from ..video_processor import VideoProcessor

    if is_safetensors_available():
        import safetensors.torch

    DTYPE_MAP = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "uint8": torch.uint8,
    }


from PIL import Image


def detect_image_type(data: bytes) -> str:
    if data.startswith(b"\xff\xd8"):
        return "jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    elif data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif"
    elif data.startswith(b"BM"):
        return "bmp"
    return "unknown"


def check_inputs_decode(
    endpoint: str,
    tensor: "torch.Tensor",
    processor: Optional[Union["VaeImageProcessor", "VideoProcessor"]] = None,
    do_scaling: bool = True,
    scaling_factor: Optional[float] = None,
    shift_factor: Optional[float] = None,
    output_type: Literal["mp4", "pil", "pt"] = "pil",
    return_type: Literal["mp4", "pil", "pt"] = "pil",
    image_format: Literal["png", "jpg"] = "jpg",
    partial_postprocess: bool = False,
    input_tensor_type: Literal["binary"] = "binary",
    output_tensor_type: Literal["binary"] = "binary",
    height: Optional[int] = None,
    width: Optional[int] = None,
):
    if tensor.ndim == 3 and height is None and width is None:
        raise ValueError("`height` and `width` required for packed latents.")
    if (
        output_type == "pt"
        and return_type == "pil"
        and not partial_postprocess
        and not isinstance(processor, (VaeImageProcessor, VideoProcessor))
    ):
        raise ValueError("`processor` is required.")
    if do_scaling and scaling_factor is None:
        deprecate(
            "do_scaling",
            "1.0.0",
            "`do_scaling` is deprecated, pass `scaling_factor` and `shift_factor` if required.",
            standard_warn=False,
        )


def postprocess_decode(
    response: requests.Response,
    processor: Optional[Union["VaeImageProcessor", "VideoProcessor"]] = None,
    output_type: Literal["mp4", "pil", "pt"] = "pil",
    return_type: Literal["mp4", "pil", "pt"] = "pil",
    partial_postprocess: bool = False,
):
    if output_type == "pt" or (output_type == "pil" and processor is not None):
        output_tensor = response.content
        parameters = response.headers
        shape = json.loads(parameters["shape"])
        dtype = parameters["dtype"]
        torch_dtype = DTYPE_MAP[dtype]
        output_tensor = torch.frombuffer(bytearray(output_tensor), dtype=torch_dtype).reshape(shape)
    if output_type == "pt":
        if partial_postprocess:
            if return_type == "pil":
                output = [Image.fromarray(image.numpy()) for image in output_tensor]
                if len(output) == 1:
                    output = output[0]
            elif return_type == "pt":
                output = output_tensor
        else:
            if processor is None or return_type == "pt":
                output = output_tensor
            else:
                if isinstance(processor, VideoProcessor):
                    output = cast(
                        List[Image.Image],
                        processor.postprocess_video(output_tensor, output_type="pil")[0],
                    )
                else:
                    output = cast(
                        Image.Image,
                        processor.postprocess(output_tensor, output_type="pil")[0],
                    )
    elif output_type == "pil" and return_type == "pil" and processor is None:
        output = Image.open(io.BytesIO(response.content)).convert("RGB")
        detected_format = detect_image_type(response.content)
        output.format = detected_format
    elif output_type == "pil" and processor is not None:
        if return_type == "pil":
            output = [
                Image.fromarray(image)
                for image in (output_tensor.permute(0, 2, 3, 1).float().numpy() * 255).round().astype("uint8")
            ]
        elif return_type == "pt":
            output = output_tensor
    elif output_type == "mp4" and return_type == "mp4":
        output = response.content
    return output


def prepare_decode(
    tensor: "torch.Tensor",
    processor: Optional[Union["VaeImageProcessor", "VideoProcessor"]] = None,
    do_scaling: bool = True,
    scaling_factor: Optional[float] = None,
    shift_factor: Optional[float] = None,
    output_type: Literal["mp4", "pil", "pt"] = "pil",
    image_format: Literal["png", "jpg"] = "jpg",
    partial_postprocess: bool = False,
    height: Optional[int] = None,
    width: Optional[int] = None,
):
    headers = {}
    parameters = {
        "image_format": image_format,
        "output_type": output_type,
        "partial_postprocess": partial_postprocess,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).split(".")[-1],
    }
    if do_scaling and scaling_factor is not None:
        parameters["scaling_factor"] = scaling_factor
    if do_scaling and shift_factor is not None:
        parameters["shift_factor"] = shift_factor
    if do_scaling and scaling_factor is None:
        parameters["do_scaling"] = do_scaling
    elif do_scaling and scaling_factor is None and shift_factor is None:
        parameters["do_scaling"] = do_scaling
    if height is not None and width is not None:
        parameters["height"] = height
        parameters["width"] = width
    headers["Content-Type"] = "tensor/binary"
    headers["Accept"] = "tensor/binary"
    if output_type == "pil" and image_format == "jpg" and processor is None:
        headers["Accept"] = "image/jpeg"
    elif output_type == "pil" and image_format == "png" and processor is None:
        headers["Accept"] = "image/png"
    elif output_type == "mp4":
        headers["Accept"] = "text/plain"
    tensor_data = safetensors.torch._tobytes(tensor, "tensor")
    return {"data": tensor_data, "params": parameters, "headers": headers}


def remote_decode(
    endpoint: str,
    tensor: "torch.Tensor",
    processor: Optional[Union["VaeImageProcessor", "VideoProcessor"]] = None,
    do_scaling: bool = True,
    scaling_factor: Optional[float] = None,
    shift_factor: Optional[float] = None,
    output_type: Literal["mp4", "pil", "pt"] = "pil",
    return_type: Literal["mp4", "pil", "pt"] = "pil",
    image_format: Literal["png", "jpg"] = "jpg",
    partial_postprocess: bool = False,
    input_tensor_type: Literal["binary"] = "binary",
    output_tensor_type: Literal["binary"] = "binary",
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Union[Image.Image, List[Image.Image], bytes, "torch.Tensor"]:
    """
    Hugging Face Hybrid Inference that allow running VAE decode remotely.

    Args:
        endpoint (`str`):
            Endpoint for Remote Decode.
        tensor (`torch.Tensor`):
            Tensor to be decoded.
        processor (`VaeImageProcessor` or `VideoProcessor`, *optional*):
            Used with `return_type="pt"`, and `return_type="pil"` for Video models.
        do_scaling (`bool`, default `True`, *optional*):
            **DEPRECATED**. **pass `scaling_factor`/`shift_factor` instead.** **still set
            do_scaling=None/do_scaling=False for no scaling until option is removed** When `True` scaling e.g. `latents
            / self.vae.config.scaling_factor` is applied remotely. If `False`, input must be passed with scaling
            applied.
        scaling_factor (`float`, *optional*):
            Scaling is applied when passed e.g. [`latents /
            self.vae.config.scaling_factor`](https://github.com/huggingface/diffusers/blob/7007febae5cff000d4df9059d9cf35133e8b2ca9/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L1083C37-L1083C77).
            - SD v1: 0.18215
            - SD XL: 0.13025
            - Flux: 0.3611
            If `None`, input must be passed with scaling applied.
        shift_factor (`float`, *optional*):
            Shift is applied when passed e.g. `latents + self.vae.config.shift_factor`.
            - Flux: 0.1159
            If `None`, input must be passed with scaling applied.
        output_type (`"mp4"` or `"pil"` or `"pt", default `"pil"):
            **Endpoint** output type. Subject to change. Report feedback on preferred type.

            `"mp4": Supported by video models. Endpoint returns `bytes` of video. `"pil"`: Supported by image and video
            models.
                Image models: Endpoint returns `bytes` of an image in `image_format`. Video models: Endpoint returns
                `torch.Tensor` with partial `postprocessing` applied.
                    Requires `processor` as a flag (any `None` value will work).
            `"pt"`: Support by image and video models. Endpoint returns `torch.Tensor`.
                With `partial_postprocess=True` the tensor is postprocessed `uint8` image tensor.

            Recommendations:
                `"pt"` with `partial_postprocess=True` is the smallest transfer for full quality. `"pt"` with
                `partial_postprocess=False` is the most compatible with third party code. `"pil"` with
                `image_format="jpg"` is the smallest transfer overall.

        return_type (`"mp4"` or `"pil"` or `"pt", default `"pil"):
            **Function** return type.

            `"mp4": Function returns `bytes` of video. `"pil"`: Function returns `PIL.Image.Image`.
                With `output_type="pil" no further processing is applied. With `output_type="pt" a `PIL.Image.Image` is
                created.
                    `partial_postprocess=False` `processor` is required. `partial_postprocess=True` `processor` is
                    **not** required.
            `"pt"`: Function returns `torch.Tensor`.
                `processor` is **not** required. `partial_postprocess=False` tensor is `float16` or `bfloat16`, without
                denormalization. `partial_postprocess=True` tensor is `uint8`, denormalized.

        image_format (`"png"` or `"jpg"`, default `jpg`):
            Used with `output_type="pil"`. Endpoint returns `jpg` or `png`.

        partial_postprocess (`bool`, default `False`):
            Used with `output_type="pt"`. `partial_postprocess=False` tensor is `float16` or `bfloat16`, without
            denormalization. `partial_postprocess=True` tensor is `uint8`, denormalized.

        input_tensor_type (`"binary"`, default `"binary"`):
            Tensor transfer type.

        output_tensor_type (`"binary"`, default `"binary"`):
            Tensor transfer type.

        height (`int`, **optional**):
            Required for `"packed"` latents.

        width (`int`, **optional**):
            Required for `"packed"` latents.

    Returns:
        output (`Image.Image` or `List[Image.Image]` or `bytes` or `torch.Tensor`).
    """
    if input_tensor_type == "base64":
        deprecate(
            "input_tensor_type='base64'",
            "1.0.0",
            "input_tensor_type='base64' is deprecated. Using `binary`.",
            standard_warn=False,
        )
        input_tensor_type = "binary"
    if output_tensor_type == "base64":
        deprecate(
            "output_tensor_type='base64'",
            "1.0.0",
            "output_tensor_type='base64' is deprecated. Using `binary`.",
            standard_warn=False,
        )
        output_tensor_type = "binary"
    check_inputs_decode(
        endpoint,
        tensor,
        processor,
        do_scaling,
        scaling_factor,
        shift_factor,
        output_type,
        return_type,
        image_format,
        partial_postprocess,
        input_tensor_type,
        output_tensor_type,
        height,
        width,
    )
    kwargs = prepare_decode(
        tensor=tensor,
        processor=processor,
        do_scaling=do_scaling,
        scaling_factor=scaling_factor,
        shift_factor=shift_factor,
        output_type=output_type,
        image_format=image_format,
        partial_postprocess=partial_postprocess,
        height=height,
        width=width,
    )
    response = requests.post(endpoint, **kwargs)
    if not response.ok:
        raise RuntimeError(response.json())
    output = postprocess_decode(
        response=response,
        processor=processor,
        output_type=output_type,
        return_type=return_type,
        partial_postprocess=partial_postprocess,
    )
    return output


def check_inputs_encode(
    endpoint: str,
    image: Union["torch.Tensor", Image.Image],
    scaling_factor: Optional[float] = None,
    shift_factor: Optional[float] = None,
):
    pass


def postprocess_encode(
    response: requests.Response,
):
    output_tensor = response.content
    parameters = response.headers
    shape = json.loads(parameters["shape"])
    dtype = parameters["dtype"]
    torch_dtype = DTYPE_MAP[dtype]
    output_tensor = torch.frombuffer(bytearray(output_tensor), dtype=torch_dtype).reshape(shape)
    return output_tensor


def prepare_encode(
    image: Union["torch.Tensor", Image.Image],
    scaling_factor: Optional[float] = None,
    shift_factor: Optional[float] = None,
):
    headers = {}
    parameters = {}
    if scaling_factor is not None:
        parameters["scaling_factor"] = scaling_factor
    if shift_factor is not None:
        parameters["shift_factor"] = shift_factor
    if isinstance(image, torch.Tensor):
        data = safetensors.torch._tobytes(image.contiguous(), "tensor")
        parameters["shape"] = list(image.shape)
        parameters["dtype"] = str(image.dtype).split(".")[-1]
    else:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
    return {"data": data, "params": parameters, "headers": headers}


def remote_encode(
    endpoint: str,
    image: Union["torch.Tensor", Image.Image],
    scaling_factor: Optional[float] = None,
    shift_factor: Optional[float] = None,
) -> "torch.Tensor":
    """
    Hugging Face Hybrid Inference that allow running VAE encode remotely.

    Args:
        endpoint (`str`):
            Endpoint for Remote Decode.
        image (`torch.Tensor` or `PIL.Image.Image`):
            Image to be encoded.
        scaling_factor (`float`, *optional*):
            Scaling is applied when passed e.g. [`latents * self.vae.config.scaling_factor`].
            - SD v1: 0.18215
            - SD XL: 0.13025
            - Flux: 0.3611
            If `None`, input must be passed with scaling applied.
        shift_factor (`float`, *optional*):
            Shift is applied when passed e.g. `latents - self.vae.config.shift_factor`.
            - Flux: 0.1159
            If `None`, input must be passed with scaling applied.

    Returns:
        output (`torch.Tensor`).
    """
    check_inputs_encode(
        endpoint,
        image,
        scaling_factor,
        shift_factor,
    )
    kwargs = prepare_encode(
        image=image,
        scaling_factor=scaling_factor,
        shift_factor=shift_factor,
    )
    response = requests.post(endpoint, **kwargs)
    if not response.ok:
        raise RuntimeError(response.json())
    output = postprocess_encode(
        response=response,
    )
    return output
