import base64
import io
import json
from typing import List, Literal, Optional, Union, cast

import requests
from PIL import Image

from .import_utils import is_safetensors_available, is_torch_available


if is_torch_available():
    import torch

    from ..image_processor import VaeImageProcessor
    from ..video_processor import VideoProcessor

    if is_safetensors_available():
        import safetensors
    DTYPE_MAP = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "uint8": torch.uint8,
    }


def check_inputs(
    endpoint: str,
    tensor: "torch.Tensor",
    processor: Optional[Union["VaeImageProcessor", "VideoProcessor"]] = None,
    do_scaling: bool = True,
    output_type: Literal["mp4", "pil", "pt"] = "pil",
    return_type: Literal["mp4", "pil", "pt"] = "pil",
    image_format: Literal["png", "jpg"] = "jpg",
    partial_postprocess: bool = False,
    input_tensor_type: Literal["base64", "binary"] = "base64",
    output_tensor_type: Literal["base64", "binary"] = "base64",
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


def _prepare_headers(
    input_tensor_type: Literal["base64", "binary"],
    output_type: Literal["mp4", "pil", "pt"],
    image_format: Literal["png", "jpg"],
    processor: Optional[Union["VaeImageProcessor", "VideoProcessor"]],
    output_tensor_type: Literal["base64", "binary"],
) -> dict:
    headers = {}
    headers["Content-Type"] = "tensor/base64" if input_tensor_type == "base64" else "tensor/binary"

    if output_type == "pil":
        if processor is None:
            headers["Accept"] = "image/jpeg" if image_format == "jpg" else "image/png"
        else:
            headers["Accept"] = "tensor/base64" if output_tensor_type == "base64" else "tensor/binary"
    elif output_type == "pt":
        headers["Accept"] = "tensor/base64" if output_tensor_type == "base64" else "tensor/binary"
    elif output_type == "mp4":
        headers["Accept"] = "text/plain"
    return headers


def _prepare_parameters(
    tensor: "torch.Tensor",
    do_scaling: bool,
    output_type: Literal["mp4", "pil", "pt"],
    partial_postprocess: bool,
    height: Optional[int],
    width: Optional[int],
) -> dict:
    params = {
        "do_scaling": do_scaling,
        "output_type": output_type,
        "partial_postprocess": partial_postprocess,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).split(".")[-1],
    }
    if height is not None and width is not None:
        params["height"] = height
        params["width"] = width
    return params


def _encode_tensor_data(tensor: "torch.Tensor", input_tensor_type: Literal["base64", "binary"]) -> dict:
    tensor_data = safetensors.torch._tobytes(tensor, "tensor")
    if input_tensor_type == "base64":
        return {"json": {"inputs": base64.b64encode(tensor_data).decode("utf-8")}}
    return {"data": tensor_data}


def _decode_tensor_response(response: requests.Response, output_tensor_type: Literal["base64", "binary"]):
    if output_tensor_type == "base64":
        content = response.json()
        tensor_bytes = base64.b64decode(content["inputs"])
        params = content["parameters"]
    else:
        tensor_bytes = response.content
        params = response.headers.copy()
        params["shape"] = json.loads(params["shape"])
    return tensor_bytes, params


def _tensor_to_pil_images(tensor: "torch.Tensor") -> Union[Image.Image, List[Image.Image]]:
    # Assuming tensor is [batch, channels, height, width].
    images = [Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")) for img in tensor]
    return images[0] if len(images) == 1 else images


def remote_decode(
    endpoint: str,
    tensor: "torch.Tensor",
    processor: Optional[Union["VaeImageProcessor", "VideoProcessor"]] = None,
    do_scaling: bool = True,
    output_type: Literal["mp4", "pil", "pt"] = "pil",
    return_type: Literal["mp4", "pil", "pt"] = "pil",
    image_format: Literal["png", "jpg"] = "jpg",
    partial_postprocess: bool = False,
    input_tensor_type: Literal["base64", "binary"] = "base64",
    output_tensor_type: Literal["base64", "binary"] = "base64",
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Union[Image.Image, List[Image.Image], bytes, "torch.Tensor"]:
    """
    Args:
        endpoint (`str`):
            Endpoint for Remote Decode.
        tensor (`torch.Tensor`):
            Tensor to be decoded.
        processor (`VaeImageProcessor` or `VideoProcessor`, *optional*):
            Used with `return_type="pt"`, and `return_type="pil"` for Video models.
        do_scaling (`bool`, default `True`, *optional*):
            When `True` scaling e.g. `latents / self.vae.config.scaling_factor` is applied remotely. If `False`, input
            must be passed with scaling applied.
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

        input_tensor_type (`"base64"` or `"binary"`, default `"base64"`):
            With `"base64"` `tensor` is sent to endpoint base64 encoded. `"binary"` reduces overhead and transfer.

        output_tensor_type (`"base64"` or `"binary"`, default `"base64"`):
            With `"base64"` `tensor` returned by endpoint is base64 encoded. `"binary"` reduces overhead and transfer.

        height (`int`, **optional**):
            Required for `"packed"` latents.

        width (`int`, **optional**):
            Required for `"packed"` latents.
    """

    check_inputs(
        endpoint,
        tensor,
        processor,
        do_scaling,
        output_type,
        return_type,
        image_format,
        partial_postprocess,
        input_tensor_type,
        output_tensor_type,
        height,
        width,
    )

    # Prepare request details.
    headers = _prepare_headers(input_tensor_type, output_type, image_format, processor, output_tensor_type)
    params = _prepare_parameters(tensor, do_scaling, output_type, partial_postprocess, height, width)
    payload = _encode_tensor_data(tensor, input_tensor_type)

    response = requests.post(endpoint, params=params, headers=headers, **payload)
    if not response.ok:
        raise RuntimeError(response.json())

    # Process responses that return a tensor.
    if output_type in ("pt",) or (output_type == "pil" and processor is not None):
        tensor_bytes, tensor_params = _decode_tensor_response(response, output_tensor_type)
        shape = tensor_params["shape"]
        dtype = tensor_params["dtype"]
        torch_dtype = DTYPE_MAP[dtype]
        output_tensor = torch.frombuffer(bytearray(tensor_bytes), dtype=torch_dtype).reshape(shape)

        if output_type == "pt":
            if partial_postprocess:
                if return_type == "pil":
                    return _tensor_to_pil_images(output_tensor)
                return output_tensor
            else:
                if processor is None or return_type == "pt":
                    return output_tensor
                if isinstance(processor, VideoProcessor):
                    return cast(List[Image.Image], processor.postprocess_video(output_tensor, output_type="pil")[0])
                return cast(Image.Image, processor.postprocess(output_tensor, output_type="pil")[0])

    if output_type == "pil" and processor is None and return_type == "pil":
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    if output_type == "pil" and processor is not None:
        tensor_bytes, tensor_params = _decode_tensor_response(response, output_tensor_type)
        shape = tensor_params["shape"]
        dtype = tensor_params["dtype"]
        torch_dtype = DTYPE_MAP[dtype]
        output_tensor = torch.frombuffer(bytearray(tensor_bytes), dtype=torch_dtype).reshape(shape)
        if return_type == "pil":
            return _tensor_to_pil_images(output_tensor)
        return output_tensor

    if output_type == "mp4" and return_type == "mp4":
        return response.content
