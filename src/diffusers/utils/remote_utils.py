import base64
import io
import json
from typing import List, Literal, Optional, Union, cast

import requests

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


from PIL import Image

def remote_decode(
    endpoint: str,
    tensor: torch.Tensor,
    processor: Optional[Union[VaeImageProcessor, VideoProcessor]] = None,
    do_scaling: bool = True,
    output_type: Literal["mp4", "pil", "pt"] = "pil",
    image_format: Literal["png", "jpg"] = "jpg",
    partial_postprocess: bool = False,
    input_tensor_type: Literal["base64", "binary"] = "base64",
    output_tensor_type: Literal["base64", "binary"] = "base64",
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Union[Image.Image, List[Image.Image], bytes, torch.Tensor]:
    if tensor.ndim == 3 and height is None and width is None:
        raise ValueError("`height` and `width` required for packed latents.")
    if output_type == "pt" and partial_postprocess is False and processor is None:
        raise ValueError("`processor` is required with `output_type='pt' and `partial_postprocess=False`.")
    headers = {}
    parameters = {
        "do_scaling": do_scaling,
        "output_type": output_type,
        "partial_postprocess": partial_postprocess,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).split(".")[-1],
    }
    if height is not None and width is not None:
        parameters["height"] = height
        parameters["width"] = width
    tensor_data = safetensors.torch._tobytes(tensor, "tensor")
    if input_tensor_type == "base64":
        headers["Content-Type"] = "tensor/base64"
    elif input_tensor_type == "binary":
        headers["Content-Type"] = "tensor/binary"
    if output_type == "pil" and image_format == "jpg" and processor is None:
        headers["Accept"] = "image/jpeg"
    elif output_type == "pil" and image_format == "png" and processor is None:
        headers["Accept"] = "image/png"
    elif (output_tensor_type == "base64" and output_type == "pt") or (
        output_tensor_type == "base64" and output_type == "pil" and processor is not None
    ):
        headers["Accept"] = "tensor/base64"
    elif (output_tensor_type == "binary" and output_type == "pt") or (
        output_tensor_type == "binary" and output_type == "pil" and processor is not None
    ):
        headers["Accept"] = "tensor/binary"
    elif output_type == "mp4":
        headers["Accept"] = "text/plain"
    if input_tensor_type == "base64":
        kwargs = {"json": {"inputs": base64.b64encode(tensor_data).decode("utf-8")}}
    elif input_tensor_type == "binary":
        kwargs = {"data": tensor_data}
    response = requests.post(endpoint, params=parameters, **kwargs, headers=headers)
    if not response.ok:
        raise RuntimeError(response.json())
    if output_type == "pt" or (output_type == "pil" and processor is not None):
        if output_tensor_type == "base64":
            content = response.json()
            output_tensor = base64.b64decode(content["inputs"])
            parameters = content["parameters"]
            shape = parameters["shape"]
            dtype = parameters["dtype"]
        elif output_tensor_type == "binary":
            output_tensor = response.content
            parameters = response.headers
            shape = json.loads(parameters["shape"])
            dtype = parameters["dtype"]
        torch_dtype = DTYPE_MAP[dtype]
        output_tensor = torch.frombuffer(bytearray(output_tensor), dtype=torch_dtype).reshape(shape)
    if output_type == "pt":
        if partial_postprocess:
            output = [Image.fromarray(image.numpy()) for image in output_tensor]
            if len(output) == 1:
                output = output[0]
        else:
            if processor is None:
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
    elif output_type == "pil" and processor is None:
        output = Image.open(io.BytesIO(response.content)).convert("RGB")
    elif output_type == "pil" and processor is not None:
        output = [
            Image.fromarray(image)
            for image in (output_tensor.permute(0, 2, 3, 1).float().numpy() * 255).round().astype("uint8")
        ]
    elif output_type == "mp4":
        output = response.content
    return output
