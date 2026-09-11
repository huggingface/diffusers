# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import argparse
from pathlib import Path

import torch
from packaging import version
from torch.onnx import export

from diffusers import AutoencoderKL


is_torch_less_than_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")


def onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset,
    use_external_data_format=False,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # PyTorch deprecated the `enable_onnx_checker` and `use_external_data_format` arguments in v1.11,
    # so we check the torch version for backwards compatibility
    if is_torch_less_than_1_11:
        export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            use_external_data_format=use_external_data_format,
            enable_onnx_checker=True,
            opset_version=opset,
        )
    else:
        export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=opset,
        )


@torch.no_grad()
def convert_models(model_path: str, output_path: str, opset: int, fp16: bool = False):
    dtype = torch.float16 if fp16 else torch.float32
    if fp16 and torch.cuda.is_available():
        device = "cuda"
    elif fp16 and not torch.cuda.is_available():
        raise ValueError("`float16` model export is only supported on GPUs with CUDA")
    else:
        device = "cpu"
    output_path = Path(output_path)

    # VAE DECODER
    vae_decoder = AutoencoderKL.from_pretrained(model_path + "/vae")
    vae_latent_channels = vae_decoder.config.latent_channels
    # forward only through the decoder part
    vae_decoder.forward = vae_decoder.decode
    onnx_export(
        vae_decoder,
        model_args=(
            torch.randn(1, vae_latent_channels, 25, 25).to(device=device, dtype=dtype),
            False,
        ),
        output_path=output_path / "vae_decoder" / "model.onnx",
        ordered_input_names=["latent_sample", "return_dict"],
        output_names=["sample"],
        dynamic_axes={
            "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        },
        opset=opset,
    )
    del vae_decoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
    )

    parser.add_argument("--output_path", type=str, required=True, help="Path to the output model.")
    parser.add_argument(
        "--opset",
        default=14,
        type=int,
        help="The version of the ONNX operator set to use.",
    )
    parser.add_argument("--fp16", action="store_true", default=False, help="Export the models in `float16` mode")

    args = parser.parse_args()
    print(args.output_path)
    convert_models(args.model_path, args.output_path, args.opset, args.fp16)
    print("SD: Done: ONNX")
