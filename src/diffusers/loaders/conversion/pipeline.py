# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Package converted components into the original Stable Diffusion checkpoint container."""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from .io import Checkpoint
from .registry import get_conversion


def export_pipeline_checkpoint(
    model_path, output_path, *, pipeline_format=None, output_format="safetensors", dtype=None
):
    """Export SD/SDXL weights using the component conversions and original component namespaces.

    Tokenizers, schedulers, safety checkers and runtime configuration are separate assets. A single output file must
    hold all component tensors during serialization; use the component CLI when sharded output is desired.
    """
    model_path, output_path = Path(model_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    if output_format not in ("safetensors", "pytorch"):
        raise ValueError("output_format must be safetensors or pytorch")
    if pipeline_format is None:
        pipeline_format = "sdxl" if (model_path / "text_encoder_2").is_dir() else "sd"
    if pipeline_format not in ("sd", "sdxl"):
        raise ValueError("pipeline_format must be sd or sdxl")

    components = [
        ("unet", "UNet2DConditionModel", "model.diffusion_model.", None),
        ("vae", "AutoencoderKL", "first_stage_model.", None),
    ]
    if pipeline_format == "sdxl":
        if (model_path / "text_encoder").is_dir():
            components.append(("text_encoder", "CLIPTextModel", "conditioner.embedders.0.transformer.", "clip"))
            second_index = 1
        else:
            second_index = 0
        components.append(
            (
                "text_encoder_2",
                "CLIPTextModelWithProjection",
                f"conditioner.embedders.{second_index}.model.",
                "openclip",
            )
        )
    else:
        text_config = json.loads((model_path / "text_encoder/config.json").read_text(encoding="utf-8"))
        v2 = text_config.get("hidden_size", 768) == 1024
        components.append(
            (
                "text_encoder",
                "CLIPTextModel",
                "cond_stage_model.model." if v2 else "cond_stage_model.transformer.",
                "openclip" if v2 else "clip",
            )
        )

    tensors = {}
    for directory, cls, prefix, original_format in components:
        path = model_path / directory
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if config.get("quantization_config") is not None:
            raise ValueError("Pipeline export requires unpacked, unquantized tensor weights.")
        if original_format:
            config["original_format"] = original_format
        conversion = get_conversion(cls, config)
        for key, tensor in conversion.iter_converted(Checkpoint(path), reverse=True):
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype)
            tensors[prefix + key] = tensor.detach().cpu().contiguous().clone()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pipeline-export-", dir=output_path.parent) as temporary:
        staging = Path(temporary) / "checkpoint"
        if output_format == "safetensors":
            save_file(tensors, staging, metadata={"format": "pt"})
        else:
            torch.save({"state_dict": tensors}, staging)
        staging.rename(output_path)
    return output_path
