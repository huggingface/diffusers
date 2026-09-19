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

from .core import Conversion


def lora_conversion(config):
    """Convert adapter layouts using explicit qualified target modules from the adapter configuration.

    The Diffusers side uses PEFT A/B names. Optional alpha tensors remain separate `<module>.alpha` entries; they must
    be passed as network alphas when loading, never multiplied into the weights. This does not merge an adapter into a
    base model. Explicit module names make underscore-delimited Kohya keys unambiguous.
    """
    original_format = config.get("original_format", "kohya")
    if original_format not in ("kohya", "diffusers", "diffusers_old", "peft", "animatediff"):
        raise ValueError("Unknown LoRA original_format.")
    modules = config["modules"]
    if isinstance(modules, str) or not modules or len(set(modules)) != len(modules):
        raise ValueError("LoRA modules must be a nonempty sequence of distinct, fully qualified module names.")
    name = config.get("adapter_name")
    adapter = f".{name}" if name else ""
    mapping = {}
    original_modules = set()
    for module in modules:
        component, _, path = module.partition(".")
        if not path:
            raise ValueError("LoRA module names must include their component, for example unet.conv_in.")
        if original_format == "kohya":
            prefixes = {
                "unet": "lora_unet",
                "text_encoder": "lora_te1",
                "text_encoder_2": "lora_te2",
                "transformer": "lora_transformer",
            }
            prefixes.update(config.get("component_prefixes", {}))
            if component not in prefixes:
                raise ValueError(f"No Kohya component prefix configured for {component}.")
            original_module = prefixes[component] + "_" + path.replace(".", "_")
            suffixes = ("lora_down.weight", "lora_up.weight")
        elif original_format == "peft":
            original_module = module
            suffixes = (f"lora_A{adapter}.weight", f"lora_B{adapter}.weight")
        elif original_format == "diffusers":
            original_module = module
            suffix = "lora" if component in ("unet", "transformer") else "lora_linear_layer"
            suffixes = (f"{suffix}.down.weight", f"{suffix}.up.weight")
        else:
            original_module = path if original_format == "animatediff" else module
            if original_format == "animatediff":
                parts = original_module.split(".")
                index = parts.index("motion_modules") + 2
                parts.insert(index, "temporal_transformer")
                original_module = (
                    ".".join(parts)
                    .replace(".norm1", ".norms.0")
                    .replace(".norm2", ".norms.1")
                    .replace(".norm3", ".ff_norm")
                    .replace(".attn1", ".attention_blocks.0")
                    .replace(".attn2", ".attention_blocks.1")
                )
            replacements = {
                "to_q": "to_q_lora",
                "to_k": "to_k_lora",
                "to_v": "to_v_lora",
                "to_out.0": "to_out_lora",
                "q_proj": "to_q_lora",
                "k_proj": "to_k_lora",
                "v_proj": "to_v_lora",
                "out_proj": "to_out_lora",
            }
            matched = next((leaf for leaf in replacements if original_module.endswith("." + leaf)), None)
            if matched is None:
                raise ValueError(f"Legacy attention LoRA has no layout for module {module}.")
            original_module = original_module[: -len(matched)] + replacements[matched]
            suffixes = ("down.weight", "up.weight")
        if original_module in original_modules:
            raise ValueError(f"Original LoRA module name collision: {original_module}.")
        original_modules.add(original_module)
        for suffix, part in zip(suffixes, ("A", "B")):
            mapping[f"{original_module}.{suffix}"] = f"{module}.lora_{part}{adapter}.weight"
        if config.get("include_alpha", False):
            mapping[f"{original_module}.alpha"] = f"{module}.alpha"
        if config.get("use_dora", False):
            if original_format not in ("kohya", "peft"):
                raise ValueError("DoRA conversion requires kohya or peft format.")
            suffix = "dora_scale" if original_format == "kohya" else f"lora_magnitude_vector{adapter}.weight"
            mapping[f"{original_module}.{suffix}"] = f"{module}.lora_magnitude_vector{adapter}.weight"
    return Conversion(mapping=mapping)
