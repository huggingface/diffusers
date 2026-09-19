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


from .core import Conversion, Rule
from .transforms import Reshape


def rae_conversion(config):
    mapping = {
        "stats.latents_mean": "_latents_mean",
        "stats.latents_std": "_latents_std",
        "decoder.trainable_cls_token": "decoder.trainable_cls_token",
    }
    modules = [("decoder." + name, "decoder." + name) for name in ("decoder_embed", "decoder_norm", "decoder_pred")]
    rules = tuple(
        Rule((f"processor.image_{name}",), (f"encoder_{name}",), Reshape((3,), (1, 3, 1, 1)))
        for name in ("mean", "std")
    )
    for i in range(config["decoder_num_hidden_layers"]):
        prefix = f"decoder.decoder_layers.{i}"
        modules.extend(
            (f"{prefix}.{a}", f"{prefix}.{b}")
            for a, b in (
                ("attention.attention.query", "attention.to_q"),
                ("attention.attention.key", "attention.to_k"),
                ("attention.attention.value", "attention.to_v"),
                ("attention.output.dense", "attention.to_out.0"),
                ("intermediate.dense", "intermediate.dense"),
                ("output.dense", "output.dense"),
                ("layernorm_before", "layernorm_before"),
                ("layernorm_after", "layernorm_after"),
            )
        )
    encoder_type = config["encoder_type"]
    modern = int(config.get("transformers_version", "4").split(".")[0]) >= 5
    if encoder_type in ("dinov2", "mae"):
        tokens = ["cls_token", "position_embeddings"]
        if encoder_type == "dinov2":
            tokens.extend(["mask_token", "register_tokens"])
        mapping.update({f"encoder.embeddings.{name}": f"encoder.embeddings.{name}" for name in tokens})
        name = "encoder.embeddings.patch_embeddings.projection"
        modules.append((name, name))
        for i in range(config["encoder_num_hidden_layers"]):
            if encoder_type == "dinov2":
                prefix = f"encoder.encoder.layer.{i}"
                names = (
                    "norm1",
                    "norm2",
                    "attention.attention.query",
                    "attention.attention.key",
                    "attention.attention.value",
                    "attention.output.dense",
                    "mlp.fc1",
                    "mlp.fc2",
                )
                for j in (1, 2):
                    key = f"{prefix}.layer_scale{j}.lambda1"
                    mapping[key] = key
            elif modern:
                prefix = f"encoder.layers.{i}"
                names = (
                    "attention.q_proj",
                    "attention.k_proj",
                    "attention.v_proj",
                    "attention.o_proj",
                    "layernorm_before",
                    "layernorm_after",
                    "mlp.fc1",
                    "mlp.fc2",
                )
            else:
                prefix = f"encoder.encoder.layer.{i}"
                names = (
                    "attention.attention.query",
                    "attention.attention.key",
                    "attention.attention.value",
                    "attention.output.dense",
                    "layernorm_before",
                    "layernorm_after",
                    "intermediate.dense",
                    "output.dense",
                )
            modules.extend((f"{prefix}.{name}", f"{prefix}.{name}") for name in names)
    elif encoder_type == "siglip2":
        prefix = "encoder." if modern else "encoder.vision_model."
        name = prefix + "embeddings.position_embedding.weight"
        mapping[name] = name
        modules.append((prefix + "embeddings.patch_embedding", prefix + "embeddings.patch_embedding"))
        for i in range(config["encoder_num_hidden_layers"]):
            modules.extend(
                (f"{prefix}encoder.layers.{i}.{name}", f"{prefix}encoder.layers.{i}.{name}")
                for name in (
                    "layer_norm1",
                    "layer_norm2",
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.out_proj",
                    "mlp.fc1",
                    "mlp.fc2",
                )
            )
        mapping.update(
            {
                prefix + "head." + name: prefix + "head." + name
                for name in ("probe", "attention.in_proj_weight", "attention.in_proj_bias")
            }
        )
        modules.extend(
            (prefix + "head." + name, prefix + "head." + name)
            for name in ("attention.out_proj", "layernorm", "mlp.fc1", "mlp.fc2")
        )
    else:
        raise ValueError(f"Unknown RAE encoder type {encoder_type}.")
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=rules)
