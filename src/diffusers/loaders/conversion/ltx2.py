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


def ltx2_conversion(config):
    mapping = {}
    modules = []
    for modality in ("", "audio_"):
        mapping[modality + "scale_shift_table"] = modality + "scale_shift_table"
        modules.extend(
            [(modality + "patchify_proj", modality + "proj_in"), (modality + "proj_out", modality + "proj_out")]
        )
        if config["use_prompt_embeddings"]:
            modules.extend(
                (modality + "caption_projection." + name, modality + "caption_projection." + name)
                for name in ("linear_1", "linear_2")
            )
    embeddings = [
        ("adaln_single", "time_embed"),
        ("audio_adaln_single", "audio_time_embed"),
        ("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift"),
        ("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift"),
        ("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate"),
        ("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate"),
    ]
    modulated = config["cross_attn_mod"] or config["audio_cross_attn_mod"]
    if modulated and config["use_prompt_adaln_single"]:
        embeddings.extend([("prompt_adaln", "prompt_adaln"), ("audio_prompt_adaln", "audio_prompt_adaln")])
    for old, new in embeddings:
        modules.extend(
            (old + "." + name, new + "." + name)
            for name in ("emb.timestep_embedder.linear_1", "emb.timestep_embedder.linear_2", "linear")
        )
    if config["use_keyframes_abs_pos_embedding"]:
        mapping["keyframes_abs_pos_embedding"] = "keyframes_abs_pos_embedding"
    for i in range(config["num_layers"]):
        prefix = f"transformer_blocks.{i}"
        for old, new in (
            ("scale_shift_table", "scale_shift_table"),
            ("audio_scale_shift_table", "audio_scale_shift_table"),
            ("scale_shift_table_a2v_ca_video", "video_a2v_cross_attn_scale_shift_table"),
            ("scale_shift_table_a2v_ca_audio", "audio_a2v_cross_attn_scale_shift_table"),
        ):
            mapping[f"{prefix}.{old}"] = f"{prefix}.{new}"
        if modulated:
            for name in ("prompt_scale_shift_table", "audio_prompt_scale_shift_table"):
                mapping[f"{prefix}.{name}"] = f"{prefix}.{name}"
        for ff, bias in (("ff", config["ff_bias"]), ("audio_ff", config["audio_ff_bias"])):
            for layer in ("net.0.proj", "net.2"):
                for p in ("weight", "bias") if bias else ("weight",):
                    key = f"{prefix}.{ff}.{layer}.{p}"
                    mapping[key] = key
        for attn in ("attn1", "attn2", "audio_attn1", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn"):
            for name in ("to_q", "to_k", "to_v", "to_out.0"):
                bias = config["attention_out_bias"] if name == "to_out.0" else config["attention_bias"]
                for p in ("weight", "bias") if bias else ("weight",):
                    key = f"{prefix}.{attn}.{name}.{p}"
                    mapping[key] = key
            for part in ("q", "k"):
                mapping[f"{prefix}.{attn}.{part}_norm.weight"] = f"{prefix}.{attn}.norm_{part}.weight"
            audio = attn in ("audio_attn1", "audio_attn2", "video_to_audio_attn")
            if config["audio_gated_attn" if audio else "gated_attn"]:
                name = f"{prefix}.{attn}.to_gate_logits"
                modules.append((name, name))
        if config["norm_elementwise_affine"]:
            for norm in (
                "norm1",
                "norm2",
                "norm3",
                "audio_norm1",
                "audio_norm2",
                "audio_norm3",
                "audio_to_video_norm",
                "video_to_audio_norm",
            ):
                key = f"{prefix}.{norm}.weight"
                mapping[key] = key
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping)
