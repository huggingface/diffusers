# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

import pytest
import torch

from diffusers import MiniMaxH3ImageTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    ContextParallelTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


# The packed layout the dummy inputs describe: a text block, then the rows of one reference image, then the rows of
# the generated image, which is the layout `MiniMaxH3ImageBlocks` builds for an edit.
NUM_TEXT_TOKENS = 4
NUM_REFERENCE_TOKENS = 4
NUM_TARGET_TOKENS = 8
NUM_IMAGE_TOKENS = NUM_REFERENCE_TOKENS + NUM_TARGET_TOKENS


class MiniMaxH3ImageTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return MiniMaxH3ImageTransformer2DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def input_shape(self) -> tuple[int, int]:
        return (NUM_IMAGE_TOKENS, 16)

    @property
    def output_shape(self) -> tuple[int, int]:
        return (NUM_IMAGE_TOKENS, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        # `num_attention_heads * attention_head_dim` (32) is deliberately different from `hidden_size` (24), as it is
        # in the released checkpoint, and `2 * 3 * rope_freq_dim` (12) is smaller than `attention_head_dim` so the
        # partial-rotary path is exercised rather than aliased away.
        return {
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "hidden_size": 24,
            "num_layers": 2,
            "num_refiner_layers": 2,
            "ffn_dim": 32,
            "in_channels": 4,
            "patch_size": (1, 2, 2),
            "text_dim": 8,
            "freq_dim": 8,
            "time_embed_hidden_dim": 24,
            "time_embed_dim": 16,
            "rope_freq_dim": 2,
            "num_pdd_heads": 3,
        }

    def get_packed_layout(self, device: str | torch.device = None) -> dict:
        r"""
        Build the structural arguments of one packed sequence.

        The transformer does not build the layout itself: the caller orders the rows, tags every row with its
        modality and its noise level, and hands over the `(t, h, w)` grid plus the two index tensors. The reference
        rows sit at their own, nearly clean timestep, so the `(timestep, modality)` AdaLN table is addressed on more
        than one row.
        """
        device = torch_device if device is None else device
        sequence_length = NUM_TEXT_TOKENS + NUM_IMAGE_TOKENS
        text_indices = torch.arange(NUM_TEXT_TOKENS, device=device)
        image_indices = torch.arange(NUM_TEXT_TOKENS, sequence_length, device=device)

        # 0 = image, 1 = text.
        token_tags = torch.zeros(sequence_length, dtype=torch.long, device=device)
        token_tags[text_indices] = 1

        timestep_indices = torch.zeros(sequence_length, dtype=torch.long, device=device)
        timestep_indices[NUM_TEXT_TOKENS : NUM_TEXT_TOKENS + NUM_REFERENCE_TOKENS] = 1

        position_ids = torch.zeros(sequence_length, 3, dtype=torch.float32, device=device)
        position_ids[:, 0] = torch.arange(sequence_length, dtype=torch.float32, device=device)
        position_ids[image_indices, 1] = torch.arange(NUM_IMAGE_TOKENS, dtype=torch.float32, device=device) % 4
        position_ids[image_indices, 2] = torch.arange(NUM_IMAGE_TOKENS, dtype=torch.float32, device=device) % 2

        return {
            "timestep": torch.tensor([0.3, 0.999], device=device),
            "timestep_indices": timestep_indices,
            "token_tags": token_tags,
            "position_ids": position_ids,
            "image_indices": image_indices,
            "text_indices": text_indices,
        }

    def get_dummy_inputs(self, batch_size: int = 2, device: str | torch.device = None) -> dict:
        device = torch_device if device is None else device
        generator = self.generator
        init_dict = self.get_init_dict()
        patch_size = init_dict["patch_size"]
        patch_dim = init_dict["in_channels"] * patch_size[0] * patch_size[1] * patch_size[2]

        return {
            "hidden_states": randn_tensor(
                (batch_size, NUM_IMAGE_TOKENS, patch_dim), generator=generator, device=device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, NUM_TEXT_TOKENS, init_dict["text_dim"]), generator=generator, device=device
            ),
            "pdd_step_index": 1,
            **self.get_packed_layout(device=device),
        }


class TestMiniMaxH3ImageTransformer(MiniMaxH3ImageTransformerTesterConfig, ModelTesterMixin):
    """Core model tests for the MiniMax-H3-Image transformer."""

    def test_output_format(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()

        with torch.no_grad():
            output = model(**self.get_dummy_inputs())
            output_tuple = model(**self.get_dummy_inputs(), return_dict=False)

        assert isinstance(output, Transformer2DModelOutput)
        assert output.sample.shape == (2, *self.output_shape)
        torch.testing.assert_close(output.sample, output_tuple[0])

    def test_pdd_step_selects_its_head(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()

        captured = {}
        model.norm_out.register_forward_hook(lambda module, args, output: captured.update(hidden_states=output))
        with torch.no_grad():
            outputs = []
            for step, head in enumerate(model.pdd_proj_out):
                output = model(**{**inputs, "pdd_step_index": step}).sample
                expected = head(captured["hidden_states"]).index_select(1, inputs["image_indices"])
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                outputs.append(output)
        assert not torch.equal(outputs[0], outputs[1])

    def test_pdd_step_out_of_range(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()

        with torch.no_grad(), pytest.raises(IndexError):
            model(**{**self.get_dummy_inputs(), "pdd_step_index": 3})


class TestMiniMaxH3ImageTransformerAdaLNCurve(MiniMaxH3ImageTransformerTesterConfig):
    """The timestep curve that stands in for the time MLP in a pruned MiniMax-H3-Image checkpoint."""

    # A 5-point grid over `[0, 1]`: rows at 0, 0.25, 0.5, 0.75 and 1.
    NUM_GRID_POINTS = 5

    def get_curve_model(self, model):
        """
        Prune `model` exactly.

        The released pruned checkpoints project the activated timestep curve onto an 8-dimensional basis; a basis that
        spans the whole `time_embed_dim` keeps the fold exact, so the curve model has to agree with `model` at every
        grid point.
        """
        curve_model = self.model_class(**self.get_init_dict(), adaln_curve_grid=self.NUM_GRID_POINTS)
        curve_model = curve_model.to(torch_device).eval()
        state_dict = {
            name: value for name, value in model.state_dict().items() if not name.startswith("time_embedder.")
        }
        grid = torch.linspace(0.0, 1.0, self.NUM_GRID_POINTS, device=torch_device)
        with torch.no_grad():
            state_dict["time_embedder.table"] = torch.nn.functional.silu(model.time_embedder(model.time_proj(grid)))
        curve_model.load_state_dict(state_dict)
        return curve_model

    def test_curve_model_matches_the_time_mlp_on_its_grid(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        curve_model = self.get_curve_model(model)
        inputs = {**self.get_dummy_inputs(), "timestep": torch.tensor([0.25, 1.0], device=torch_device)}

        with torch.no_grad():
            torch.testing.assert_close(curve_model(**inputs).sample, model(**inputs).sample)

    def test_curve_interpolates_between_grid_points_and_clamps_outside(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        curve = self.get_curve_model(model).time_embedder
        table = curve.table

        with torch.no_grad():
            midpoint = curve(torch.tensor([0.125], device=torch_device))
            torch.testing.assert_close(midpoint, (0.5 * (table[0] + table[1])).unsqueeze(0))
            ends = curve(torch.tensor([-1.0, 2.0], device=torch_device))
            torch.testing.assert_close(ends, torch.stack([table[0], table[-1]]))


class TestMiniMaxH3ImageTransformerComfySingleFile(MiniMaxH3ImageTransformerTesterConfig):
    """Loading the ComfyUI single files of MiniMax-H3-Image, which are pruned and stack the PDD heads."""

    NUM_GRID_POINTS = 5

    def get_pruned_model(self):
        model = self.model_class(**self.get_init_dict(), adaln_curve_grid=self.NUM_GRID_POINTS)
        with torch.no_grad():
            model.time_embedder.table.normal_(generator=self.generator)
        return model.to(torch_device).eval()

    def get_comfy_state_dict(self, model):
        """
        `model`'s weights in the layout of the Comfy-Org single files: the released key names, a fused `[q; k; v]`
        projection, a `[gate; value]` SwiGLU input, the timestep curve as `adaln_t_table`, and the PDD heads as a bank
        of the first head followed by every later head's offset from it.
        """
        state_dict = dict(model.state_dict())
        num_heads = model.config.num_pdd_heads
        comfy_state_dict = {
            "adaln_t_table": state_dict.pop("time_embedder.table"),
            "rope.inv_freq": model.rope.inv_freq,
            # the attention-backend settings some releases store beside the weights
            "blocks.0.attn.comfy_attention.config": torch.zeros(37, dtype=torch.uint8),
        }
        for suffix in ("weight", "bias"):
            heads = [state_dict.pop(f"pdd_proj_out.{step}.{suffix}") for step in range(num_heads)]
            bank = [heads[0], *[head - heads[0] for head in heads[1:]]]
            comfy_state_dict[f"final_layer.video_out.{suffix}"] = torch.cat(bank)

        renames = {
            "transformer_blocks.": "blocks.",
            "token_refiner.refiner_blocks.": "token_refiner.blocks.",
            "proj_in.": "video_patch_proj.",
            "context_embedder.": "condition_proj.",
            "norm_out.norm.": "final_layer.norm.",
            "norm_out.linear.": "final_layer.adaln_proj.linear.",
            ".attn.norm_q.": ".attn.q_norm.",
            ".attn.norm_k.": ".attn.k_norm.",
            ".attn.to_out.0.": ".attn.out_proj.",
            ".ff.net.2.": ".mlp.fc2.",
        }
        for key, value in state_dict.items():
            if key.endswith((".attn.to_k.weight", ".attn.to_v.weight")):
                continue
            if key.endswith(".attn.to_q.weight"):
                prefix = key[: -len("to_q.weight")]
                value = torch.cat([value, state_dict[prefix + "to_k.weight"], state_dict[prefix + "to_v.weight"]])
                key = prefix + "qkv_proj.weight"
            elif key.endswith(".ff.net.0.proj.weight"):
                value, gate = value.chunk(2, dim=0)
                value = torch.cat([gate, value], dim=0)
                key = key.replace(".ff.net.0.proj.", ".mlp.fc1.")
            for diffusers_name, comfy_name in renames.items():
                key = key.replace(diffusers_name, comfy_name)
            comfy_state_dict[key] = value
        return comfy_state_dict

    def test_single_file_reproduces_the_pruned_model(self, tmp_path):
        # The repository's config describes the released, unpruned model; the curve and the head count come from the
        # checkpoint.
        with torch.device("meta"):
            self.model_class(**self.get_init_dict()).save_config(tmp_path)
        model = self.get_pruned_model()

        loaded = self.model_class.from_single_file(self.get_comfy_state_dict(model), config=str(tmp_path))
        loaded = loaded.to(torch_device).eval()

        assert loaded.config.adaln_curve_grid == self.NUM_GRID_POINTS
        assert loaded.config.num_pdd_heads == model.config.num_pdd_heads
        inputs = self.get_dummy_inputs()
        with torch.no_grad():
            torch.testing.assert_close(loaded(**inputs).sample, model(**inputs).sample)

    def test_single_file_is_told_apart_from_the_video_model(self):
        from diffusers.loaders.single_file_utils import infer_diffusers_model_type

        comfy_state_dict = self.get_comfy_state_dict(self.get_pruned_model())
        assert infer_diffusers_model_type(comfy_state_dict) == "minimax-h3-image"
        comfy_state_dict["audio_patch_proj.weight"] = torch.zeros(24, 32)
        assert infer_diffusers_model_type(comfy_state_dict) == "minimax-h3"

    def test_quantized_single_file_is_refused(self, tmp_path):
        with torch.device("meta"):
            self.model_class(**self.get_init_dict()).save_config(tmp_path)
        comfy_state_dict = self.get_comfy_state_dict(self.get_pruned_model())
        comfy_state_dict["blocks.0.attn.qkv_proj.comfy_quant"] = torch.zeros(19, dtype=torch.uint8)

        with pytest.raises(ValueError, match="quantized ComfyUI checkpoint"):
            self.model_class.from_single_file(comfy_state_dict, config=str(tmp_path))


class TestMiniMaxH3ImageTransformerMemory(MiniMaxH3ImageTransformerTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for the MiniMax-H3-Image transformer."""


class TestMiniMaxH3ImageTransformerTraining(MiniMaxH3ImageTransformerTesterConfig, TrainingTesterMixin):
    """Training tests for the MiniMax-H3-Image transformer."""

    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(
            expected_set={"MiniMaxH3ImageTransformer2DModel", "MiniMaxH3ImageTokenRefiner"}
        )


class TestMiniMaxH3ImageTransformerAttention(MiniMaxH3ImageTransformerTesterConfig, AttentionTesterMixin):
    """Attention processor tests for the MiniMax-H3-Image transformer."""


class TestMiniMaxH3ImageTransformerTorchCompile(MiniMaxH3ImageTransformerTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for the MiniMax-H3-Image transformer."""


class TestMiniMaxH3ImageTransformerContextParallel(MiniMaxH3ImageTransformerTesterConfig, ContextParallelTesterMixin):
    """Context parallel inference tests for the MiniMax-H3-Image transformer."""
