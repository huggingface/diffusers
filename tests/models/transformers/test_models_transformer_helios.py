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

from unittest import mock

import pytest
import torch

from diffusers import HeliosTransformer3DModel
from diffusers.models.transformers.transformer_helios import HeliosOutputNorm, HeliosTransformerBlock
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    LoraTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class HeliosTransformer3DTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return HeliosTransformer3DModel

    @property
    def pretrained_model_name_or_path(self):
        return "hf-internal-testing/tiny-helios-base-transformer"

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (4, 2, 16, 16)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (4, 2, 16, 16)

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int] | tuple | str | bool]:
        return {
            "patch_size": (1, 2, 2),
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "in_channels": 4,
            "out_channels": 4,
            "text_dim": 16,
            "freq_dim": 256,
            "ffn_dim": 32,
            "num_layers": 2,
            "cross_attn_norm": True,
            "qk_norm": "rms_norm_across_heads",
            "rope_dim": (4, 4, 4),
            "has_multi_term_memory_patch": True,
            "guidance_cross_attn": True,
            "zero_history_timestep": True,
            "is_amplify_history": False,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_channels = 4
        num_frames = 2
        height = 16
        width = 16
        text_encoder_embedding_dim = 16
        sequence_length = 12

        hidden_states = randn_tensor(
            (batch_size, num_channels, num_frames, height, width),
            generator=self.generator,
            device=torch_device,
        )
        timestep = torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device)
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, text_encoder_embedding_dim),
            generator=self.generator,
            device=torch_device,
        )
        indices_hidden_states = torch.ones((batch_size, num_frames)).to(torch_device)
        indices_latents_history_short = torch.ones((batch_size, num_frames - 1)).to(torch_device)
        indices_latents_history_mid = torch.ones((batch_size, num_frames - 1)).to(torch_device)
        indices_latents_history_long = torch.ones((batch_size, (num_frames - 1) * 4)).to(torch_device)
        latents_history_short = randn_tensor(
            (batch_size, num_channels, num_frames - 1, height, width),
            generator=self.generator,
            device=torch_device,
        )
        latents_history_mid = randn_tensor(
            (batch_size, num_channels, num_frames - 1, height, width),
            generator=self.generator,
            device=torch_device,
        )
        latents_history_long = randn_tensor(
            (batch_size, num_channels, (num_frames - 1) * 4, height, width),
            generator=self.generator,
            device=torch_device,
        )

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "indices_hidden_states": indices_hidden_states,
            "indices_latents_history_short": indices_latents_history_short,
            "indices_latents_history_mid": indices_latents_history_mid,
            "indices_latents_history_long": indices_latents_history_long,
            "latents_history_short": latents_history_short,
            "latents_history_mid": latents_history_mid,
            "latents_history_long": latents_history_long,
        }


class TestHeliosTransformer3D(HeliosTransformer3DTesterConfig, ModelTesterMixin):
    """Core model tests for Helios Transformer 3D."""


class TestHeliosTransformer3DModulation(HeliosTransformer3DTesterConfig):
    def test_modulation_rows_match_per_token_table(self):
        """Applying the two modulation rows by token range gives the same block outputs and transformer output, bit
        for bit, as modulating with a per-token (batch_size, seq_len, 6, dim) table that has one row per token."""
        inputs = self.get_dummy_inputs()

        def tables(model, seq_len, context_len):
            # reference: the per-token temb and timestep_proj tables, one row per token
            timestep, encoder_hidden_states = inputs["timestep"], inputs["encoder_hidden_states"]
            batch_size, history_len = timestep.shape[0], seq_len - context_len
            if model.zero_history_timestep:
                timestep_t0 = torch.zeros((1), dtype=timestep.dtype, device=timestep.device)
                temb_t0, timestep_proj_t0, _ = model.condition_embedder(
                    timestep_t0, encoder_hidden_states, is_return_encoder_hidden_states=False
                )
                temb_t0 = temb_t0.unsqueeze(1).expand(batch_size, history_len, -1)
                timestep_proj_t0 = (
                    timestep_proj_t0.unflatten(-1, (6, -1)).view(1, 6, 1, -1).expand(batch_size, -1, history_len, -1)
                )
            temb, timestep_proj, _ = model.condition_embedder(timestep, encoder_hidden_states)
            timestep_proj = timestep_proj.unflatten(-1, (6, -1))
            repeat_size = context_len if model.zero_history_timestep else seq_len
            temb = temb.view(batch_size, 1, -1).expand(batch_size, repeat_size, -1)
            timestep_proj = timestep_proj.view(batch_size, 6, 1, -1).expand(batch_size, 6, repeat_size, -1)
            if model.zero_history_timestep:
                temb = torch.cat([temb_t0, temb], dim=1)
                timestep_proj = torch.cat([timestep_proj_t0, timestep_proj], dim=2)
            return temb, timestep_proj.permute(0, 2, 1, 3)

        def block_forward_reference(block, hidden_states, encoder_hidden_states, temb, rotary_emb, context_len):
            # HeliosTransformerBlock.forward with the per-token table; the rows the model passes are not used
            temb = tables(model, hidden_states.shape[1], context_len)[1]
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [
                t.squeeze(2) for t in (block.scale_shift_table.unsqueeze(0) + temb.float()).chunk(6, dim=2)
            ]
            norm_hidden_states = block.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
            attn_output = block.attn1(norm_hidden_states.type_as(hidden_states), None, None, rotary_emb, context_len)
            hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)
            if block.guidance_cross_attn:
                history_hidden_states, hidden_states = torch.split(
                    hidden_states, [hidden_states.shape[1] - context_len, context_len], dim=1
                )
            norm_hidden_states = block.norm2(hidden_states.float()).type_as(hidden_states)
            hidden_states = hidden_states + block.attn2(
                norm_hidden_states, encoder_hidden_states, None, None, context_len
            )
            if block.guidance_cross_attn:
                hidden_states = torch.cat([history_hidden_states, hidden_states], dim=1)
            norm_hidden_states = block.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
            ff_output = block.ffn(norm_hidden_states.type_as(hidden_states))
            return (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        def norm_out_forward_reference(norm_out, hidden_states, temb, context_len):
            temb = tables(model, hidden_states.shape[1], context_len)[0][:, -context_len:, :]
            shift, scale = (norm_out.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            hidden_states = hidden_states[:, -context_len:, :]
            norm_hidden_states = norm_out.norm(hidden_states.float()) * (1 + scale.squeeze(2)) + shift.squeeze(2)
            return norm_hidden_states.type_as(hidden_states)

        def run(model, inputs):
            block_outputs = []
            hooks = [
                block.register_forward_hook(lambda module, args, output: block_outputs.append(output))
                for block in model.blocks
            ]
            sample = model(**inputs).sample
            for hook in hooks:
                hook.remove()
            return sample, block_outputs

        def check(model, inputs, grad=False):
            with torch.set_grad_enabled(grad):
                out, out_blocks = run(model, inputs)
                with (
                    mock.patch.object(HeliosTransformerBlock, "forward", block_forward_reference),
                    mock.patch.object(HeliosOutputNorm, "forward", norm_out_forward_reference),
                ):
                    ref, ref_blocks = run(model, inputs)
            assert torch.isfinite(out).all() and out.abs().sum() > 0
            assert torch.equal(out, ref)
            assert len(out_blocks) == len(ref_blocks) == len(model.blocks)
            for hidden, ref_hidden in zip(out_blocks, ref_blocks):
                assert torch.equal(hidden, ref_hidden)
            return out, ref

        def make_model(dtype, zero_history_timestep):
            torch.manual_seed(0)
            init_dict = {**self.get_init_dict(), "zero_history_timestep": zero_history_timestep}
            model = self.model_class(**init_dict).to(torch_device).eval()
            if dtype == torch.bfloat16:  # bfloat16, with the float32 modules and buffers of from_pretrained
                for name, param in model.named_parameters():
                    if not any(key in name.split(".") for key in model._keep_in_fp32_modules):
                        param.data = param.data.to(dtype)
            return model

        def make_inputs(dtype):
            # batch of 2: different latents and a different timestep per sample, so the samples get different rows
            inputs = {k: torch.cat([v, v]) for k, v in self.get_dummy_inputs().items()}
            for k in ("hidden_states", "encoder_hidden_states", *[k for k in inputs if k.startswith("latents_")]):
                inputs[k] = inputs[k].to(dtype)
            inputs["hidden_states"][1] = -inputs["hidden_states"][1]
            inputs["timestep"] = torch.tensor([10, 900], device=torch_device)
            return inputs

        history_keys = [k for k in inputs if "history" in k]
        for dtype in (torch.float32, torch.bfloat16):
            inputs = make_inputs(dtype)
            for zero_history_timestep in (True, False):
                model = make_model(dtype, zero_history_timestep)
                # with the history latents (history_len > 0 if zero_history_timestep) and without (history_len = 0)
                check(model, inputs)
                check(model, {k: v for k, v in inputs.items() if k not in history_keys})

        # autograd path (float32, history_len > 0): same output; gradients equal up to the summation order
        model = make_model(torch.float32, True).train()
        inputs = make_inputs(torch.float32)
        out, ref = check(model, inputs, grad=True)
        grads = torch.autograd.grad(out.square().sum(), list(model.parameters()), allow_unused=True)
        ref_grads = torch.autograd.grad(ref.square().sum(), list(model.parameters()), allow_unused=True)
        for grad, ref_grad in zip(grads, ref_grads):
            assert (grad is None) == (ref_grad is None)
            if grad is not None:
                torch.testing.assert_close(grad, ref_grad, rtol=1e-4, atol=1e-5)


class TestHeliosTransformer3DMemory(HeliosTransformer3DTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for Helios Transformer 3D."""


class TestHeliosTransformer3DTraining(HeliosTransformer3DTesterConfig, TrainingTesterMixin):
    """Training tests for Helios Transformer 3D."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"HeliosTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestHeliosTransformer3DAttention(HeliosTransformer3DTesterConfig, AttentionTesterMixin):
    """Attention processor tests for Helios Transformer 3D."""


class TestHeliosTransformer3DCompile(HeliosTransformer3DTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for Helios Transformer 3D."""

    @pytest.mark.xfail(
        reason="Helios DiT does not compile when deterministic algorithms are used due to https://github.com/pytorch/pytorch/issues/170079"
    )
    def test_torch_compile_recompilation_and_graph_break(self):
        super().test_torch_compile_recompilation_and_graph_break()


class TestHeliosTransformer3DLoRA(HeliosTransformer3DTesterConfig, LoraTesterMixin):
    pass
