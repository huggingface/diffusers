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

import os
import subprocess
import sys

import pytest
import torch

from diffusers import Flux2Transformer2DModel
from diffusers.loaders.lora_pipeline import Flux2LoraLoaderMixin
from diffusers.models.transformers.transformer_flux2 import (
    Flux2KVAttnProcessor,
    Flux2KVCache,
    Flux2KVLayerCache,
    Flux2KVParallelSelfAttnProcessor,
)
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, is_tensor_parallel, require_torch_neuron, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    BitsAndBytesTesterMixin,
    ContextParallelTesterMixin,
    GGUFCompileTesterMixin,
    GGUFTesterMixin,
    LoKrTesterMixin,
    LoraHotSwappingForModelTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    SingleFileTesterMixin,
    TensorParallelTesterMixin,
    TensorParallelTPUTesterMixin,
    TorchAoCompileTesterMixin,
    TorchAoTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)
from ..testing_utils.lokr import check_lokr_deltas, make_lokr_factors


enable_full_determinism()


class Flux2TransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return Flux2Transformer2DModel

    @property
    def output_shape(self) -> tuple[int, int]:
        return (16, 4)

    @property
    def input_shape(self) -> tuple[int, int]:
        return (16, 4)

    @property
    def model_split_percents(self) -> list:
        # We override the items here because the transformer under consideration is small.
        return [0.7, 0.6, 0.6]

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def uses_custom_attn_processor(self) -> bool:
        # Skip setting testing with default: AttnProcessor
        return True

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int]]:
        return {
            "patch_size": 1,
            "in_channels": 4,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 16,
            "num_attention_heads": 2,
            "joint_attention_dim": 32,
            "timestep_guidance_channels": 256,  # Hardcoded in original code
            "axes_dims_rope": [4, 4, 4, 4],
        }

    def get_dummy_inputs(
        self, height: int = 4, width: int = 4, batch_size: int = 1, device: str = torch_device
    ) -> dict[str, torch.Tensor]:
        num_latent_channels = 4
        sequence_length = 48
        embedding_dim = 32

        hidden_states = randn_tensor(
            (batch_size, height * width, num_latent_channels), generator=self.generator, device=device
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, embedding_dim), generator=self.generator, device=device
        )

        t_coords = torch.arange(1)
        h_coords = torch.arange(height)
        w_coords = torch.arange(width)
        l_coords = torch.arange(1)
        image_ids = torch.cartesian_prod(t_coords, h_coords, w_coords, l_coords)  # [height * width, 4]
        image_ids = image_ids.unsqueeze(0).expand(batch_size, -1, -1).to(device)

        text_t_coords = torch.arange(1)
        text_h_coords = torch.arange(1)
        text_w_coords = torch.arange(1)
        text_l_coords = torch.arange(sequence_length)
        text_ids = torch.cartesian_prod(text_t_coords, text_h_coords, text_w_coords, text_l_coords)
        text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1).to(device)

        timestep = torch.tensor([1.0]).to(device).expand(batch_size)
        guidance = torch.tensor([1.0]).to(device).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            "timestep": timestep,
            "guidance": guidance,
        }


class TestFlux2Transformer(Flux2TransformerTesterConfig, ModelTesterMixin):
    pass


class TestFlux2TransformerMemory(Flux2TransformerTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for Flux2 Transformer."""


class TestFlux2TransformerTraining(Flux2TransformerTesterConfig, TrainingTesterMixin):
    """Training tests for Flux2 Transformer."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"Flux2Transformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestFlux2TransformerAttention(Flux2TransformerTesterConfig, AttentionTesterMixin):
    """Attention processor tests for Flux2 Transformer."""


class TestFlux2TransformerContextParallel(Flux2TransformerTesterConfig, ContextParallelTesterMixin):
    """Context Parallel inference tests for Flux2 Transformer."""


class TestFlux2TransformerTensorParallel(Flux2TransformerTesterConfig, TensorParallelTesterMixin):
    """Tensor Parallel inference tests for Flux2 Transformer (CUDA/XPU multi-accelerator)."""


def make_neuron_tp_spec():
    """Model spec consumed by the generic Neuron TP worker (`_neuron_tp_worker.py`).

    Returns `(model_class, init_dict, cpu_inputs)`. Defined here so all Flux2-specific test data lives in this file
    while the worker stays model-agnostic. Reuses the shared tester config so the spec never drifts from the rest of
    the Flux2 tests.
    """
    config = Flux2TransformerTesterConfig()
    return Flux2Transformer2DModel, config.get_init_dict(), config.get_dummy_inputs(device="cpu")


class TestFlux2TransformerTensorParallelTPU(Flux2TransformerTesterConfig, TensorParallelTPUTesterMixin):
    """Tensor Parallel inference test for Flux2 Transformer on TPU."""

    def get_init_dict(self):
        # One head per chip.
        return {**super().get_init_dict(), "num_attention_heads": self.tp_world_size}


@is_tensor_parallel
@require_torch_neuron
class TestFlux2TransformerTensorParallelNeuron:
    """Tensor Parallel inference test for Flux2 Transformer on AWS Neuron.

    Neuron TP runs through `torchrun` with the `"neuron"` distributed backend, so it cannot use the
    `torch.multiprocessing`/NCCL spawn path of `TensorParallelTesterMixin`. This launches the generic worker
    with the Flux2 model spec (`make_neuron_tp_spec`); the worker asserts the sharded output matches a single-device
    reference, and the test checks its exit code.
    """

    def test_tensor_parallel_neuron_inference(self):
        worker = os.path.join(os.path.dirname(__file__), "_neuron_tp_worker.py")
        spec = "tests.models.transformers.test_models_transformer_flux2:make_neuron_tp_spec"
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", worker, spec]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"Neuron tensor-parallel worker failed (exit {result.returncode}).\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


class TestFlux2TransformerLoRA(Flux2TransformerTesterConfig, LoraTesterMixin):
    """LoRA adapter tests for Flux2 Transformer."""


class TestFlux2TransformerLoKr(Flux2TransformerTesterConfig, LoKrTesterMixin):
    """LoKr adapter tests for Flux2 Transformer, including the Flux2 LoKr checkpoint formats."""

    # ai-toolkit stores a placeholder alpha for full-matrix factors, where LoKr applies no scaling.
    placeholder_alpha = torch.tensor(9999220736.0)

    def get_bfl_qkv_state_dict(self, model):
        """A BFL-format LoKr state dict on the fused QKV projections of the first double block."""
        to_q = model.transformer_blocks[0].attn.to_q
        state_dict, expected_deltas = {}, {}
        for bfl_path, diffusers_path in [
            ("double_blocks.0.img_attn.qkv", "transformer_blocks.0.attn.to_qkv"),
            ("double_blocks.0.txt_attn.qkv", "transformer_blocks.0.attn.to_added_qkv"),
        ]:
            factors, expected_deltas[diffusers_path] = make_lokr_factors(3 * to_q.out_features, to_q.in_features)
            state_dict.update({f"diffusion_model.{bfl_path}.{k}": v for k, v in factors.items()})
            state_dict[f"diffusion_model.{bfl_path}.alpha"] = self.placeholder_alpha
        return state_dict, expected_deltas

    @torch.no_grad()
    def test_lokr_bfl_checkpoint(self):
        # BFL checkpoints (e.g. ai-toolkit) apply LoKr to the fused QKV projections. A Kronecker product delta cannot
        # be split exactly into Q/K/V, so loading fuses the model's projections and maps the adapter 1:1.
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict()).eval().to(torch_device)
        state_dict, expected_deltas = self.get_bfl_qkv_state_dict(model)
        for bfl_path, diffusers_path in [
            ("single_blocks.0.linear1", "single_transformer_blocks.0.attn.to_qkv_mlp_proj"),
            ("double_blocks.0.img_attn.proj", "transformer_blocks.0.attn.to_out.0"),
            ("double_blocks.0.img_mlp.0", "transformer_blocks.0.ff.linear_in"),
        ]:
            linear = model.get_submodule(diffusers_path)
            factors, expected_deltas[diffusers_path] = make_lokr_factors(linear.out_features, linear.in_features)
            state_dict.update({f"diffusion_model.{bfl_path}.{k}": v for k, v in factors.items()})
            state_dict[f"diffusion_model.{bfl_path}.alpha"] = self.placeholder_alpha

        converted = Flux2LoraLoaderMixin.lora_state_dict(state_dict)
        model.load_lora_adapter(converted, prefix="transformer", adapter_name="default")

        assert model.transformer_blocks[0].attn.fused_projections
        check_lokr_deltas(model, expected_deltas)

    def test_lokr_fused_qkv_checkpoint_refuses_when_unfused_projections_are_adapted(self):
        # Fusing would replace to_q and orphan the adapter already injected there.
        from peft import LoraConfig

        model = self.model_class(**self.get_init_dict()).eval().to(torch_device)
        model.add_adapter(LoraConfig(r=2, target_modules=["to_q"]), adapter_name="lora")
        state_dict, _ = self.get_bfl_qkv_state_dict(model)
        converted = Flux2LoraLoaderMixin.lora_state_dict(state_dict)

        with pytest.raises(ValueError, match="already loaded on the unfused projections"):
            model.load_lora_adapter(converted, prefix="transformer", adapter_name="lokr")
        assert not model.transformer_blocks[0].attn.fused_projections

    @torch.no_grad()
    def test_lokr_lycoris_checkpoint(self):
        # LyCORIS wraps the diffusers model and encodes module paths with underscores under a `lycoris_` prefix.
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict()).eval().to(torch_device)
        state_dict, expected_deltas = {}, {}
        for diffusers_path in [
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj",
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.attn.to_out.0",
            "transformer_blocks.0.ff.linear_in",
        ]:
            linear = model.get_submodule(diffusers_path)
            factors, expected_deltas[diffusers_path] = make_lokr_factors(linear.out_features, linear.in_features)
            lycoris_path = "lycoris_" + diffusers_path.replace(".", "_")
            state_dict.update({f"{lycoris_path}.{k}": v for k, v in factors.items()})
            state_dict[f"{lycoris_path}.alpha"] = torch.tensor(16.0)

        converted = Flux2LoraLoaderMixin.lora_state_dict(state_dict)
        model.load_lora_adapter(converted, prefix="transformer", adapter_name="default")

        check_lokr_deltas(model, expected_deltas)

    def test_lokr_lycoris_checkpoint_with_unknown_keys_raises(self):
        state_dict = {
            "lycoris_transformer_blocks_0_attn_to_q.lokr_w1": torch.randn(4, 4),
            "lycoris_transformer_blocks_0_attn_norm_q.lokr_w1": torch.randn(4, 4),
        }
        with pytest.raises(ValueError, match="lycoris_transformer_blocks_0_attn_norm_q.lokr_w1"):
            Flux2LoraLoaderMixin.lora_state_dict(state_dict)

    @torch.no_grad()
    def test_lokr_diffusers_names_checkpoint(self):
        # Checkpoints that store the diffusers module paths directly, with alpha keys and no prefix (e.g. SimpleTuner,
        # `bghira/flux2-klein-9b-distillation-lokr`). Alpha scales the rank-decomposed factors only.
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict()).eval().to(torch_device)
        rank, alpha = 2, 1.0
        state_dict, expected_deltas = {}, {}
        for diffusers_path, factor_rank in [
            ("single_transformer_blocks.0.attn.to_out", None),
            ("transformer_blocks.0.attn.to_k", rank),
        ]:
            linear = model.get_submodule(diffusers_path)
            factors, delta = make_lokr_factors(linear.out_features, linear.in_features, rank=factor_rank)
            state_dict.update({f"{diffusers_path}.{k}": v for k, v in factors.items()})
            state_dict[f"{diffusers_path}.alpha"] = torch.tensor(alpha)
            expected_deltas[diffusers_path] = delta if factor_rank is None else (alpha / rank) * delta

        converted = Flux2LoraLoaderMixin.lora_state_dict(state_dict)
        model.load_lora_adapter(converted, prefix="transformer", adapter_name="default")

        check_lokr_deltas(model, expected_deltas)


class TestFlux2TransformerLoRAHotSwap(Flux2TransformerTesterConfig, LoraHotSwappingForModelTesterMixin):
    """LoRA hot-swapping tests for Flux2 Transformer."""

    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        """Override to support dynamic height/width for LoRA hotswap tests."""
        batch_size = 1
        num_latent_channels = 4
        sequence_length = 48
        embedding_dim = 32

        hidden_states = randn_tensor(
            (batch_size, height * width, num_latent_channels), generator=self.generator, device=torch_device
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
        )

        t_coords = torch.arange(1)
        h_coords = torch.arange(height)
        w_coords = torch.arange(width)
        l_coords = torch.arange(1)
        image_ids = torch.cartesian_prod(t_coords, h_coords, w_coords, l_coords)
        image_ids = image_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        text_t_coords = torch.arange(1)
        text_h_coords = torch.arange(1)
        text_w_coords = torch.arange(1)
        text_l_coords = torch.arange(sequence_length)
        text_ids = torch.cartesian_prod(text_t_coords, text_h_coords, text_w_coords, text_l_coords)
        text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)
        guidance = torch.tensor([1.0]).to(torch_device).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            "timestep": timestep,
            "guidance": guidance,
        }


class TestFlux2TransformerCompile(Flux2TransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        """Override to support dynamic height/width for compilation tests."""
        batch_size = 1
        num_latent_channels = 4
        sequence_length = 48
        embedding_dim = 32

        hidden_states = randn_tensor(
            (batch_size, height * width, num_latent_channels), generator=self.generator, device=torch_device
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
        )

        t_coords = torch.arange(1)
        h_coords = torch.arange(height)
        w_coords = torch.arange(width)
        l_coords = torch.arange(1)
        image_ids = torch.cartesian_prod(t_coords, h_coords, w_coords, l_coords)
        image_ids = image_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        text_t_coords = torch.arange(1)
        text_h_coords = torch.arange(1)
        text_w_coords = torch.arange(1)
        text_l_coords = torch.arange(sequence_length)
        text_ids = torch.cartesian_prod(text_t_coords, text_h_coords, text_w_coords, text_l_coords)
        text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)
        guidance = torch.tensor([1.0]).to(torch_device).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            "timestep": timestep,
            "guidance": guidance,
        }


class Flux2TransformerQuantTesterConfig(Flux2TransformerTesterConfig):
    """Shared config for quantized Flux2 Transformer tests (loads the tiny Hub checkpoint)."""

    @property
    def pretrained_model_name_or_path(self):
        return "hf-internal-testing/tiny-flux2"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}

    def get_dummy_inputs(self, height: int = 4, width: int = 4, batch_size: int = 1) -> dict[str, torch.Tensor]:
        """Override to match the tiny Hub checkpoint (joint_attention_dim=16) and the quantizer compute dtype."""
        inputs = super().get_dummy_inputs(height=height, width=width, batch_size=batch_size)
        inputs["encoder_hidden_states"] = randn_tensor(
            (batch_size, inputs["encoder_hidden_states"].shape[1], 16), generator=self.generator, device=torch_device
        )
        return {k: v.to(self.torch_dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}


class TestFlux2TransformerBitsAndBytes(Flux2TransformerQuantTesterConfig, BitsAndBytesTesterMixin):
    """BitsAndBytes quantization tests for Flux2 Transformer."""

    @property
    def torch_dtype(self):
        return torch.float16


class TestFlux2TransformerTorchAo(Flux2TransformerQuantTesterConfig, TorchAoTesterMixin):
    """TorchAO quantization tests for Flux2 Transformer."""

    @property
    def torch_dtype(self):
        return torch.bfloat16


class TestFlux2TransformerGGUF(Flux2TransformerTesterConfig, GGUFTesterMixin):
    """GGUF quantization tests for Flux2 Transformer."""

    @property
    def gguf_filename(self):
        return "https://huggingface.co/unsloth/FLUX.2-dev-GGUF/blob/main/flux2-dev-Q2_K.gguf"

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def get_dummy_inputs(self):
        """Override to provide inputs matching the real FLUX2 model dimensions.

        Flux2 defaults: in_channels=128, joint_attention_dim=15360
        """
        batch_size = 1
        height = 64
        width = 64
        sequence_length = 512

        hidden_states = randn_tensor(
            (batch_size, height * width, 128), generator=self.generator, device=torch_device, dtype=self.torch_dtype
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, 15360), generator=self.generator, device=torch_device, dtype=self.torch_dtype
        )

        # Flux2 uses 4D image/text IDs (t, h, w, l)
        t_coords = torch.arange(1)
        h_coords = torch.arange(height)
        w_coords = torch.arange(width)
        l_coords = torch.arange(1)
        image_ids = torch.cartesian_prod(t_coords, h_coords, w_coords, l_coords)
        image_ids = image_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        text_t_coords = torch.arange(1)
        text_h_coords = torch.arange(1)
        text_w_coords = torch.arange(1)
        text_l_coords = torch.arange(sequence_length)
        text_ids = torch.cartesian_prod(text_t_coords, text_h_coords, text_w_coords, text_l_coords)
        text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        timestep = torch.tensor([1.0]).to(torch_device, self.torch_dtype)
        guidance = torch.tensor([3.5]).to(torch_device, self.torch_dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            "timestep": timestep,
            "guidance": guidance,
        }


class TestFlux2TransformerTorchAoCompile(Flux2TransformerQuantTesterConfig, TorchAoCompileTesterMixin):
    """TorchAO + compile tests for Flux2 Transformer."""

    @property
    def torch_dtype(self):
        return torch.bfloat16


class TestFlux2TransformerGGUFCompile(Flux2TransformerTesterConfig, GGUFCompileTesterMixin):
    """GGUF + compile tests for Flux2 Transformer."""

    @property
    def gguf_filename(self):
        return "https://huggingface.co/unsloth/FLUX.2-dev-GGUF/blob/main/flux2-dev-Q2_K.gguf"

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def get_dummy_inputs(self):
        """Override to provide inputs matching the real FLUX2 model dimensions.

        Flux2 defaults: in_channels=128, joint_attention_dim=15360
        """
        batch_size = 1
        height = 64
        width = 64
        sequence_length = 512

        hidden_states = randn_tensor(
            (batch_size, height * width, 128), generator=self.generator, device=torch_device, dtype=self.torch_dtype
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, 15360), generator=self.generator, device=torch_device, dtype=self.torch_dtype
        )

        # Flux2 uses 4D image/text IDs (t, h, w, l)
        t_coords = torch.arange(1)
        h_coords = torch.arange(height)
        w_coords = torch.arange(width)
        l_coords = torch.arange(1)
        image_ids = torch.cartesian_prod(t_coords, h_coords, w_coords, l_coords)
        image_ids = image_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        text_t_coords = torch.arange(1)
        text_h_coords = torch.arange(1)
        text_w_coords = torch.arange(1)
        text_l_coords = torch.arange(sequence_length)
        text_ids = torch.cartesian_prod(text_t_coords, text_h_coords, text_w_coords, text_l_coords)
        text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        timestep = torch.tensor([1.0]).to(torch_device, self.torch_dtype)
        guidance = torch.tensor([3.5]).to(torch_device, self.torch_dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            "timestep": timestep,
            "guidance": guidance,
        }


class Flux2TransformerKVCacheTesterConfig(BaseModelTesterConfig):
    num_ref_tokens = 4

    @property
    def model_class(self):
        return Flux2Transformer2DModel

    @property
    def output_shape(self) -> tuple[int, int]:
        return (16, 4)

    @property
    def input_shape(self) -> tuple[int, int]:
        return (16, 4)

    @property
    def model_split_percents(self) -> list:
        return [0.7, 0.6, 0.6]

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def uses_custom_attn_processor(self) -> bool:
        return True

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int]]:
        return {
            "patch_size": 1,
            "in_channels": 4,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 16,
            "num_attention_heads": 2,
            "joint_attention_dim": 32,
            "timestep_guidance_channels": 256,
            "axes_dims_rope": [4, 4, 4, 4],
        }

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_latent_channels = 4
        sequence_length = 48
        embedding_dim = 32
        num_ref_tokens = self.num_ref_tokens

        ref_hidden_states = randn_tensor(
            (batch_size, num_ref_tokens, num_latent_channels), generator=self.generator, device=torch_device
        )
        img_hidden_states = randn_tensor(
            (batch_size, height * width, num_latent_channels), generator=self.generator, device=torch_device
        )
        hidden_states = torch.cat([ref_hidden_states, img_hidden_states], dim=1)

        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
        )

        ref_t_coords = torch.arange(1)
        ref_h_coords = torch.arange(num_ref_tokens)
        ref_w_coords = torch.arange(1)
        ref_l_coords = torch.arange(1)
        ref_ids = torch.cartesian_prod(ref_t_coords, ref_h_coords, ref_w_coords, ref_l_coords)
        ref_ids = ref_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        t_coords = torch.arange(1)
        h_coords = torch.arange(height)
        w_coords = torch.arange(width)
        l_coords = torch.arange(1)
        image_ids = torch.cartesian_prod(t_coords, h_coords, w_coords, l_coords)
        image_ids = image_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)
        image_ids = torch.cat([ref_ids, image_ids], dim=1)

        text_t_coords = torch.arange(1)
        text_h_coords = torch.arange(1)
        text_w_coords = torch.arange(1)
        text_l_coords = torch.arange(sequence_length)
        text_ids = torch.cartesian_prod(text_t_coords, text_h_coords, text_w_coords, text_l_coords)
        text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)
        guidance = torch.tensor([1.0]).to(torch_device).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            "timestep": timestep,
            "guidance": guidance,
        }


class TestFlux2TransformerKVCache(Flux2TransformerKVCacheTesterConfig):
    """KV cache tests for Flux2 Transformer."""

    def test_kv_layer_cache_store_and_get(self):
        cache = Flux2KVLayerCache()
        k = torch.randn(1, 4, 2, 16)
        v = torch.randn(1, 4, 2, 16)
        cache.store(k, v)
        k_out, v_out = cache.get()
        assert torch.equal(k, k_out)
        assert torch.equal(v, v_out)

    def test_kv_layer_cache_get_before_store_raises(self):
        cache = Flux2KVLayerCache()
        try:
            cache.get()
            assert False, "Expected RuntimeError"
        except RuntimeError:
            pass

    def test_kv_layer_cache_clear(self):
        cache = Flux2KVLayerCache()
        cache.store(torch.randn(1, 4, 2, 16), torch.randn(1, 4, 2, 16))
        cache.clear()
        assert cache.k_ref is None
        assert cache.v_ref is None

    def test_kv_cache_structure(self):
        num_double = 3
        num_single = 2
        cache = Flux2KVCache(num_double, num_single)
        assert len(cache.double_block_caches) == num_double
        assert len(cache.single_block_caches) == num_single
        assert cache.num_ref_tokens == 0

        for i in range(num_double):
            assert isinstance(cache.get_double(i), Flux2KVLayerCache)
        for i in range(num_single):
            assert isinstance(cache.get_single(i), Flux2KVLayerCache)

    def test_kv_cache_clear(self):
        cache = Flux2KVCache(2, 1)
        cache.num_ref_tokens = 4
        cache.get_double(0).store(torch.randn(1, 4, 2, 16), torch.randn(1, 4, 2, 16))
        cache.clear()
        assert cache.num_ref_tokens == 0
        assert cache.get_double(0).k_ref is None

    def _set_kv_attn_processors(self, model):
        for block in model.transformer_blocks:
            block.attn.set_processor(Flux2KVAttnProcessor())
        for block in model.single_transformer_blocks:
            block.attn.set_processor(Flux2KVParallelSelfAttnProcessor())

    @torch.no_grad()
    def test_extract_mode_returns_cache(self):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()
        self._set_kv_attn_processors(model)

        output = model(
            **self.get_dummy_inputs(),
            kv_cache_mode="extract",
            num_ref_tokens=self.num_ref_tokens,
            ref_fixed_timestep=0.0,
        )

        assert output.kv_cache is not None
        assert isinstance(output.kv_cache, Flux2KVCache)
        assert output.kv_cache.num_ref_tokens == self.num_ref_tokens

        for layer_cache in output.kv_cache.double_block_caches:
            assert layer_cache.k_ref is not None
            assert layer_cache.v_ref is not None

        for layer_cache in output.kv_cache.single_block_caches:
            assert layer_cache.k_ref is not None
            assert layer_cache.v_ref is not None

    @torch.no_grad()
    def test_extract_mode_output_shape(self):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        height, width = 4, 4
        output = model(
            **self.get_dummy_inputs(height=height, width=width),
            kv_cache_mode="extract",
            num_ref_tokens=self.num_ref_tokens,
            ref_fixed_timestep=0.0,
        )

        assert output.sample.shape == (1, height * width, 4)

    @torch.no_grad()
    def test_cached_mode_uses_cache(self):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        height, width = 4, 4
        extract_output = model(
            **self.get_dummy_inputs(height=height, width=width),
            kv_cache_mode="extract",
            num_ref_tokens=self.num_ref_tokens,
            ref_fixed_timestep=0.0,
        )

        base_config = Flux2TransformerTesterConfig()
        cached_inputs = base_config.get_dummy_inputs(height=height, width=width)
        cached_output = model(
            **cached_inputs,
            kv_cache=extract_output.kv_cache,
            kv_cache_mode="cached",
        )

        assert cached_output.sample.shape == (1, height * width, 4)
        assert cached_output.kv_cache is None

    @torch.no_grad()
    def test_extract_return_dict_false(self):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        output = model(
            **self.get_dummy_inputs(),
            kv_cache_mode="extract",
            num_ref_tokens=self.num_ref_tokens,
            ref_fixed_timestep=0.0,
            return_dict=False,
        )

        assert isinstance(output, tuple)
        assert len(output) == 2
        assert isinstance(output[1], Flux2KVCache)

    @torch.no_grad()
    def test_no_kv_cache_mode_returns_no_cache(self):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        base_config = Flux2TransformerTesterConfig()
        output = model(**base_config.get_dummy_inputs())

        assert output.kv_cache is None


class TestFlux2Transformer2DSingleFile(Flux2TransformerTesterConfig, SingleFileTesterMixin):
    @property
    def ckpt_path(self):
        return "https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/flux2-dev.safetensors"

    @property
    def pretrained_model_name_or_path(self):
        return "black-forest-labs/FLUX.2-dev"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}
