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

from diffusers import Flux3Transformer2DModel
from diffusers.hooks import HookRegistry, ModelHook
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    BFLCompileTesterMixin,
    BFLTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    SingleFileTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class Flux3TransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return Flux3Transformer2DModel

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (16, 4)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (16, 4)

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
            "in_channels": 4,
            "cond_in_channels": 4,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 16,
            "num_attention_heads": 2,
            "joint_attention_dim": 32,
            "timestep_channels": 32,
            "axes_dims_rope": [4, 4, 4, 4],
        }

    def get_dummy_inputs(
        self, height: int = 4, width: int = 4, batch_size: int = 1, num_cond_tokens: int | None = None
    ) -> dict[str, torch.Tensor]:
        # all per-mode blocks share one code object; equal text/image lengths keep compiled shapes uniform
        num_tokens = height * width
        generator = self.generator
        inputs = {
            "hidden_states": randn_tensor((batch_size, num_tokens, 4), generator=generator, device=torch_device),
            "encoder_hidden_states": randn_tensor(
                (batch_size, num_tokens, 32), generator=generator, device=torch_device
            ),
            "timestep": torch.full((batch_size,), 0.5, device=torch_device),
            "img_ids": torch.zeros(num_tokens, 4, device=torch_device),
            "txt_ids": torch.zeros(num_tokens, 4, device=torch_device),
        }
        if num_cond_tokens is not None:
            inputs["image_cond_hidden_states"] = randn_tensor(
                (batch_size, num_cond_tokens, 4), generator=generator, device=torch_device
            )
            inputs["image_cond_ids"] = torch.zeros(num_cond_tokens, 4, device=torch_device)
        return inputs


class TestFlux3TransformerModel(Flux3TransformerTesterConfig, ModelTesterMixin):
    @torch.no_grad()
    def test_image_cond_inputs(self):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()

        # a distinct reference token count: the output is sliced back to the image tokens only
        inputs = self.get_dummy_inputs(num_cond_tokens=10)
        output = model(**inputs, return_dict=False)[0]
        assert output.shape == inputs["hidden_states"].shape

        # the references change the prediction
        text_only = model(**{k: v for k, v in inputs.items() if not k.startswith("image_cond")}, return_dict=False)[0]
        assert not torch.allclose(output, text_only)

    @torch.no_grad()
    def test_flow_map(self):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict(), flow_map=True).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        target_timestep = torch.full_like(inputs["timestep"], 0.25)

        output = model(**inputs, target_timestep=target_timestep, return_dict=False)[0]
        assert output.shape == inputs["hidden_states"].shape

        # the target timestep changes the prediction
        other = model(**inputs, target_timestep=torch.zeros_like(target_timestep), return_dict=False)[0]
        assert not torch.allclose(output, other)

        # a flow map requires the target timestep; a plain model refuses it
        with pytest.raises(ValueError, match="target_timestep"):
            model(**inputs, return_dict=False)
        plain = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        assert plain.target_timestep_embedder is None
        with pytest.raises(ValueError, match="target_timestep"):
            plain(**inputs, target_timestep=target_timestep, return_dict=False)

    def _cfg_inputs(self, num_cond_tokens=None):
        # two guider batches: the same image inputs, different text inputs
        cond = self.get_dummy_inputs(num_cond_tokens=num_cond_tokens)
        uncond = dict(cond)
        uncond["encoder_hidden_states"] = randn_tensor(
            cond["encoder_hidden_states"].shape, generator=torch.Generator("cpu").manual_seed(1), device=torch_device
        )
        uncond["txt_ids"] = torch.ones_like(cond["txt_ids"])
        return cond, uncond

    def _count_image_prefix_calls(self, model):
        calls = []
        handle = model.x_embedders["image"].register_forward_hook(lambda *args: calls.append(1))
        return calls, handle

    @pytest.mark.parametrize("flow_map", [False, True])
    @pytest.mark.parametrize("num_cond_tokens", [None, 10])
    @torch.no_grad()
    def test_cache_image_prefix(self, flow_map, num_cond_tokens):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict(), flow_map=flow_map).to(torch_device).eval()
        cond, uncond = self._cfg_inputs(num_cond_tokens)
        if flow_map:
            cond["target_timestep"] = uncond["target_timestep"] = torch.full_like(cond["timestep"], 0.25)

        expected = [model(**inputs, return_dict=False)[0] for inputs in (cond, uncond)]
        calls, handle = self._count_image_prefix_calls(model)
        with model.cache_image_prefix():
            outputs = [model(**inputs, return_dict=False)[0] for inputs in (cond, uncond)]
            assert model._image_prefix_cache
        handle.remove()

        assert len(calls) == 1
        assert model._image_prefix_cache is None
        assert not torch.equal(expected[0], expected[1])
        for output, reference in zip(outputs, expected):
            assert torch.equal(output, reference)

    @torch.no_grad()
    def test_cache_image_prefix_new_image_inputs(self):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        cond, uncond = self._cfg_inputs()
        # equal values in a new tensor
        uncond["hidden_states"] = cond["hidden_states"].clone()

        calls, handle = self._count_image_prefix_calls(model)
        with model.cache_image_prefix():
            model(**cond, return_dict=False)
            model(**uncond, return_dict=False)
            assert len(calls) == 2
            # an in-place update of the cached tensor
            uncond["hidden_states"].add_(1.0)
            output = model(**uncond, return_dict=False)[0]
            assert len(calls) == 3
        handle.remove()

        assert torch.equal(output, model(**uncond, return_dict=False)[0])

    @pytest.mark.parametrize(
        "guard", ["grad", "prefix_hook", "other_hook", "prefix_hook_between_calls", "context_parallel"]
    )
    def test_cache_image_prefix_disabled(self, guard):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        cond, uncond = self._cfg_inputs()
        with torch.set_grad_enabled(guard == "grad"):
            expected = [model(**inputs, return_dict=False)[0] for inputs in (cond, uncond)]

        def add_hook(module):
            HookRegistry.check_if_exists_or_initialize(module).register_hook(ModelHook(), "test_hook")

        if guard == "prefix_hook":
            add_hook(model.mode_blocks["image"][0])
        elif guard == "other_hook":
            add_hook(model.single_transformer_blocks[0])
        elif guard == "context_parallel":
            model._parallel_config = object()

        calls, handle = self._count_image_prefix_calls(model)
        with torch.set_grad_enabled(guard == "grad"), model.cache_image_prefix():
            outputs = [model(**cond, return_dict=False)[0]]
            if guard == "prefix_hook_between_calls":
                # a guider registers its layer hooks for one batch only (e.g. perturbed-attention guidance)
                add_hook(model.mode_blocks["image"][0])
            outputs.append(model(**uncond, return_dict=False)[0])
        handle.remove()

        assert len(calls) == 2
        for output, reference in zip(outputs, expected):
            assert torch.equal(output, reference)


class TestFlux3TransformerMemory(Flux3TransformerTesterConfig, MemoryTesterMixin):
    pass


class TestFlux3TransformerTorchCompile(Flux3TransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]


class TestFlux3TransformerTraining(Flux3TransformerTesterConfig, TrainingTesterMixin):
    pass


class TestFlux3TransformerAttention(Flux3TransformerTesterConfig, AttentionTesterMixin):
    pass


class TestFlux3SingleFile(Flux3TransformerTesterConfig, SingleFileTesterMixin):
    @property
    def ckpt_path(self):
        return "https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/model.safetensors"

    @property
    def pretrained_model_name_or_path(self):
        return "black-forest-labs/flux-3-image-klein-base"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}

    @property
    def torch_dtype(self):
        return torch.bfloat16


class Flux3TransformerBFLTesterConfig(Flux3TransformerTesterConfig):
    """Shared config for the BFL quantization tests: which linears to quantize comes from the real checkpoints'
    headers; the model itself is the small test model in the compute dtype."""

    @property
    def bfl_checkpoints(self):
        return {
            "fp8r": "https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/variants/fp8r/model.safetensors",
            "nvfp4": "https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/variants/nvfp4/model.safetensors",
        }

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def get_dummy_inputs(self, **kwargs):
        """Override to build inputs in the quantizer compute dtype (unquantized linears do strict-dtype matmuls)."""
        inputs = super().get_dummy_inputs(**kwargs)
        return {k: v.to(self.torch_dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}


class TestFlux3TransformerBFL(Flux3TransformerBFLTesterConfig, BFLTesterMixin):
    """BFL fp8r / nvfp4 quantization tests for the FLUX.3 image transformer."""


class TestFlux3TransformerBFLCompile(Flux3TransformerBFLTesterConfig, BFLCompileTesterMixin):
    """BFL fp8r / nvfp4 + torch.compile tests for the FLUX.3 image transformer."""
