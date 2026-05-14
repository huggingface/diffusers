# Copyright 2025 HuggingFace Inc.
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

import pytest
import torch

from diffusers import ZImageTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import assert_tensors_close, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    LoraTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


# Z-Image requires torch.use_deterministic_algorithms(False) due to complex64 RoPE operations
# Cannot use enable_full_determinism() which sets it to True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if hasattr(torch.backends, "cuda"):
    torch.backends.cuda.matmul.allow_tf32 = False


def _concat_list_output(output):
    """Model output `sample` is a list of tensors. Concatenate them for comparison."""
    return torch.cat([t.flatten() for t in output])


class ZImageTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return ZImageTransformer2DModel

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (4, 32, 32)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (4, 32, 32)

    @property
    def model_split_percents(self) -> list:
        return [0.9, 0.9, 0.9]

    @property
    def main_input_name(self) -> str:
        return "x"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "all_patch_size": (2,),
            "all_f_patch_size": (1,),
            "in_channels": 16,
            "dim": 16,
            "n_layers": 1,
            "n_refiner_layers": 1,
            "n_heads": 1,
            "n_kv_heads": 2,
            "qk_norm": True,
            "cap_feat_dim": 16,
            "rope_theta": 256.0,
            "t_scale": 1000.0,
            "axes_dims": [8, 4, 4],
            "axes_lens": [256, 32, 32],
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor | list]:
        batch_size = 1
        num_channels = 16
        embedding_dim = 16
        sequence_length = 16
        height = 16
        width = 16

        hidden_states = [
            randn_tensor((num_channels, 1, height, width), generator=self.generator, device=torch_device)
            for _ in range(batch_size)
        ]
        encoder_hidden_states = [
            randn_tensor((sequence_length, embedding_dim), generator=self.generator, device=torch_device)
            for _ in range(batch_size)
        ]
        timestep = torch.tensor([0.0]).to(torch_device)

        return {"x": hidden_states, "cap_feats": encoder_hidden_states, "t": timestep}


class TestZImageTransformer(ZImageTransformerTesterConfig, ModelTesterMixin):
    """Core model tests for Z-Image Transformer."""

    @torch.no_grad()
    def test_determinism(self, atol=1e-5, rtol=0):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        inputs_dict = self.get_dummy_inputs()
        first = _concat_list_output(model(**inputs_dict, return_dict=False)[0])
        second = _concat_list_output(model(**inputs_dict, return_dict=False)[0])

        mask = ~(torch.isnan(first) | torch.isnan(second))
        assert_tensors_close(
            first[mask], second[mask], atol=atol, rtol=rtol, msg="Model outputs are not deterministic"
        )

    def test_from_save_pretrained(self, tmp_path, atol=5e-5, rtol=5e-5):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        model.save_pretrained(tmp_path)
        new_model = self.model_class.from_pretrained(tmp_path)
        new_model.to(torch_device)

        for param_name in model.state_dict().keys():
            param_1 = model.state_dict()[param_name]
            param_2 = new_model.state_dict()[param_name]
            assert param_1.shape == param_2.shape

        inputs_dict = self.get_dummy_inputs()
        image = _concat_list_output(model(**inputs_dict, return_dict=False)[0])
        new_image = _concat_list_output(new_model(**inputs_dict, return_dict=False)[0])

        assert_tensors_close(image, new_image, atol=atol, rtol=rtol, msg="Models give different forward passes.")

    @torch.no_grad()
    def test_from_save_pretrained_variant(self, tmp_path, atol=5e-5, rtol=0):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        model.save_pretrained(tmp_path, variant="fp16")
        new_model = self.model_class.from_pretrained(tmp_path, variant="fp16")

        with pytest.raises(OSError) as exc_info:
            self.model_class.from_pretrained(tmp_path)

        assert "Error no file named diffusion_pytorch_model.bin found in directory" in str(exc_info.value)

        new_model.to(torch_device)

        inputs_dict = self.get_dummy_inputs()
        image = _concat_list_output(model(**inputs_dict, return_dict=False)[0])
        new_image = _concat_list_output(new_model(**inputs_dict, return_dict=False)[0])

        assert_tensors_close(image, new_image, atol=atol, rtol=rtol, msg="Models give different forward passes.")

    @pytest.mark.skip("Model output `sample` is a list of tensors, not a single tensor.")
    def test_outputs_equivalence(self, atol=1e-5, rtol=0):
        pass

    def test_sharded_checkpoints_with_parallel_loading(self, tmp_path, atol=1e-5, rtol=0):
        from diffusers.utils import SAFE_WEIGHTS_INDEX_NAME, constants

        from ..testing_utils.common import calculate_expected_num_shards, compute_module_persistent_sizes

        torch.manual_seed(0)
        config = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**config).eval()
        model = model.to(torch_device)

        base_output = _concat_list_output(model(**inputs_dict, return_dict=False)[0])

        model_size = compute_module_persistent_sizes(model)[""]
        max_shard_size = int((model_size * 0.75) / (2**10))

        original_parallel_loading = constants.HF_ENABLE_PARALLEL_LOADING
        original_parallel_workers = getattr(constants, "HF_PARALLEL_WORKERS", None)

        try:
            model.cpu().save_pretrained(tmp_path, max_shard_size=f"{max_shard_size}KB")
            assert os.path.exists(os.path.join(tmp_path, SAFE_WEIGHTS_INDEX_NAME))

            expected_num_shards = calculate_expected_num_shards(os.path.join(tmp_path, SAFE_WEIGHTS_INDEX_NAME))
            actual_num_shards = len([file for file in os.listdir(tmp_path) if file.endswith(".safetensors")])
            assert actual_num_shards == expected_num_shards

            constants.HF_ENABLE_PARALLEL_LOADING = False
            self.model_class.from_pretrained(tmp_path).eval().to(torch_device)

            constants.HF_ENABLE_PARALLEL_LOADING = True
            constants.DEFAULT_HF_PARALLEL_LOADING_WORKERS = 2

            torch.manual_seed(0)
            model_parallel = self.model_class.from_pretrained(tmp_path).eval()
            model_parallel = model_parallel.to(torch_device)

            output_parallel = _concat_list_output(model_parallel(**inputs_dict, return_dict=False)[0])

            assert_tensors_close(
                base_output, output_parallel, atol=atol, rtol=rtol, msg="Output should match with parallel loading"
            )
        finally:
            constants.HF_ENABLE_PARALLEL_LOADING = original_parallel_loading
            if original_parallel_workers is not None:
                constants.HF_PARALLEL_WORKERS = original_parallel_workers


class TestZImageTransformerMemory(ZImageTransformerTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for Z-Image Transformer."""

    @pytest.mark.skip(
        "Ensure `x_pad_token` and `cap_pad_token` are cast to the same dtype as the destination tensor before they are assigned to the padding indices."
    )
    def test_layerwise_casting_training(self):
        pass


class TestZImageTransformerTraining(ZImageTransformerTesterConfig, TrainingTesterMixin):
    """Training tests for Z-Image Transformer."""

    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set={"ZImageTransformer2DModel"})

    @pytest.mark.skip("Test is not supported for handling main inputs that are lists.")
    def test_training(self):
        pass

    @pytest.mark.skip("Test is not supported for handling main inputs that are lists.")
    def test_training_with_ema(self):
        pass

    @pytest.mark.skip("Test is not supported for handling main inputs that are lists.")
    def test_gradient_checkpointing_equivalence(self, loss_tolerance=1e-5, param_grad_tol=5e-5, skip=None):
        pass


class TestZImageTransformerLoRA(ZImageTransformerTesterConfig, LoraTesterMixin):
    """LoRA adapter tests for Z-Image Transformer."""

    @pytest.mark.skip("Model output `sample` is a list of tensors, not a single tensor.")
    def test_save_load_lora_adapter(self, tmp_path, rank=4, lora_alpha=4, use_dora=False, atol=1e-4, rtol=1e-4):
        pass


# TODO: Add pretrained_model_name_or_path once a tiny Z-Image model is available on the Hub
# class TestZImageTransformerBitsAndBytes(ZImageTransformerTesterConfig, BitsAndBytesTesterMixin):
#     """BitsAndBytes quantization tests for Z-Image Transformer."""


# TODO: Add pretrained_model_name_or_path once a tiny Z-Image model is available on the Hub
# class TestZImageTransformerTorchAo(ZImageTransformerTesterConfig, TorchAoTesterMixin):
#     """TorchAo quantization tests for Z-Image Transformer."""


class TestZImageTransformerCompile(ZImageTransformerTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for Z-Image Transformer."""

    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 16, width: int = 16) -> dict[str, torch.Tensor | list]:
        batch_size = 1
        num_channels = 16
        embedding_dim = 16
        sequence_length = 16

        hidden_states = [
            randn_tensor((num_channels, 1, height, width), generator=self.generator, device=torch_device)
            for _ in range(batch_size)
        ]
        encoder_hidden_states = [
            randn_tensor((sequence_length, embedding_dim), generator=self.generator, device=torch_device)
            for _ in range(batch_size)
        ]
        timestep = torch.tensor([0.0]).to(torch_device)

        return {"x": hidden_states, "cap_feats": encoder_hidden_states, "t": timestep}

    @pytest.mark.skip(
        "The repeated block in this model is ZImageTransformerBlock, which is used for noise_refiner, context_refiner, and layers. The inputs recorded for the block would vary during compilation and full compilation with fullgraph=True would trigger recompilation at least thrice."
    )
    def test_torch_compile_recompilation_and_graph_break(self):
        pass

    @pytest.mark.skip("Fullgraph AoT is broken")
    def test_compile_works_with_aot(self, tmp_path):
        pass

    @pytest.mark.skip("Fullgraph is broken")
    def test_compile_on_different_shapes(self):
        pass
