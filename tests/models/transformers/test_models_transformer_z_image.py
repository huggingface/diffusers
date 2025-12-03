# coding=utf-8
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

import gc
import os
import unittest

import torch

from diffusers import ZImageTransformer2DModel

from ...testing_utils import IS_GITHUB_ACTIONS, torch_device
from ..test_modeling_common import ModelTesterMixin, TorchCompileTesterMixin


# Z-Image requires torch.use_deterministic_algorithms(False) due to complex64 RoPE operations
# Cannot use enable_full_determinism() which sets it to True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if hasattr(torch.backends, "cuda"):
    torch.backends.cuda.matmul.allow_tf32 = False


@unittest.skipIf(
    IS_GITHUB_ACTIONS,
    reason="Skipping test-suite inside the CI because the model has `torch.empty()` inside of it during init and we don't have a clear way to override it in the modeling tests.",
)
class ZImageTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = ZImageTransformer2DModel
    main_input_name = "x"
    # We override the items here because the transformer under consideration is small.
    model_split_percents = [0.9, 0.9, 0.9]

    def prepare_dummy_input(self, height=16, width=16):
        batch_size = 1
        num_channels = 16
        embedding_dim = 16
        sequence_length = 16

        hidden_states = [torch.randn((num_channels, 1, height, width)).to(torch_device) for _ in range(batch_size)]
        encoder_hidden_states = [
            torch.randn((sequence_length, embedding_dim)).to(torch_device) for _ in range(batch_size)
        ]
        timestep = torch.tensor([0.0]).to(torch_device)

        return {"x": hidden_states, "cap_feats": encoder_hidden_states, "t": timestep}

    @property
    def dummy_input(self):
        return self.prepare_dummy_input()

    @property
    def input_shape(self):
        return (4, 32, 32)

    @property
    def output_shape(self):
        return (4, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
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
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def setUp(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"ZImageTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    @unittest.skip("Test is not supported for handling main inputs that are lists.")
    def test_training(self):
        super().test_training()

    @unittest.skip("Test is not supported for handling main inputs that are lists.")
    def test_ema_training(self):
        super().test_ema_training()

    @unittest.skip("Test is not supported for handling main inputs that are lists.")
    def test_effective_gradient_checkpointing(self):
        super().test_effective_gradient_checkpointing()

    @unittest.skip(
        "Test needs to be revisited. But we need to ensure `x_pad_token` and `cap_pad_token` are cast to the same dtype as the destination tensor before they are assigned to the padding indices."
    )
    def test_layerwise_casting_training(self):
        super().test_layerwise_casting_training()

    @unittest.skip("Test is not supported for handling main inputs that are lists.")
    def test_outputs_equivalence(self):
        super().test_outputs_equivalence()

    @unittest.skip("Test will pass if we change to deterministic values instead of empty in the DiT.")
    def test_group_offloading(self):
        super().test_group_offloading()

    @unittest.skip("Test will pass if we change to deterministic values instead of empty in the DiT.")
    def test_group_offloading_with_disk(self):
        super().test_group_offloading_with_disk()


class ZImageTransformerCompileTests(TorchCompileTesterMixin, unittest.TestCase):
    model_class = ZImageTransformer2DModel
    different_shapes_for_compilation = [(4, 4), (4, 8), (8, 8)]

    def prepare_init_args_and_inputs_for_common(self):
        return ZImageTransformerTests().prepare_init_args_and_inputs_for_common()

    def prepare_dummy_input(self, height, width):
        return ZImageTransformerTests().prepare_dummy_input(height=height, width=width)

    @unittest.skip(
        "The repeated block in this model is ZImageTransformerBlock, which is used for noise_refiner, context_refiner, and layers. As a consequence of this, the inputs recorded for the block would vary during compilation and full compilation with fullgraph=True would trigger recompilation at least thrice."
    )
    def test_torch_compile_recompilation_and_graph_break(self):
        super().test_torch_compile_recompilation_and_graph_break()

    @unittest.skip("Fullgraph AoT is broken")
    def test_compile_works_with_aot(self):
        super().test_compile_works_with_aot()

    @unittest.skip("Fullgraph is broken")
    def test_compile_on_different_shapes(self):
        super().test_compile_on_different_shapes()
