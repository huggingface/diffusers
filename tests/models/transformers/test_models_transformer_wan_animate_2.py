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

from diffusers import WanAnimate2Transformer3DModel
from diffusers.models.transformers.transformer_wan_animate_2 import WanAnimate2KVCache
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import assert_tensors_close, enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


# One 5-frame, 32x32 segment, sized the way the pipeline's geometry step derives it: 32 // 8 = 4 latent
# pixels per side, halved again by the (1, 2, 2) patch size, and (5 - 1) // 4 + 1 = 2 latent frames — a
# 2 x 2 x 2 patch grid. The generation stream carries one extra latent frame, the reference image's slot.
SEGMENT_FRAME_LENGTH = 5
SEGMENT_AREA = [32, 32]
LATENT_HEIGHT = 4
LATENT_WIDTH = 4
REFERENCE_LATENT_FRAMES = 2
GENERATION_LATENT_FRAMES = REFERENCE_LATENT_FRAMES + 1
PATCHES_PER_FRAME = (LATENT_HEIGHT // 2) * (LATENT_WIDTH // 2)

LATENT_CHANNELS = 16
CONDITION_CHANNELS = 20  # the conditioning latents plus a 4-channel i2v mask
TEXT_SEQ_LEN = 8
TEXT_DIM = 32
CLIP_SEQ_LEN = 4
CLIP_DIM = 1280  # `MLPProj` hardcodes the CLIP vision hidden size
NUM_LAYERS = 2


class WanAnimate2TransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return WanAnimate2Transformer3DModel

    @property
    def pretrained_model_name_or_path(self):
        return "hf-internal-testing/tiny-wan-animate-2-modular"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (LATENT_CHANNELS, REFERENCE_LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (LATENT_CHANNELS, REFERENCE_LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | tuple | bool | float]:
        # Mirrors the config of the tiny checkpoint in `pretrained_model_name_or_path`.
        return {
            "patch_size": (1, 2, 2),
            "text_len": 16,
            "in_dim": LATENT_CHANNELS + CONDITION_CHANNELS,
            "dim": 128,
            "ffn_dim": 256,
            "freq_dim": 64,
            "text_dim": TEXT_DIM,
            "out_dim": LATENT_CHANNELS,
            "num_heads": 4,
            "num_layers": NUM_LAYERS,
            "cross_attn_norm": True,
            "use_img_emb": True,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor | list | int | str]:
        # The reference pass (`kv_cache_mode="extract"`), which the pipeline runs once per segment before
        # denoising: dense self-attention over the driving-video latents, writing every layer's K/V into a
        # fresh cache. The generation pass needs a populated cache, so it cannot be built from inputs alone
        # — `get_dummy_generation_inputs` covers it, driven from a cache this pass filled.
        generator = self.generator
        return {
            "hidden_states": [
                randn_tensor(
                    (LATENT_CHANNELS, REFERENCE_LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH),
                    generator=generator,
                    device=torch_device,
                    dtype=self.torch_dtype,
                )
            ],
            "timestep": torch.tensor([1.0], device=torch_device, dtype=self.torch_dtype),
            "encoder_hidden_states": [
                randn_tensor(
                    (TEXT_SEQ_LEN, TEXT_DIM), generator=generator, device=torch_device, dtype=self.torch_dtype
                )
            ],
            "condition_latents": [
                randn_tensor(
                    (CONDITION_CHANNELS, REFERENCE_LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH),
                    generator=generator,
                    device=torch_device,
                    dtype=self.torch_dtype,
                )
            ],
            "encoder_hidden_states_image": randn_tensor(
                (1, CLIP_SEQ_LEN, CLIP_DIM), generator=generator, device=torch_device, dtype=self.torch_dtype
            ),
            "kv_cache": WanAnimate2KVCache(NUM_LAYERS),
            "kv_cache_mode": "extract",
            "seq_len": REFERENCE_LATENT_FRAMES * PATCHES_PER_FRAME,
            "offset_grid_sizes": torch.tensor(
                [[REFERENCE_LATENT_FRAMES, LATENT_HEIGHT // 2, LATENT_WIDTH // 2]], dtype=torch.long
            ),
        }

    def get_dummy_generation_inputs(self, kv_cache: WanAnimate2KVCache) -> dict[str, torch.Tensor | list | int | str]:
        """Inputs for one denoising step (`kv_cache_mode="cached"`) against an already populated `kv_cache`."""
        generator = self.generator
        return {
            "hidden_states": [
                randn_tensor(
                    (LATENT_CHANNELS, GENERATION_LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH),
                    generator=generator,
                    device=torch_device,
                    dtype=self.torch_dtype,
                )
            ],
            "timestep": torch.tensor([500.0], device=torch_device, dtype=self.torch_dtype),
            "encoder_hidden_states": [
                randn_tensor(
                    (TEXT_SEQ_LEN, TEXT_DIM), generator=generator, device=torch_device, dtype=self.torch_dtype
                )
            ],
            "condition_latents": [
                randn_tensor(
                    (CONDITION_CHANNELS, GENERATION_LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH),
                    generator=generator,
                    device=torch_device,
                    dtype=self.torch_dtype,
                )
            ],
            "encoder_hidden_states_image": randn_tensor(
                (1, CLIP_SEQ_LEN, CLIP_DIM), generator=generator, device=torch_device, dtype=self.torch_dtype
            ),
            "kv_cache": kv_cache,
            "kv_cache_mode": "cached",
            "seq_len": GENERATION_LATENT_FRAMES * PATCHES_PER_FRAME,
            "reference_grid_sizes": torch.tensor(
                [[REFERENCE_LATENT_FRAMES, LATENT_HEIGHT // 2, LATENT_WIDTH // 2]], dtype=torch.long
            ),
            "origin_len": SEGMENT_FRAME_LENGTH,
            "origin_area": SEGMENT_AREA,
        }


class TestWanAnimate2Transformer(WanAnimate2TransformerTesterConfig, ModelTesterMixin):
    """Core model tests for the Wan-Animate-2 transformer."""

    @torch.no_grad()
    def test_output(self, base_model_output):
        # The base version compares `output[0].shape` against a single expected shape; this model returns one
        # unpatchified tensor per input latent, so assert on the length and on every element.
        assert len(base_model_output) == 1, f"Expected one sample per input latent, got {len(base_model_output)}"
        for sample in base_model_output:
            assert sample.shape == self.output_shape, (
                f"Output shape does not match expected. Expected {self.output_shape}, got {sample.shape}"
            )

    @torch.no_grad()
    def test_determinism(self, atol=1e-5, rtol=0):
        # The base version flattens the output, which assumes a single tensor.
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        first = model(**self.get_dummy_inputs(), return_dict=False)[0]
        second = model(**self.get_dummy_inputs(), return_dict=False)[0]

        assert_tensors_close(first, second, atol=atol, rtol=rtol, msg="Model outputs are not deterministic")

    @torch.no_grad()
    def test_outputs_equivalence(self, atol=1e-5, rtol=0):
        # The base version walks the dict output calling `.values()` at every level, which does not fit a
        # `sample` that is itself a list of tensors.
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        outputs_dict = model(**self.get_dummy_inputs())
        outputs_tuple = model(**self.get_dummy_inputs(), return_dict=False)

        assert_tensors_close(
            outputs_tuple[0], outputs_dict.sample, atol=atol, rtol=rtol, msg="Tuple and dict output are not equal"
        )

    @pytest.mark.skip(
        "`img_emb.proj.1` is a single 1280x1280 Linear — `MLPProj` hardcodes the CLIP hidden size, so it is 6.5 MB "
        "of this 9.8 MB test model and does not fit the per-GPU budget `get_balanced_memory` hands device 0. "
        "`device_map='auto'` therefore puts the whole model on one device and the map never spans both GPUs. "
        "`test_cpu_offload` covers split placement instead."
    )
    def test_model_parallelism(self, base_model_output, tmp_path, atol=1e-5, rtol=0):
        pass

    @torch.no_grad()
    def test_generation_pass_attends_the_cached_reference(self):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        reference_inputs = self.get_dummy_inputs()
        model(**reference_inputs)
        output = model(**self.get_dummy_generation_inputs(reference_inputs["kv_cache"]), return_dict=False)[0]

        expected_shape = (LATENT_CHANNELS, GENERATION_LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
        assert output[0].shape == expected_shape, f"Expected {expected_shape}, got {output[0].shape}"

        # Nothing but the cached reference differs between the two denoising passes, so a different driving
        # video has to move the prediction — that is the whole point of the in-context reference mechanism.
        other_reference_inputs = self.get_dummy_inputs()
        other_reference_inputs["hidden_states"] = [u.flip(1) for u in other_reference_inputs["hidden_states"]]
        model(**other_reference_inputs)
        other_output = model(
            **self.get_dummy_generation_inputs(other_reference_inputs["kv_cache"]), return_dict=False
        )[0]

        assert not torch.allclose(output[0], other_output[0]), "The denoising pass ignored the cached reference K/V."

    @torch.no_grad()
    def test_generation_pass_with_an_empty_cache_raises(self):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        with pytest.raises(RuntimeError, match="The KV cache is empty"):
            model(**self.get_dummy_generation_inputs(WanAnimate2KVCache(NUM_LAYERS)))

    @torch.no_grad()
    def test_unknown_kv_cache_mode_raises(self):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        inputs = self.get_dummy_inputs()
        inputs["kv_cache_mode"] = "reference"
        with pytest.raises(ValueError, match="`kv_cache_mode` must be either"):
            model(**inputs)

    @torch.no_grad()
    def test_seq_len_mismatch_raises(self):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        inputs = self.get_dummy_inputs()
        inputs["seq_len"] += 1
        with pytest.raises(ValueError, match="must hold exactly `seq_len`"):
            model(**inputs)

    @torch.no_grad()
    def test_unconditional_branch_skips_block_nine(self):
        # `is_uncondtion=True` drops the block at index 9, so it only bites on a model deep enough to have one.
        num_layers = 12
        torch.manual_seed(0)
        model = self.model_class(**{**self.get_init_dict(), "num_layers": num_layers})
        model.to(torch_device)
        model.eval()

        inputs = self.get_dummy_inputs()
        inputs["kv_cache"] = WanAnimate2KVCache(num_layers)
        uncond_output = model(**inputs, is_uncondtion=True, return_dict=False)[0]

        # Deleting that block has to reproduce the same output. Re-indexing the remaining blocks is harmless
        # here: under `kv_cache_mode="extract"` a block only writes its own cache slot and reads none.
        del model.blocks[9]
        inputs = self.get_dummy_inputs()
        inputs["kv_cache"] = WanAnimate2KVCache(num_layers)
        output = model(**inputs, return_dict=False)[0]

        assert_tensors_close(
            output, uncond_output, atol=0, rtol=0, msg="The unconditional branch skipped a different block."
        )


class TestWanAnimate2TransformerMemory(WanAnimate2TransformerTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for the Wan-Animate-2 transformer."""

    @pytest.mark.skip("The main input is a list of latents, so the mixin cannot shape the training loss from it.")
    def test_layerwise_casting_training(self):
        pass


class TestWanAnimate2TransformerTraining(WanAnimate2TransformerTesterConfig, TrainingTesterMixin):
    """Training tests for the Wan-Animate-2 transformer."""

    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set={"WanAnimate2Transformer3DModel"})

    @pytest.mark.skip(
        "`forward` calls `self._gradient_checkpointing_func(block, hidden_states, kv_cache=..., **block_kwargs)`, "
        "but the default wrapper installed by `ModelMixin.enable_gradient_checkpointing` is declared as "
        "`(module, *args)` and rejects keyword arguments. The checkpointed math itself matches the "
        "non-checkpointed pass; only the call raises `TypeError`."
    )
    def test_gradient_checkpointing_equivalence(self, loss_tolerance=1e-5, param_grad_tol=5e-5, skip=None):
        # The base version builds the loss with `torch.randn_like(out)`, which assumes a single output tensor.
        skip = skip or set()

        init_dict = self.get_init_dict()

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.to(torch_device)

        torch.manual_seed(0)
        model_2 = self.model_class(**init_dict)
        model_2.load_state_dict(model.state_dict())
        model_2.to(torch_device)
        model_2.enable_gradient_checkpointing()

        assert not model.is_gradient_checkpointing and model.training
        assert model_2.is_gradient_checkpointing and model_2.training

        out = model(**self.get_dummy_inputs(), return_dict=False)[0]
        labels = [torch.randn_like(sample) for sample in out]

        model.zero_grad()
        loss = torch.stack([(sample - label).mean() for sample, label in zip(out, labels)]).mean()
        loss.backward()

        out_2 = model_2(**self.get_dummy_inputs(), return_dict=False)[0]
        model_2.zero_grad()
        loss_2 = torch.stack([(sample - label).mean() for sample, label in zip(out_2, labels)]).mean()
        loss_2.backward()

        assert (loss - loss_2).abs() < loss_tolerance, (
            f"Loss difference {(loss - loss_2).abs()} exceeds tolerance {loss_tolerance}"
        )

        named_params_2 = dict(model_2.named_parameters())
        for name, param in model.named_parameters():
            if name in skip or param.grad is None:
                continue
            assert_tensors_close(
                param.grad.data,
                named_params_2[name].grad.data,
                atol=param_grad_tol,
                rtol=0,
                msg=f"Gradient mismatch for {name}",
            )

    @pytest.mark.skip("The output is a list of latents, so the mixin cannot shape the training loss from it.")
    def test_training(self):
        pass

    @pytest.mark.skip("The output is a list of latents, so the mixin cannot shape the training loss from it.")
    def test_training_with_ema(self):
        pass

    @pytest.mark.skip("The output is a list of latents, so the mixin cannot shape the training loss from it.")
    def test_mixed_precision_training(self):
        pass


class TestWanAnimate2TransformerAttention(WanAnimate2TransformerTesterConfig, AttentionTesterMixin):
    """Attention processor tests for the Wan-Animate-2 transformer."""


class TestWanAnimate2TransformerCompile(WanAnimate2TransformerTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for the Wan-Animate-2 transformer."""

    # `rope_apply` slices the rotary frequencies inside a Python loop over `grid_sizes.tolist()`, so dynamo
    # cannot guard the frame/height/width extents and `fullgraph=True` compilation fails. Only the
    # graph-break-tolerant paths (regional compilation under group offloading) are exercised here.
    _NO_FULLGRAPH = "`rope_apply` indexes the rotary frequencies by data-dependent grid sizes, which `fullgraph=True` cannot guard on."

    @pytest.mark.skip(_NO_FULLGRAPH)
    def test_torch_compile_recompilation_and_graph_break(self):
        pass

    @pytest.mark.skip(_NO_FULLGRAPH)
    def test_torch_compile_repeated_blocks(self, recompile_limit=1):
        pass

    @pytest.mark.skip(
        "`torch.export` rejects `WanAnimate2KVCache` as an input: it is neither a tensor nor a registered "
        "pytree container."
    )
    def test_compile_works_with_aot(self, tmp_path):
        pass
