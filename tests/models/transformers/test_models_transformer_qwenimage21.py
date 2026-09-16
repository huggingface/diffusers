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
from torch.nn.attention.flex_attention import create_mask

from diffusers import QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import build_qwenimage21_block_causal_mask
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class QwenImage21TransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return QwenImage21Transformer2DModel

    @property
    def output_shape(self) -> tuple[int, int]:
        return (8, 4)

    @property
    def input_shape(self) -> tuple[int, int]:
        return (4, 4)

    @property
    def model_split_percents(self) -> list:
        return [0.7, 0.6, 0.6]

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int]]:
        # `attention_head_dim` is 16 because flex_attention requires a head dim of at least 16, and
        # `axes_dims_rope` must sum to it.
        return {
            "patch_size": 1,
            "in_channels": 4,
            "out_channels": 4,
            "num_layers": 2,
            "attention_head_dim": 16,
            "num_attention_heads": 2,
            "context_in_dim": 8,
            "mlp_ratio": 2,
            "axes_dims_rope": (4, 6, 6),
        }

    def get_dummy_inputs(self, batch_size: int = 1, device=torch_device) -> dict[str, torch.Tensor]:
        text_len, target_height, target_width = 4, 2, 2
        target_tokens = target_height * target_width

        hidden_states = randn_tensor((batch_size, target_tokens, 4), generator=self.generator, device=device)
        encoder_hidden_states = randn_tensor((batch_size, text_len, 8), generator=self.generator, device=device)
        encoder_hidden_states_mask = torch.ones((batch_size, text_len), device=device, dtype=torch.long)
        # One vision slot per 2x2 group of target latents, appended after the text tokens.
        img_mask = torch.zeros((batch_size, text_len + target_tokens // 4), device=device, dtype=torch.bool)
        img_mask[:, text_len:] = True

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": torch.tensor([1.0], device=device).expand(batch_size),
            "img_shapes": [[(1, target_height, target_width)]] * batch_size,
            "img_mask": img_mask,
        }


class TestQwenImage21Transformer(QwenImage21TransformerTesterConfig, ModelTesterMixin):
    @pytest.mark.skip(
        reason="The block-causal BlockMask's mask_mod closes over per-token id tensors bound to one device. "
        "`BlockMask.to()` relocates the mask's own index tensors but not those captures, so sharding a single "
        "forward across devices mixes them. Same limitation as AnyFlowFARTransformer3DModel."
    )
    def test_model_parallelism(self):
        pass

    def test_kv_cache_matches_full_forward(self):
        """
        Decoding from a cache prefilled at a different timestep must match a full forward. This only holds because
        `causal_condition` modulates text and condition-image tokens from t=0, making their activations independent of
        the denoising step.
        """
        inputs = self.get_dummy_inputs()
        target_tokens = inputs["hidden_states"].shape[1]

        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

        torch.manual_seed(0)
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device).eval()
        kv_cache = QwenImage21KVCache(init_dict["num_layers"])

        prefill_inputs = dict(inputs, timestep=torch.tensor([0.9], device=torch_device))
        decode_inputs = dict(inputs, timestep=torch.tensor([0.4], device=torch_device))
        with torch.no_grad():
            model(**prefill_inputs, kv_cache=kv_cache, kv_cache_mode="extract", return_dict=False)
            decoded = model(**decode_inputs, kv_cache=kv_cache, kv_cache_mode="cached", return_dict=False)[0]
            reference = model(**decode_inputs, return_dict=False)[0]

        assert decoded.shape[1] == target_tokens
        torch.testing.assert_close(decoded, reference[:, -target_tokens:], atol=2e-5, rtol=2e-5)

    def test_kv_cache_owns_its_memory(self):
        """
        The cached prefix must own its storage. At batch size 1 the prefix slice already counts as contiguous, so
        storing `key[:, :prefix].contiguous()` hands the cache a view that pins the whole prefill K/V — 8 GiB at
        2048² — for every step of the denoising loop.
        """
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device).eval()
        kv_cache = QwenImage21KVCache(init_dict["num_layers"])
        with torch.no_grad():
            model(**self.get_dummy_inputs(batch_size=1), kv_cache=kv_cache, kv_cache_mode="extract")

        for index in range(init_dict["num_layers"]):
            for cached in kv_cache.get_layer(index).get():
                assert cached.untyped_storage().nbytes() == cached.numel() * cached.element_size(), (
                    f"layer {index} cached a view into the full prefill K/V instead of a copy"
                )

    def test_kv_cache_requires_causal_condition(self):
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

        init_dict = dict(self.get_init_dict(), causal_condition=False)
        model = self.model_class(**init_dict).to(torch_device).eval()
        kv_cache = QwenImage21KVCache(init_dict["num_layers"])
        with pytest.raises(ValueError, match="causal_condition"):
            model(**self.get_dummy_inputs(), kv_cache=kv_cache, kv_cache_mode="extract")

    def test_non_flex_backend_rejected_when_causal(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        model.set_attention_backend("native")
        with pytest.raises(ValueError, match="flex"):
            model(**self.get_dummy_inputs())


class TestQwenImage21BlockCausalMask:
    """
    The mask is `(q_idx >= kv_idx) or same_image_block`: causal over the joint sequence, bidirectional inside each
    image block. Verified elementwise, since a wrong mask degrades quality silently rather than raising.
    """

    def _layout(self):
        # text, two *adjacent* condition images, text, target image.
        img_shapes = [(1, 2, 2), (1, 2, 2), (1, 4, 4)]
        vlm_mask = torch.tensor([False] * 3 + [True, True] + [False, False] + [True] * 4)
        image_pad_mask = torch.repeat_interleave(vlm_mask, torch.where(vlm_mask, 4, 1))
        return img_shapes, image_pad_mask

    def _dense_mask(self, image_ids, key_valid=None, batch_size=1):
        block_mask = build_qwenimage21_block_causal_mask(image_ids, key_valid, batch_size, torch.device("cpu"))
        padded = block_mask.shape[-1]
        seq_len = image_ids.shape[0]
        dense = create_mask(block_mask.mask_mod, batch_size, 1, padded, padded, device=torch.device("cpu"))
        return dense[:, 0, :seq_len, :seq_len]

    def test_block_ids_keep_adjacent_condition_images_separate(self):
        img_shapes, image_pad_mask = self._layout()
        image_ids, target_token_mask = QwenImage21Transformer2DModel.build_token_metadata(image_pad_mask, img_shapes)

        assert int(image_ids.max()) + 1 == 3, "two adjacent condition images must not merge into one block"
        assert (image_ids[~image_pad_mask] == -1).all(), "text tokens must carry no block id"
        target_len = img_shapes[-1][1] * img_shapes[-1][2]
        assert int(target_token_mask.sum()) == target_len
        assert target_token_mask[-target_len:].all()

    def test_block_ids_reject_inconsistent_shapes(self):
        img_shapes, image_pad_mask = self._layout()
        with pytest.raises(ValueError, match="image tokens"):
            QwenImage21Transformer2DModel.build_token_metadata(image_pad_mask, img_shapes[:-1])

    def test_mask_is_causal_across_text_and_full_within_images(self):
        img_shapes, image_pad_mask = self._layout()
        image_ids, _ = QwenImage21Transformer2DModel.build_token_metadata(image_pad_mask, img_shapes)
        mask = self._dense_mask(image_ids)[0]

        query = torch.arange(mask.shape[0]).view(-1, 1)
        key = torch.arange(mask.shape[0]).view(1, -1)
        same_block = (image_ids.view(-1, 1) == image_ids.view(1, -1)) & (image_ids.view(-1, 1) >= 0)
        expected = (query >= key) | same_block
        torch.testing.assert_close(mask, expected)

        text = (~image_pad_mask).nonzero(as_tuple=True)[0]
        assert (mask[text][:, text] == (text.view(-1, 1) >= text.view(1, -1))).all()

        first = (image_ids == 0).nonzero(as_tuple=True)[0]
        second = (image_ids == 1).nonzero(as_tuple=True)[0]
        assert mask[first][:, first].all() and mask[second][:, second].all()
        assert not mask[first][:, second].any(), "an image block must not see a later block"
        assert mask[second][:, first].all(), "a later block must see an earlier one"

    def test_padded_text_is_never_attended(self):
        img_shapes, image_pad_mask = self._layout()
        image_ids, _ = QwenImage21Transformer2DModel.build_token_metadata(image_pad_mask, img_shapes)

        key_valid = torch.ones(1, image_ids.shape[0], dtype=torch.bool)
        key_valid[0, 1] = False
        mask = self._dense_mask(image_ids, key_valid)[0]

        assert not mask[:, 1].any(), "a padded position must never be attended as a key"
        assert mask[1].any(), "a padded position must keep at least one key so its row is not fully masked"


class TestQwenImage21TransformerMemory(QwenImage21TransformerTesterConfig, MemoryTesterMixin):
    pass


class TestQwenImage21TransformerTraining(QwenImage21TransformerTesterConfig, TrainingTesterMixin):
    pass


class TestQwenImage21TransformerAttention(QwenImage21TransformerTesterConfig, AttentionTesterMixin):
    pass
