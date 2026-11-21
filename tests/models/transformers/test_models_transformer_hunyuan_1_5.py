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

import torch
import torch.nn.functional as F

from diffusers import HunyuanVideo15Transformer3DModel
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers import transformer_hunyuan_video15
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class HunyuanVideo15TransformerTesterConfig(BaseModelTesterConfig):
    text_embed_dim = 16
    text_embed_2_dim = 8
    image_embed_dim = 12

    @property
    def model_class(self):
        return HunyuanVideo15Transformer3DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def model_split_percents(self) -> list:
        return [0.99, 0.99, 0.99]

    @property
    def output_shape(self) -> tuple:
        return (4, 1, 8, 8)

    @property
    def input_shape(self) -> tuple:
        return (4, 1, 8, 8)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": 4,
            "out_channels": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "num_layers": 2,
            "num_refiner_layers": 1,
            "mlp_ratio": 2.0,
            "patch_size": 1,
            "patch_size_t": 1,
            "text_embed_dim": self.text_embed_dim,
            "text_embed_2_dim": self.text_embed_2_dim,
            "image_embed_dim": self.image_embed_dim,
            "rope_axes_dim": (2, 2, 4),
            "target_size": 16,
            "task_type": "t2v",
        }

    def get_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        num_channels = 4
        num_frames = 1
        height = 8
        width = 8
        sequence_length = 6
        sequence_length_2 = 4
        image_sequence_length = 3

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, num_frames, height, width), generator=self.generator, device=torch_device
            ),
            "timestep": torch.tensor([1.0]).to(torch_device).expand(batch_size),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, self.text_embed_dim), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states_2": randn_tensor(
                (batch_size, sequence_length_2, self.text_embed_2_dim), generator=self.generator, device=torch_device
            ),
            "encoder_attention_mask": torch.ones((batch_size, sequence_length), device=torch_device),
            "encoder_attention_mask_2": torch.ones((batch_size, sequence_length_2), device=torch_device),
            "image_embeds": torch.zeros(
                (batch_size, image_sequence_length, self.image_embed_dim), device=torch_device
            ),
        }


class TestHunyuanVideo15Transformer(HunyuanVideo15TransformerTesterConfig, ModelTesterMixin):
    pass


class TestHunyuanVideo15TransformerAttentionMask(HunyuanVideo15TransformerTesterConfig):
    def test_key_only_attention_mask_matches_dense_mask(self):
        """The mask the processor passes gives the same video tokens, the same valid encoder rows and the same output,
        bit for bit, as a reference with the dense (batch, 1, seq_len, seq_len) mask that also masks the padded
        queries (the mask before the key-only change)."""
        encoder_masks = []

        def capture_encoder_mask(module, args, kwargs):
            encoder_masks.append(kwargs["attention_mask"])

        def dispatch_with_dense_mask(query, key, value, attn_mask=None, **kwargs):
            # the mask HunyuanVideo15AttnProcessor2_0 built before (padded tokens masked as keys and as queries),
            # rebuilt from the block's encoder mask; the mask the processor passes is not used
            batch_size, seq_len = query.shape[:2]
            mask = F.pad(encoder_masks[-1], (seq_len - encoder_masks[-1].shape[1], 0), value=True).bool()
            mask = mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            return dispatch_attention_fn(query, key, value, attn_mask=mask & mask.transpose(2, 3), **kwargs)

        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        for block in model.transformer_blocks:
            block.attn.register_forward_pre_hook(capture_encoder_mask, with_kwargs=True)

        def run(inputs):
            block_outputs = []
            hooks = [
                block.register_forward_hook(lambda module, args, output: block_outputs.append(output))
                for block in model.transformer_blocks
            ]
            with torch.no_grad():
                sample = model(**inputs).sample
            for hook in hooks:
                hook.remove()
            return sample, block_outputs

        inputs = self.get_dummy_inputs(batch_size=2)
        # padded prompts, different per sample; the second sample has no valid ByT5 token
        qwen_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.float32, device=torch_device)
        byt5_mask = torch.tensor([[1, 1, 1, 0], [0, 0, 0, 0]], dtype=torch.float32, device=torch_device)
        inputs["encoder_attention_mask"], inputs["encoder_attention_mask_2"] = qwen_mask, byt5_mask
        t2v_image_embeds = inputs["image_embeds"]
        i2v_image_embeds = randn_tensor(t2v_image_embeds.shape, generator=self.generator, device=torch_device)
        # all-zero image_embeds: t2v, the image-embedding tokens are padded too; otherwise i2v, they are valid.
        # Valid encoder tokens per sample = image-embedding + ByT5 + Qwen
        for image_embeds, num_valid in ((t2v_image_embeds, [8, 3]), (i2v_image_embeds, [11, 6])):
            inputs["image_embeds"] = image_embeds
            encoder_masks.clear()
            out, out_blocks = run(inputs)
            encoder_mask = encoder_masks[-1].bool()
            assert encoder_mask.sum(dim=1).tolist() == num_valid
            with mock.patch.object(transformer_hunyuan_video15, "dispatch_attention_fn", dispatch_with_dense_mask):
                ref, ref_blocks = run(inputs)

            assert torch.isfinite(out).all() and torch.isfinite(ref).all()
            assert torch.equal(out, ref)
            assert len(out_blocks) == len(ref_blocks) == len(model.transformer_blocks)
            for (hidden, encoder), (ref_hidden, ref_encoder) in zip(out_blocks, ref_blocks):
                assert torch.equal(hidden, ref_hidden)
                # valid encoder tokens come first; the padded ones are never read as unmasked keys and are dropped
                for i, n in enumerate(num_valid):
                    assert encoder_mask[i, :n].all()
                    assert torch.equal(encoder[i, :n], ref_encoder[i, :n])


class TestHunyuanVideo15TransformerTraining(HunyuanVideo15TransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"HunyuanVideo15Transformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
