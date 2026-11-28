# Copyright 2026 The VAST-AI Team. All rights reserved.
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

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
    DINOv3ViTConfig,
    DINOv3ViTModel,
    UMT5Config,
    UMT5EncoderModel,
)

from diffusers import (
    AutoencoderKLWan,
    EdenLiveLightTAEDecoder,
    EdenLivePipeline,
    EdenLiveTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
)

from ...testing_utils import torch_device
from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_PARAMS, VIDEO_TO_VIDEO_BATCH_PARAMS
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


class EdenLivePipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = EdenLivePipeline
    profile = "5b"
    required_input_params_in_call_signature = TEXT_GUIDED_IMAGE_VARIATION_PARAMS | {"video"}
    optional_input_params = frozenset(
        ["num_frames", "num_inference_steps", "guidance_scale", "generator", "latents", "output_type", "return_dict"]
    )
    batch_input_params = VIDEO_TO_VIDEO_BATCH_PARAMS | {"image"}
    output_shape = (9, 3, 16, 16)

    def get_dummy_components(self):
        torch.manual_seed(0)
        vae = AutoencoderKLWan(
            base_dim=4,
            z_dim=4,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
        )
        torch.manual_seed(0)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = UMT5EncoderModel(
            UMT5Config(
                vocab_size=len(tokenizer), d_model=32, d_ff=32, d_kv=8, num_layers=1, num_heads=4, dropout_rate=0.0
            )
        ).eval()
        torch.manual_seed(0)
        proxy_encoder = DINOv3ViTModel(
            DINOv3ViTConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                patch_size=16,
                image_size=32,
                num_register_tokens=4,
            )
        ).eval()
        torch.manual_seed(0)
        transformer = EdenLiveTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=4,
            attention_head_dim=12,
            in_channels=4 if self.profile == "5b" else 12,
            out_channels=4,
            proxy_input_channels=32,
            proxy_channels=8,
            condition_rmsnorm=self.profile == "14b",
            text_dim=32,
            freq_dim=16,
            ffn_dim=64,
            num_layers=1,
            image_dim=32 if self.profile == "14b" else None,
            added_kv_proj_dim=48 if self.profile == "14b" else None,
            rope_max_seq_len=64,
        )
        if self.profile == "14b":
            torch.manual_seed(0)
            image_encoder = CLIPVisionModel(
                CLIPVisionConfig(
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    image_size=16,
                    patch_size=8,
                    projection_dim=32,
                )
            ).eval()
            image_processor = CLIPImageProcessor(size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16})
            scheduler = FlowMatchEulerDiscreteScheduler(shift=3)
        else:
            image_encoder, image_processor = None, None
            scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5)
        return {
            "vae": vae,
            "decoder": None,
            "proxy_image_height": 32,
            "proxy_image_width": 32,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "transformer": transformer,
            "scheduler": scheduler,
            "proxy_encoder": proxy_encoder,
            "image_encoder": image_encoder,
            "image_processor": image_processor,
        }

    def get_dummy_inputs(self):
        generator = np.random.default_rng(0)
        return {
            "prompt": "a character walks through snow",
            "negative_prompt": "blurry",
            "image": Image.fromarray(generator.integers(0, 256, (16, 16, 3), dtype=np.uint8)),
            "video": generator.random((9, 16, 16, 3), dtype=np.float32),
            "height": 16,
            "width": 16,
            "num_frames": 9,
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "generator": self.get_generator(0),
            "max_sequence_length": 16,
            "output_type": "pt",
        }


class EdenLive14BRequiredComponentsTesterMixin:
    def test_save_load_optional_components(self, tmp_path):
        pipe = self.get_pipeline()
        pipe.register_modules(image_encoder=None, image_processor=None)
        pipe.save_pretrained(tmp_path)
        restored = self.pipeline_class.from_pretrained(tmp_path).to(torch_device)
        assert restored.image_encoder is None and restored.image_processor is None
        with pytest.raises(ValueError, match="CLIP image encoder"):
            restored(**self.get_dummy_inputs())


class EdenLiveOptionalDecoderTesterMixin:
    def test_lighttae_save_load_and_short_tail(self, tmp_path):
        pipe = self.get_pipeline().to(torch_device)
        torch.manual_seed(0)
        pipe.register_modules(
            decoder=EdenLiveLightTAEDecoder(latent_channels=4, block_out_channels=(4, 4, 4, 4)).to(torch_device)
        )
        inputs = self.get_dummy_inputs()
        inputs["video"] = np.concatenate([inputs["video"], inputs["video"][:4]])
        inputs["num_frames"] = 13
        expected = pipe(**inputs).frames
        assert expected.shape == (1, 13, 3, 16, 16)
        pipe.save_pretrained(tmp_path)
        restored = self.pipeline_class.from_pretrained(tmp_path).to(torch_device)
        assert isinstance(restored.decoder, EdenLiveLightTAEDecoder)
        inputs["generator"] = self.get_generator(0)
        torch.testing.assert_close(restored(**inputs).frames, expected, atol=1e-5, rtol=0)


class TestEdenLivePipeline(EdenLiveOptionalDecoderTesterMixin, EdenLivePipelineTesterConfig, PipelineTesterMixin):
    def test_inference_batch_single_identical(self, monkeypatch):
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
        super().test_inference_batch_single_identical()

    @pytest.mark.parametrize("num_frames", [0, 4, 8, 13])
    def test_invalid_frame_count(self, num_frames):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = num_frames
        with pytest.raises(ValueError, match="frames|frame"):
            pipe(**inputs)

    def test_automatic_frame_prefix(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = None
        inputs["video"] = np.concatenate([inputs["video"], inputs["video"][:2]])
        output = pipe(**inputs).frames
        assert output.shape == (1, *self.output_shape)

    @pytest.mark.parametrize("output_type", ["np", "pt", "pil", "latent"])
    def test_output_types(self, output_type):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["output_type"] = output_type
        output = pipe(**inputs).frames
        if output_type == "np":
            assert output.shape == (1, 9, 16, 16, 3)
        elif output_type == "pt":
            assert output.shape == (1, *self.output_shape)
        elif output_type == "latent":
            assert output.shape == (1, 4, 3, 2, 2)
        else:
            assert len(output) == 1 and len(output[0]) == 9
            assert isinstance(output[0][0], Image.Image)

    @pytest.mark.parametrize("condition", ["image", "video", "prompt"])
    def test_conditions_change_generation(self, condition):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        expected = pipe(**inputs).frames
        inputs = self.get_dummy_inputs()
        if condition == "prompt":
            inputs[condition] = "a racing car on a sunny road"
        elif condition == "image":
            inputs[condition] = Image.new("RGB", (16, 16), "white")
        else:
            inputs[condition] = 1 - inputs[condition]
        actual = pipe(**inputs).frames
        assert (actual - expected).abs().max() > 1e-6


class TestEdenLive14BPipeline(EdenLive14BRequiredComponentsTesterMixin, TestEdenLivePipeline):
    profile = "14b"


class TestEdenLivePipelineMemory(EdenLivePipelineTesterConfig, MemoryTesterMixin):
    pass


class TestEdenLive14BPipelineMemory(EdenLivePipelineTesterConfig, MemoryTesterMixin):
    profile = "14b"
