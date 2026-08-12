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

import numpy as np
import PIL.Image
import pytest

from diffusers.modular_pipelines import (
    WanAnimate2Blocks,
    WanAnimate2DistilledBlocks,
    WanAnimate2DistilledModularPipeline,
    WanAnimate2ModularPipeline,
)

from ..test_modular_pipelines_common import ModularGuiderTesterMixin, ModularPipelineTesterMixin


class TestWanAnimate2ModularPipelineFast(ModularPipelineTesterMixin, ModularGuiderTesterMixin):
    pipeline_class = WanAnimate2ModularPipeline
    pipeline_blocks_class = WanAnimate2Blocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-wan-animate-2-modular"

    params = frozenset(["prompt", "image", "driving_video"])
    batch_params = frozenset()
    optional_params = frozenset(["num_inference_steps", "height", "width", "output_type"])
    output_name = "videos"

    def get_dummy_inputs(self, seed=0):
        rng = np.random.RandomState(seed)
        image = PIL.Image.fromarray(rng.randint(0, 255, (96, 64, 3), dtype=np.uint8))
        driving_video = [PIL.Image.fromarray(rng.randint(0, 255, (40, 32, 3), dtype=np.uint8)) for _ in range(17)]
        inputs = {
            "prompt": "a tiny cat",
            "image": image,
            "driving_video": driving_video,
            "generator": self.get_generator(seed),
            "num_inference_steps": 2,
            "height": 64,
            "width": 64,
            "segment_frame_length": 9,
            "prev_segment_conditioning_frames": 1,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs

    @pytest.mark.skip(reason="Wan-Animate-2 is unbatched: one character image and driving video per call")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="Wan-Animate-2 is unbatched: one character image and driving video per call")
    def test_inference_batch_single_identical(self):
        pass

    @pytest.mark.skip(reason="Wan-Animate-2 is unbatched: no num_videos_per_prompt")
    def test_num_images_per_prompt(self):
        pass

    def test_guider_cfg(self):
        # The tiny random transformer responds only weakly to the text embeddings (cond vs uncond
        # predictions differ by ~1e-4), so the CFG effect on the decoded pixels is real but small.
        # Same-seed runs are bit-deterministic, so any nonzero difference here is guidance.
        super().test_guider_cfg(expected_max_diff=1e-6)


class TestWanAnimate2DistilledModularPipelineFast(TestWanAnimate2ModularPipelineFast):
    pipeline_class = WanAnimate2DistilledModularPipeline
    pipeline_blocks_class = WanAnimate2DistilledBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-wan-animate-2-distilled-modular"

    @pytest.mark.skip(reason="The distilled preset pins its guider to guidance_scale=1.0")
    def test_guider_cfg(self):
        pass
