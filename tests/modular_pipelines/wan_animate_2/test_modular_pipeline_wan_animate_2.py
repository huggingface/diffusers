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

from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularGuiderTesterMixin,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


# Every component with its class, every input — optional ones with their defaults — and the config values worth
# watching: the guider scale is what tells the two presets apart.
_COMPONENTS = {
    "text_encoder": "UMT5EncoderModel",
    "tokenizer": "AutoTokenizer",
    "image_processor": "WanAnimate2VideoProcessor",
    "image_encoder": "CLIPVisionModel",
    "video_processor": "WanAnimate2VideoProcessor",
    "vae": "AutoencoderKLWan",
    "transformer": "WanAnimate2Transformer3DModel",
    "scheduler": "SchedulerMixin",
    "guider": "ClassifierFreeGuidance",
}
_INPUTS = {
    "negative_prompt": None,
    "prompt_ref": "人物动作的参考视频",
    "max_sequence_length": 512,
    "height": 800,
    "width": 640,
    "driving_video_fps": None,
    "fps": 24,
    "segment_frame_length": 81,
    "prev_segment_conditioning_frames": 1,
    "generator": None,
    "output_type": "np",
}
WAN_ANIMATE_2_DEFAULTS = {
    None: {
        "components": _COMPONENTS,
        "configs": {},
        "required_inputs": ["prompt", "image", "driving_video"],
        "inputs": {**_INPUTS, "num_inference_steps": 40},
        "component_configs": {"guider": {"guidance_scale": 3.0}},
    }
}
WAN_ANIMATE_2_DISTILLED_DEFAULTS = {
    None: {
        "components": _COMPONENTS,
        "configs": {},
        "required_inputs": ["prompt", "image", "driving_video"],
        "inputs": {**_INPUTS, "num_inference_steps": 10},
        "component_configs": {"guider": {"guidance_scale": 1.0}},
    }
}


class WanAnimate2ModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = WanAnimate2ModularPipeline
    pipeline_blocks_class = WanAnimate2Blocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-wan-animate-2-modular"

    params = frozenset(["prompt", "image", "driving_video"])
    batch_params = frozenset()
    optional_params = frozenset(["num_inference_steps", "height", "width", "output_type"])
    output_name = "videos"
    expected_workflow_defaults = WAN_ANIMATE_2_DEFAULTS

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


class WanAnimate2ModularPipelineFastTesterMixin(ModularPipelineTesterMixin):
    """`ModularPipelineTesterMixin` minus the tests that assume a batchable pipeline; shared by both presets."""

    @pytest.mark.skip(reason="Wan-Animate-2 is unbatched: one character image and driving video per call")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="Wan-Animate-2 is unbatched: one character image and driving video per call")
    def test_inference_batch_single_identical(self):
        pass

    @pytest.mark.skip(reason="Wan-Animate-2 is unbatched: no num_videos_per_prompt")
    def test_num_images_per_prompt(self):
        pass


class TestWanAnimate2ModularPipelineFast(
    WanAnimate2ModularPipelineTesterConfig, WanAnimate2ModularPipelineFastTesterMixin
):
    pass


class TestWanAnimate2ModularPipelineLoading(WanAnimate2ModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestWanAnimate2ModularPipelineWorkflow(WanAnimate2ModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestWanAnimate2ModularPipelineMemory(WanAnimate2ModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass


class TestWanAnimate2ModularPipelineGuider(WanAnimate2ModularPipelineTesterConfig, ModularGuiderTesterMixin):
    def test_guider_cfg(self):
        # The tiny random transformer responds only weakly to the text embeddings (cond vs uncond
        # predictions differ by ~1e-4), so the CFG effect on the decoded pixels is real but small.
        # Same-seed runs are bit-deterministic, so any nonzero difference here is guidance.
        super().test_guider_cfg(expected_max_diff=1e-6)


class WanAnimate2DistilledModularPipelineTesterConfig(WanAnimate2ModularPipelineTesterConfig):
    pipeline_class = WanAnimate2DistilledModularPipeline
    pipeline_blocks_class = WanAnimate2DistilledBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-wan-animate-2-distilled-modular"
    expected_workflow_defaults = WAN_ANIMATE_2_DISTILLED_DEFAULTS


# No guider test class for the distilled preset: it pins its guider to guidance_scale=1.0.
class TestWanAnimate2DistilledModularPipelineFast(
    WanAnimate2DistilledModularPipelineTesterConfig, WanAnimate2ModularPipelineFastTesterMixin
):
    pass


class TestWanAnimate2DistilledModularPipelineLoading(
    WanAnimate2DistilledModularPipelineTesterConfig, ModularLoadingTesterMixin
):
    pass


class TestWanAnimate2DistilledModularPipelineWorkflow(
    WanAnimate2DistilledModularPipelineTesterConfig, ModularWorkflowTesterMixin
):
    pass


class TestWanAnimate2DistilledModularPipelineMemory(
    WanAnimate2DistilledModularPipelineTesterConfig, ModularMemoryTesterMixin
):
    pass
