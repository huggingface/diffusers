import random
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
    VisualClozeGenerationPipeline,
)
from diffusers.utils import logging

from ...testing_utils import (
    CaptureLogger,
    enable_full_determinism,
    floats_tensor,
    require_accelerator,
    torch_device,
)
from ..test_pipelines_common import PipelineTesterMixin, to_np


enable_full_determinism()


class VisualClozeGenerationPipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = VisualClozeGenerationPipeline
    params = frozenset(
        [
            "task_prompt",
            "content_prompt",
            "guidance_scale",
            "prompt_embeds",
            "pooled_prompt_embeds",
        ]
    )
    batch_params = frozenset(["task_prompt", "content_prompt", "image"])
    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True

    supports_dduf = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=12,
            out_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=6,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[2, 2, 2],
        )
        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )

        torch.manual_seed(0)
        text_encoder = CLIPTextModel(clip_text_encoder_config)

        torch.manual_seed(0)
        text_encoder_2 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=1,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "transformer": transformer,
            "vae": vae,
            "resolution": 32,
        }

    def get_dummy_inputs(self, device, seed=0):
        # Create example images to simulate the input format required by VisualCloze
        context_image = [
            Image.fromarray(floats_tensor((32, 32, 3), rng=random.Random(seed), scale=255).numpy().astype(np.uint8))
            for _ in range(2)
        ]
        query_image = [
            Image.fromarray(
                floats_tensor((32, 32, 3), rng=random.Random(seed + 1), scale=255).numpy().astype(np.uint8)
            ),
            None,
        ]

        # Create an image list that conforms to the VisualCloze input format
        image = [
            context_image,  # In-Context example
            query_image,  # Query image
        ]

        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        inputs = {
            "task_prompt": "Each row outlines a logical process, starting from [IMAGE1] gray-based depth map with detailed object contours, to achieve [IMAGE2] an image with flawless clarity.",
            "content_prompt": "A beautiful landscape with mountains and a lake",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "max_sequence_length": 77,
            "output_type": "np",
        }
        return inputs

    def test_visualcloze_different_prompts(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["task_prompt"] = "A different task to perform."
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = np.abs(output_same_prompt - output_different_prompts).max()

        # Outputs should be different
        assert max_diff > 1e-6

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=1e-3)

    def test_different_task_prompts(self, expected_min_diff=1e-1):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        output_original = pipe(**inputs).images[0]

        inputs["task_prompt"] = "A different task description for image generation"
        output_different_task = pipe(**inputs).images[0]

        # Different task prompts should produce different outputs
        max_diff = np.abs(output_original - output_different_task).max()
        assert max_diff > expected_min_diff

    def test_save_load_local(self, expected_max_difference=5e-4):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]

        logger = logging.get_logger("diffusers.pipelines.pipeline_utils")
        logger.setLevel(diffusers.logging.INFO)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir, safe_serialization=False)

            with CaptureLogger(logger) as cap_logger:
                # NOTE: Resolution must be set to 32 for loading otherwise will lead to OOM on CI hardware
                # This attribute is not serialized in the config of the pipeline
                pipe_loaded = self.pipeline_class.from_pretrained(tmpdir, resolution=32)

            for component in pipe_loaded.components.values():
                if hasattr(component, "set_default_attn_processor"):
                    component.set_default_attn_processor()

            for name in pipe_loaded.components.keys():
                if name not in pipe_loaded._optional_components:
                    assert name in str(cap_logger)

            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(max_diff, expected_max_difference)

    def test_save_load_optional_components(self, expected_max_difference=1e-4):
        if not hasattr(self.pipeline_class, "_optional_components"):
            return
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # set all optional components to None
        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)

        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        torch.manual_seed(0)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir, safe_serialization=False)
            # NOTE: Resolution must be set to 32 for loading otherwise will lead to OOM on CI hardware
            # This attribute is not serialized in the config of the pipeline
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir, resolution=32)
            for component in pipe_loaded.components.values():
                if hasattr(component, "set_default_attn_processor"):
                    component.set_default_attn_processor()
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in pipe._optional_components:
            self.assertTrue(
                getattr(pipe_loaded, optional_component) is None,
                f"`{optional_component}` did not stay set to None after loading.",
            )

        inputs = self.get_dummy_inputs(generator_device)
        torch.manual_seed(0)
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(max_diff, expected_max_difference)

    @unittest.skipIf(torch_device not in ["cuda", "xpu"], reason="float16 requires CUDA or XPU")
    @require_accelerator
    def test_save_load_float16(self, expected_max_diff=1e-2):
        components = self.get_dummy_components()
        for name, module in components.items():
            if hasattr(module, "half"):
                components[name] = module.to(torch_device).half()

        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            # NOTE: Resolution must be set to 32 for loading otherwise will lead to OOM on CI hardware
            # This attribute is not serialized in the config of the pipeline
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir, torch_dtype=torch.float16, resolution=32)
            for component in pipe_loaded.components.values():
                if hasattr(component, "set_default_attn_processor"):
                    component.set_default_attn_processor()
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for name, component in pipe_loaded.components.items():
            if hasattr(component, "dtype"):
                self.assertTrue(
                    component.dtype == torch.float16,
                    f"`{name}.dtype` switched from `float16` to {component.dtype} after loading.",
                )

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs)[0]
        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(
            max_diff, expected_max_diff, "The output of the fp16 pipeline changed after saving and loading."
        )

    @unittest.skip("Skipped due to missing layout_prompt. Needs further investigation.")
    def test_encode_prompt_works_in_isolation(self, extra_required_param_value_dict=None, atol=0.0001, rtol=0.0001):
        pass
