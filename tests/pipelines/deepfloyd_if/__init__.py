import tempfile

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import AttnAddedKVProcessor
from diffusers.pipelines.deepfloyd_if import IFWatermarker

from ...testing_utils import torch_device
from ..test_pipelines_common import to_np


# WARN: the hf-internal-testing/tiny-random-t5 text encoder has some non-determinism in the `save_load` tests.


class IFPipelineTesterMixin:
    def _get_dummy_components(self):
        torch.manual_seed(0)
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            sample_size=32,
            layers_per_block=1,
            block_out_channels=[32, 64],
            down_block_types=[
                "ResnetDownsampleBlock2D",
                "SimpleCrossAttnDownBlock2D",
            ],
            mid_block_type="UNetMidBlock2DSimpleCrossAttn",
            up_block_types=["SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"],
            in_channels=3,
            out_channels=6,
            cross_attention_dim=32,
            encoder_hid_dim=32,
            attention_head_dim=8,
            addition_embed_type="text",
            addition_embed_type_num_heads=2,
            cross_attention_norm="group_norm",
            resnet_time_scale_shift="scale_shift",
            act_fn="gelu",
        )
        unet.set_attn_processor(AttnAddedKVProcessor())  # For reproducibility tests

        torch.manual_seed(0)
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            beta_start=0.0001,
            beta_end=0.02,
            thresholding=True,
            dynamic_thresholding_ratio=0.95,
            sample_max_value=1.0,
            prediction_type="epsilon",
            variance_type="learned_range",
        )

        torch.manual_seed(0)
        watermarker = IFWatermarker()

        return {
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "unet": unet,
            "scheduler": scheduler,
            "watermarker": watermarker,
            "safety_checker": None,
            "feature_extractor": None,
        }

    def _get_superresolution_dummy_components(self):
        torch.manual_seed(0)
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            sample_size=32,
            layers_per_block=[1, 2],
            block_out_channels=[32, 64],
            down_block_types=[
                "ResnetDownsampleBlock2D",
                "SimpleCrossAttnDownBlock2D",
            ],
            mid_block_type="UNetMidBlock2DSimpleCrossAttn",
            up_block_types=["SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"],
            in_channels=6,
            out_channels=6,
            cross_attention_dim=32,
            encoder_hid_dim=32,
            attention_head_dim=8,
            addition_embed_type="text",
            addition_embed_type_num_heads=2,
            cross_attention_norm="group_norm",
            resnet_time_scale_shift="scale_shift",
            act_fn="gelu",
            class_embed_type="timestep",
            mid_block_scale_factor=1.414,
            time_embedding_act_fn="gelu",
            time_embedding_dim=32,
        )
        unet.set_attn_processor(AttnAddedKVProcessor())  # For reproducibility tests

        torch.manual_seed(0)
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            beta_start=0.0001,
            beta_end=0.02,
            thresholding=True,
            dynamic_thresholding_ratio=0.95,
            sample_max_value=1.0,
            prediction_type="epsilon",
            variance_type="learned_range",
        )

        torch.manual_seed(0)
        image_noising_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            beta_start=0.0001,
            beta_end=0.02,
        )

        torch.manual_seed(0)
        watermarker = IFWatermarker()

        return {
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "unet": unet,
            "scheduler": scheduler,
            "image_noising_scheduler": image_noising_scheduler,
            "watermarker": watermarker,
            "safety_checker": None,
            "feature_extractor": None,
        }

    # this test is modified from the base class because if pipelines set the text encoder
    # as optional with the intention that the user is allowed to encode the prompt once
    # and then pass the embeddings directly to the pipeline. The base class test uses
    # the unmodified arguments from `self.get_dummy_inputs` which will pass the unencoded
    # prompt to the pipeline when the text encoder is set to None, throwing an error.
    # So we make the test reflect the intended usage of setting the text encoder to None.
    def _test_save_load_optional_components(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)

        prompt = inputs["prompt"]
        generator = inputs["generator"]
        num_inference_steps = inputs["num_inference_steps"]
        output_type = inputs["output_type"]

        if "image" in inputs:
            image = inputs["image"]
        else:
            image = None

        if "mask_image" in inputs:
            mask_image = inputs["mask_image"]
        else:
            mask_image = None

        if "original_image" in inputs:
            original_image = inputs["original_image"]
        else:
            original_image = None

        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(prompt)

        # inputs with prompt converted to embeddings
        inputs = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "output_type": output_type,
        }

        if image is not None:
            inputs["image"] = image

        if mask_image is not None:
            inputs["mask_image"] = mask_image

        if original_image is not None:
            inputs["original_image"] = original_image

        # set all optional components to None
        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)

        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        pipe_loaded.unet.set_attn_processor(AttnAddedKVProcessor())  # For reproducibility tests

        for optional_component in pipe._optional_components:
            self.assertTrue(
                getattr(pipe_loaded, optional_component) is None,
                f"`{optional_component}` did not stay set to None after loading.",
            )

        inputs = self.get_dummy_inputs(torch_device)

        generator = inputs["generator"]
        num_inference_steps = inputs["num_inference_steps"]
        output_type = inputs["output_type"]

        # inputs with prompt converted to embeddings
        inputs = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "output_type": output_type,
        }

        if image is not None:
            inputs["image"] = image

        if mask_image is not None:
            inputs["mask_image"] = mask_image

        if original_image is not None:
            inputs["original_image"] = original_image

        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(max_diff, 1e-4)

    # Modified from `PipelineTesterMixin` to set the attn processor as it's not serialized.
    # This should be handled in the base test and then this method can be removed.
    def _test_save_load_local(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        pipe_loaded.unet.set_attn_processor(AttnAddedKVProcessor())  # For reproducibility tests

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(max_diff, 1e-4)
