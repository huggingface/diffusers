import json

import numpy as np
import PIL.Image
import pytest
import torch
from transformers import AutoConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from diffusers import (
    AutoencoderKLFlux2,
    BriaFibo2Pipeline,
    BriaFibo2Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)

from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


class BriaFibo2PipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = BriaFibo2Pipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])
    output_shape = (3, 8, 8)
    # `encode_prompt` runs the Qwen3-VL processor as well as the text encoder
    text_stack_component_names = ("text", "processor")

    def get_dummy_components(self):
        tiny_ckpt_id = "huangfeice/tiny-random-Qwen3VLForConditionalGeneration"

        torch.manual_seed(0)
        transformer = BriaFibo2Transformer2DModel(
            in_channels=1,
            patch_size=2,
            dim=32,
            n_layers=2,
            n_refiner_layers=1,
            n_heads=2,
            cap_feat_dim=48,  # three Qwen3-VL layers of width 16, side by side
            injection_layer_ids=(1,),
            perceiver_num_layers=1,
            min_num_gist_tokens=4,
            max_num_gist_tokens=8,
            gist_step=4,
            gist_min_text_len=4,
            gist_max_text_len=16,
            axes_dims=(4, 6, 6),
            axes_lens=(16, 16, 16),
        )

        # The pipeline reads Qwen3-VL's hidden states up to layer 35, so the tiny text encoder keeps 36 layers
        config = AutoConfig.from_pretrained(tiny_ckpt_id)
        config.text_config.num_hidden_layers = 36
        torch.manual_seed(0)
        text_encoder = Qwen3VLForConditionalGeneration(config)
        processor = Qwen3VLProcessor.from_pretrained(tiny_ckpt_id)
        text_encoder.resize_token_embeddings(len(processor.tokenizer))

        torch.manual_seed(0)
        vae = AutoencoderKLFlux2(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=1,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "transformer": transformer,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "processor": processor,
        }

    def get_dummy_inputs(self):
        inputs = {
            "prompt": '{"short_description":"a red bicycle leaning against a white brick wall"}',
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 8,
            "width": 8,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }
        return inputs


class TestBriaFibo2Pipeline(BriaFibo2PipelineTesterConfig, PipelineTesterMixin):
    def test_default_negative_prompt_with_guidance(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["guidance_scale"] = 5.0
        default_negative = pipe(**inputs).images

        inputs = self.get_dummy_inputs()
        inputs["guidance_scale"] = 5.0
        inputs["negative_prompt"] = '{"short_description":"blurry"}'
        other_negative = pipe(**inputs).images

        assert default_negative.shape == (1, *self.output_shape)
        assert not torch.allclose(default_negative, other_negative)

    def get_edit_image(self, width=16, height=12, seed=0):
        pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
        return PIL.Image.fromarray(pixels)

    def test_edit(self):
        pipe = self.get_pipeline()
        generated = pipe(**self.get_dummy_inputs()).images

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = "Make the bicycle blue"
        inputs["image"] = self.get_edit_image()
        edited = pipe(**inputs).images

        assert edited.shape == (1, *self.output_shape)
        assert not torch.allclose(edited, generated)

    def test_edit_with_mask(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = "Remove the bicycle"
        inputs["image"] = self.get_edit_image()
        edited = pipe(**inputs).images

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = "Remove the bicycle"
        inputs["image"] = self.get_edit_image()
        inputs["mask"] = PIL.Image.new("L", (16, 12))
        inputs["mask"].paste(255, (4, 2, 12, 10))
        masked = pipe(**inputs).images

        assert masked.shape == (1, *self.output_shape)
        assert not torch.allclose(masked, edited)

    def test_edit_several_images(self):
        pipe = self.get_pipeline()
        # Several images are each encoded at the default area, which the tiny model keeps small
        pipe.default_sample_size = 16
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = '{"short_description":"a blue bicycle","edit_instruction":"Paint <image_1> like <image_2>"}'
        inputs["image"] = [self.get_edit_image(16, 12, seed=0), self.get_edit_image(10, 20, seed=1)]
        edited = pipe(**inputs).images

        assert edited.shape == (1, *self.output_shape)

    def test_edit_prompt(self):
        # The prompt shape fibo-2 was trained to edit with: a vision marker per image, the cleaned caption, then the
        # instruction
        pipe = self.get_pipeline()
        marker = "<|vision_start|><|image_pad|><|vision_end|>"
        caption = {
            "short_description": "a blue bicycle",
            "objects": [],
            "aesthetics": {"mood": "calm", "aesthetic_score": "high"},
            "edit_instruction": "Paint <image_1> like <image_2>",
        }

        assert pipe._get_edit_prompt("Make it blue", 1) == f'{{"image":"{marker}","edit_instruction":"Make it blue"}}'
        assert pipe._get_edit_prompt(json.dumps(caption), 2) == (
            f'{{"image_1":"{marker}","image_2":"{marker}","structured_caption":{{"short_description":"a blue bicycle",'
            f'"aesthetics":{{"mood":"calm"}}}},"edit_instruction":"Paint <image_1> like <image_2>"}}'
        )

    def test_edit_inputs_are_checked(self):
        pipe = self.get_pipeline()
        mask = PIL.Image.new("L", (16, 12))
        for wrong_inputs in [
            {"image": [self.get_edit_image()] * 6},
            {"mask": mask},
            {"image": [self.get_edit_image()] * 2, "mask": mask},
            {"image": self.get_edit_image(), "mask": PIL.Image.new("L", (12, 16))},
            {"image": self.get_edit_image(), "prompt": '{"short_description":"no edit instruction"}'},
        ]:
            inputs = {**self.get_dummy_inputs(), "prompt": "Make it blue", **wrong_inputs}
            with pytest.raises(ValueError):
                pipe(**inputs)


class TestBriaFibo2PipelineMemory(BriaFibo2PipelineTesterConfig, MemoryTesterMixin):
    pass
