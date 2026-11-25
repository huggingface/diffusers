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
from diffusers.pipelines.bria_fibo2.image_processor import BriaFibo2ImageProcessor
from diffusers.pipelines.bria_fibo2.text_images import draw_text_image
from diffusers.utils import logging

from ...testing_utils import CaptureLogger
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

    def test_mask_should_be_made_binary(self):
        # Training greyed out the region through a binary mask: from 128 up a pixel is greyed out fully, below not at all
        pipe = self.get_pipeline()

        def edit(value):
            mask = PIL.Image.new("L", (16, 12))
            mask.paste(value, (4, 2, 12, 10))
            inputs = {**self.get_dummy_inputs(), "prompt": "Remove the bicycle", "image": self.get_edit_image()}
            return pipe(**inputs, mask=mask).images

        assert torch.equal(edit(128), edit(255))
        assert torch.equal(edit(127), edit(0))
        assert not torch.allclose(edit(255), edit(0))

    def test_edit_several_images(self):
        pipe = self.get_pipeline()
        # Several images are each encoded at the default area, which the tiny model keeps small
        pipe.default_sample_size = 16
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = '{"short_description":"a blue bicycle","edit_instruction":"Paint <image_1> like <image_2>"}'
        inputs["image"] = [self.get_edit_image(16, 12, seed=0), self.get_edit_image(10, 20, seed=1)]
        edited = pipe(**inputs).images

        assert edited.shape == (1, *self.output_shape)

    def get_text_image(self, width=24, height=8):
        return PIL.Image.new("RGB", (width, height), "white")

    def record_text_lengths(self, pipe):
        # The length of the text the transformer is conditioned on, at each call
        lengths = []
        pipe.transformer.register_forward_pre_hook(
            lambda module, args, kwargs: lengths.append(kwargs["encoder_hidden_states"].shape[2]), with_kwargs=True
        )
        return lengths

    def test_generation_prompt_should_take_the_trained_form(self):
        # Empty values, NaN and the score fields are dropped and the caption serialized compactly, so a pretty-printed
        # caption with them renders as its clean compact form. A short description and its text_render list keep
        # their trained key order
        pipe = self.get_pipeline()
        pretty = (
            '{\n  "short_description": "a red bicycle",\n  "objects": [],\n  "context": NaN,\n'
            '  "aesthetics": {"mood": "calm", "aesthetic_score": "high"},\n  "preference_score": "very high"\n}'
        )
        clean = '{"short_description":"a red bicycle","aesthetics":{"mood":"calm"}}'

        for given, trained in [
            (pretty, clean),
            ('{"text_render": [], "short_description": "a sign"}', '{"short_description":"a sign","text_render":[]}'),
        ]:
            from_given = pipe(**{**self.get_dummy_inputs(), "prompt": given}).images
            from_trained = pipe(**{**self.get_dummy_inputs(), "prompt": trained}).images
            assert torch.equal(from_given, from_trained)

    def test_generation_with_text_images(self):
        pipe = self.get_pipeline()
        prompt = '{"short_description":"a shop sign","text_render":[{"text":"OPEN"}]}'
        without = pipe(**{**self.get_dummy_inputs(), "prompt": prompt, "render_text_images": False}).images
        with_images = pipe(
            **{**self.get_dummy_inputs(), "prompt": prompt, "text_images": [self.get_text_image()]}
        ).images

        assert with_images.shape == (1, *self.output_shape)
        assert not torch.allclose(with_images, without)

    def test_generation_shows_text_images_only_at_low_noise(self):
        pipe = self.get_pipeline()
        lengths = self.record_text_lengths(pipe)
        inputs = {
            **self.get_dummy_inputs(),
            "num_inference_steps": 4,
            "prompt": '{"short_description":"a shop sign","text_render":[{"text":"OPEN"}]}',
            "text_images": [self.get_text_image()],
        }
        pipe(**inputs)

        shown = [level <= pipe.text_images_max_noise_level for level in (pipe.scheduler.timesteps / 1000).tolist()]
        assert True in shown and False in shown
        assert [length == max(lengths) for length in lengths] == shown

    def test_edit_shows_text_images_at_every_step(self):
        pipe = self.get_pipeline()
        lengths = self.record_text_lengths(pipe)
        inputs = {
            **self.get_dummy_inputs(),
            "num_inference_steps": 4,
            "prompt": '{"text_render":[{"text":"OPEN"}],"edit_instruction":"Make the sign blue"}',
            "image": self.get_edit_image(),
            "text_images": [self.get_text_image()],
        }
        pipe(**inputs)

        assert len(lengths) == 4 and len(set(lengths)) == 1

    def test_text_images_are_checked(self):
        pipe = self.get_pipeline()
        caption = '{"short_description":"a shop sign","text_render":[{"text":"OPEN"}]}'
        for wrong_inputs in [
            {"prompt": caption, "text_images": [self.get_text_image()] * 2},
            {"prompt": "Write OPEN on the sign", "text_images": [self.get_text_image()]},
            {"prompt": [caption, caption], "text_images": [self.get_text_image()]},
            {"prompt": caption, "text_images": ["not an image"]},
            {
                "prompt": '{"structured_caption":"a shop sign","edit_instruction":"Make the sign blue"}',
                "image": self.get_edit_image(),
                "text_images": [self.get_text_image()],
            },
        ]:
            with pytest.raises(ValueError):
                pipe(**{**self.get_dummy_inputs(), **wrong_inputs})

    @pytest.mark.parametrize("edit", [False, True], ids=["generation", "edit"])
    def test_render_text_images(self, edit):
        # By default the pipeline draws the images it would otherwise be given as text_images
        pipe = self.get_pipeline()
        caption = {"short_description": "a shop sign", "text_render": [{"text": "OPEN"}, {"text": "7 AM"}]}
        inputs = {**self.get_dummy_inputs(), "prompt": json.dumps(caption)}
        if edit:
            inputs["prompt"] = json.dumps({**caption, "edit_instruction": "Make the sign blue"})
            inputs["image"] = self.get_edit_image()

        rendered = pipe(**inputs).images
        inputs["generator"] = self.get_generator(0)
        # The text is ASCII, which Pillow's built-in font draws without fallback fonts
        drawn = [draw_text_image(entry["text"]) for entry in caption["text_render"]]
        given = pipe(**inputs, text_images=drawn).images
        inputs["generator"] = self.get_generator(0)
        without = pipe(**inputs, render_text_images=False).images

        assert torch.equal(rendered, given)
        assert not torch.allclose(rendered, without)

    def test_render_text_images_without_text(self):
        # A caption without text, a prompt without a caption and a short description get no images of text
        pipe = self.get_pipeline()
        for prompt_inputs in [
            {"prompt": '{"short_description":"a shop sign","text_render":[]}'},
            {"prompt": "a shop sign"},
            {
                "prompt": '{"structured_caption":"a shop sign","edit_instruction":"Make the sign blue"}',
                "image": self.get_edit_image(),
            },
        ]:
            inputs = {**self.get_dummy_inputs(), **prompt_inputs}
            rendered = pipe(**inputs).images
            inputs["generator"] = self.get_generator(0)
            without = pipe(**inputs, render_text_images=False).images

            assert torch.equal(rendered, without)

    def test_text_the_built_in_font_lacks_should_warn_without_fonts(self):
        # The pipeline doesn't fetch fallback fonts itself: drawing a text that needs them warns when none are given
        pipe = self.get_pipeline()
        inputs = {
            **self.get_dummy_inputs(),
            "prompt": '{"short_description":"a shop sign","text_render":[{"text":"ΑΡΤΟΠΟΙΕΙΟ"}]}',
        }
        logger = logging.get_logger("diffusers.pipelines.bria_fibo2.text_images")
        with CaptureLogger(logger) as captured:
            pipe(**inputs)

        assert "`fonts`" in captured.out

    def test_given_text_images_should_take_precedence_over_drawn_ones(self):
        pipe = self.get_pipeline()
        inputs = {
            **self.get_dummy_inputs(),
            "prompt": '{"short_description":"a shop sign","text_render":[{"text":"OPEN"}]}',
        }
        given = pipe(**inputs, text_images=[self.get_text_image()]).images
        inputs["generator"] = self.get_generator(0)
        only_given = pipe(**inputs, text_images=[self.get_text_image()], render_text_images=False).images
        inputs["generator"] = self.get_generator(0)
        drawn = pipe(**inputs).images

        assert torch.equal(given, only_given)
        assert not torch.allclose(given, drawn)

    def test_a_batch_should_draw_each_captions_own_text_images(self):
        # Each prompt's images follow its own markers, so a batch renders as its prompts do one at a time
        pipe = self.get_pipeline()
        captions = [
            '{"short_description":"a shop sign","text_render":[{"text":"OPEN"}]}',
            '{"short_description":"a red bicycle"}',
        ]
        generators = [self.get_generator(0), self.get_generator(1)]
        batch = pipe(**{**self.get_dummy_inputs(), "prompt": captions, "generator": generators}).images
        singles = [
            pipe(**{**self.get_dummy_inputs(), "prompt": caption, "generator": self.get_generator(seed)}).images
            for seed, caption in enumerate(captions)
        ]

        assert torch.allclose(batch, torch.cat(singles), atol=1e-4)

    def test_edit_inputs_are_checked(self):
        pipe = self.get_pipeline()
        mask = PIL.Image.new("L", (16, 12))
        for wrong_inputs in [
            {"image": [self.get_edit_image()] * 6},
            {"mask": mask},
            {"image": [self.get_edit_image()] * 2, "mask": mask},
            {"image": self.get_edit_image(), "mask": PIL.Image.new("L", (12, 16))},
            {"image": self.get_edit_image(), "prompt": '{"short_description":"no edit instruction"}'},
            {
                "image": self.get_edit_image(),
                "prompt": '{"structured_caption":["a sign"],"edit_instruction":"Make it blue"}',
            },
            {
                "image": self.get_edit_image(),
                "prompt": '{"structured_caption":"a sign","objects":[],"edit_instruction":"Make it blue"}',
            },
        ]:
            inputs = {**self.get_dummy_inputs(), "prompt": "Make it blue", **wrong_inputs}
            with pytest.raises(ValueError):
                pipe(**inputs)


class TestBriaFibo2PipelineMemory(BriaFibo2PipelineTesterConfig, MemoryTesterMixin):
    pass


class TestBriaFibo2ImageProcessor:
    @pytest.mark.parametrize(
        "aspect_ratio, size", [(16 / 9, (1424, 800)), (1.0, (1024, 1024)), (4 / 3, (1152, 864)), (1 / 5, (464, 2320))]
    )
    def test_get_image_size_should_be_the_size_fibo2_was_trained_at(self, aspect_ratio, size):
        assert BriaFibo2ImageProcessor().get_image_size(aspect_ratio) == size

    def test_preprocess_should_crop_several_images_each_to_its_own_trained_size(self):
        images = [PIL.Image.new("RGB", (1600, 900)), PIL.Image.new("RGB", (500, 500))]
        vae_images, text_encoder_images = BriaFibo2ImageProcessor().preprocess(images)

        assert [tuple(img.shape) for img in vae_images] == [(1, 3, 800, 1424), (1, 3, 1024, 1024)]
        assert [img.size for img in text_encoder_images] == [(768, 431), (768, 768)]

    def test_preprocess_should_grey_out_the_masked_region(self):
        image = PIL.Image.new("RGB", (1024, 1024), (255, 0, 0))
        mask = PIL.Image.new("L", (1024, 1024))
        mask.paste(255, (0, 0, 512, 1024))
        _, (text_encoder_image,) = BriaFibo2ImageProcessor().preprocess(image, mask=mask)

        assert text_encoder_image.getpixel((0, 0)) == (128, 128, 128)
        assert text_encoder_image.getpixel((767, 0)) == (255, 0, 0)
