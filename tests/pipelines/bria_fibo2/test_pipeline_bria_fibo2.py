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

    def test_edit_prompt_caption_forms(self):
        # fibo-2 was also trained to edit with a short description, as a string, and with a short description and the
        # caption's text_render list, in that order, the list kept even when empty
        pipe = self.get_pipeline()
        marker = "<|vision_start|><|image_pad|><|vision_end|>"
        description = "A woman in a red coat on a snowy street."
        instruction = "Make the coat green."
        short_json = {"short_description": description, "text_render": []}

        short = json.dumps({"structured_caption": description, "edit_instruction": instruction})
        assert pipe._get_edit_prompt(short, 1) == (
            f'{{"image":"{marker}","structured_caption":"{description}","edit_instruction":"{instruction}"}}'
        )
        for prompt in [
            {**short_json, "edit_instruction": instruction},
            {"text_render": [], "short_description": description, "edit_instruction": instruction},
            {"structured_caption": short_json, "edit_instruction": instruction},
        ]:
            assert pipe._get_edit_prompt(json.dumps(prompt), 1) == (
                f'{{"image":"{marker}","structured_caption":{{"short_description":"{description}","text_render":[]}},'
                f'"edit_instruction":"{instruction}"}}'
            )
        # An empty description is no caption, as in training
        no_caption = json.dumps({"structured_caption": "", "edit_instruction": instruction})
        assert pipe._get_edit_prompt(no_caption, 1) == f'{{"image":"{marker}","edit_instruction":"{instruction}"}}'

    def get_text_image(self, width=24, height=8):
        return PIL.Image.new("RGB", (width, height), "white")

    def record_text_lengths(self, pipe):
        # The length of the text the transformer is conditioned on, at each call
        lengths = []
        pipe.transformer.register_forward_pre_hook(
            lambda module, args, kwargs: lengths.append(kwargs["encoder_hidden_states"].shape[2]), with_kwargs=True
        )
        return lengths

    def test_text_prompt(self):
        # Each text_render entry with text gets the marker of its image, or an empty one where the image isn't shown
        pipe = self.get_pipeline()
        marker = "<|vision_start|><|image_pad|><|vision_end|>"
        prompt = json.dumps(
            {
                "short_description": "a shop sign",
                "text_render": [{"text": "OPEN"}, {"text": ""}, {"location": "door"}, {"text": "7 AM"}],
            }
        )
        entries = '[{{"text":"OPEN","image_tokens":"{0}"}},{{"text":""}},{{"location":"door"}},{{"text":"7 AM","image_tokens":"{0}"}}]'

        assert pipe._get_text_prompt(prompt, marker) == (
            f'{{"short_description":"a shop sign","text_render":{entries.format(marker)}}}'
        )
        assert (
            pipe._get_text_prompt(prompt, "")
            == f'{{"short_description":"a shop sign","text_render":{entries.format("")}}}'
        )

    def test_edit_prompt_with_text_images(self):
        # The caption is cleaned first, then each text entry is marked: its image follows the image to edit
        pipe = self.get_pipeline()
        marker = "<|vision_start|><|image_pad|><|vision_end|>"
        caption = {
            "short_description": "a cafe window",
            "text_render": [{"text": "OPEN DAILY", "font": ""}],
            "edit_instruction": "Turn it into a flat illustration",
        }

        assert pipe._get_edit_prompt(json.dumps(caption), 1, text_images=True) == (
            f'{{"image":"{marker}","structured_caption":{{"short_description":"a cafe window","text_render":'
            f'[{{"text":"OPEN DAILY","image_tokens":"{marker}"}}]}},"edit_instruction":"Turn it into a flat illustration"}}'
        )

    def test_generation_prompt_should_take_the_trained_form(self):
        # Empty values, NaN and the score fields dropped, serialized compactly. A short description and a caption of a
        # short description and its text_render list keep their own trained forms
        pipe = self.get_pipeline()
        pretty = (
            '{\n  "short_description": "a red bicycle",\n  "objects": [],\n  "context": NaN,\n'
            '  "aesthetics": {"mood": "calm", "aesthetic_score": "high"},\n  "preference_score": "very high"\n}'
        )
        clean = '{"short_description":"a red bicycle","aesthetics":{"mood":"calm"}}'

        assert pipe._get_generation_prompt(pretty) == clean
        assert pipe._get_generation_prompt("A red bicycle by a door.") == "A red bicycle by a door."
        assert (
            pipe._get_generation_prompt('{"text_render": [], "short_description": "a sign"}')
            == '{"short_description":"a sign","text_render":[]}'
        )
        from_pretty = pipe(**{**self.get_dummy_inputs(), "prompt": pretty}).images
        from_clean = pipe(**{**self.get_dummy_inputs(), "prompt": clean}).images
        assert torch.equal(from_pretty, from_clean)

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

    def test_draw_text_images(self):
        # One image per text_render entry with text: the text in one line, black on white, split into lines of 256
        # characters
        pipe = self.get_pipeline()
        caption = {"text_render": [{"text": "OPEN\n  DAILY"}, {"text": ""}, {"location": "door"}, {"text": "x" * 257}]}
        images = pipe._draw_text_images(caption)
        one_line = pipe._draw_text_images({"text_render": [{"text": "OPEN DAILY"}, {"text": "x" * 256}]})

        assert len(images) == 2
        assert images[0].mode == "RGB" and images[0].getpixel((0, 0)) == (255, 255, 255)
        assert np.asarray(images[0]).min() == 0
        assert np.array_equal(np.asarray(images[0]), np.asarray(one_line[0]))
        # Lines of 256 and 1 characters, each as tall as the line of 256, and 4 pixels apart
        assert images[1].size == (one_line[1].width, 2 * one_line[1].height)

    def test_text_images_read_fonts_only_when_needed(self):
        # Pillow's built-in font draws ASCII on its own: only a character it lacks needs the fallback fonts
        pipe = self.get_pipeline()
        calls = []
        pipe._get_text_image_fonts = lambda: calls.append(True) or []

        pipe._draw_text_images({"text_render": [{"text": "OPEN 7AM-4PM"}]})
        assert calls == []
        pipe._draw_text_images({"text_render": [{"text": "OPEN 7AM–4PM"}]})
        assert calls == [True]

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
        given = pipe(**inputs, text_images=pipe._draw_text_images(caption)).images
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
