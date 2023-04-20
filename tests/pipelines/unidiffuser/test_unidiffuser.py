import random
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModel,
    GPT2Tokenizer,
)

from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UniDiffuserModel,
    UniDiffuserPipeline,
    UniDiffuserTextDecoder,
)
from diffusers.utils import floats_tensor, slow
from diffusers.utils.testing_utils import require_torch_gpu

from ...pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ...test_pipelines_common import PipelineTesterMixin


class UniDiffuserPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = UniDiffuserPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UniDiffuserModel(
            text_dim=32,
            clip_img_dim=32,
            num_text_tokens=77,
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=4,
            out_channels=4,
            num_layers=2,
            dropout=0.0,
            norm_num_groups=32,
            attention_bias=False,
            sample_size=16,
            patch_size=2,
            activation_fn="gelu",
            num_embeds_ada_norm=1000,
            norm_type="layer_norm",
            block_type="unidiffuser",
            pre_layer_norm=False,
            use_timestep_embedding=False,
            norm_elementwise_affine=True,
            ff_final_dropout=True,
        )

        scheduler = DPMSolverMultistepScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
        )

        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )

        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        torch.manual_seed(0)
        image_encoder_config = CLIPVisionConfig(
            image_size=32,
            patch_size=2,
            num_channels=3,
            hidden_size=32,
            projection_dim=32,
            num_hidden_layers=5,
            num_attention_heads=4,
            intermediate_size=37,
            dropout=0.1,
            attention_dropout=0.1,
            initializer_range=0.02,
        )
        image_encoder = CLIPVisionModel(image_encoder_config)
        # From the Stable Diffusion Image Variation pipeline tests
        image_processor = CLIPImageProcessor(crop_size=32, size=32)
        # image_processor = CLIPImageProcessor.from_pretrained("hf-internal-testing/tiny-random-clip")

        torch.manual_seed(0)
        # From https://huggingface.co/hf-internal-testing/tiny-random-GPT2Model/blob/main/config.json
        text_tokenizer = GPT2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-GPT2Model")
        text_decoder = UniDiffuserTextDecoder(
            prefix_length=77,
            prefix_hidden_dim=32,
            vocab_size=text_tokenizer.vocab_size,
            n_positions=1024,
            n_embd=32,
            n_layer=5,
            n_head=4,
            n_inner=37,
            activation_function="gelu",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
        )

        components = {
            "vae": vae,
            "text_encoder": text_encoder,
            "image_encoder": image_encoder,
            "image_processor": image_processor,
            "clip_tokenizer": tokenizer,
            "text_decoder": text_decoder,
            "text_tokenizer": text_tokenizer,
            "unet": unet,
            "scheduler": scheduler,
        }

        return components

    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB")
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "an elephant under the sea",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_unidiffuser_default_joint(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        unidiffuser_pipe = UniDiffuserPipeline(**components)
        unidiffuser_pipe = unidiffuser_pipe.to(device)
        unidiffuser_pipe.set_progress_bar_config(disable=None)

        # Set mode to 'joint'
        unidiffuser_pipe.set_joint_mode()
        assert unidiffuser_pipe.mode == "joint"

        inputs = self.get_dummy_inputs(device)
        # Delete prompt and image for joint inference.
        del inputs["prompt"]
        del inputs["image"]
        sample = unidiffuser_pipe(**inputs)
        image = sample.images
        text = sample.text
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        print(image_slice.flatten())
        expected_slice = np.array([0.3965, 0.4568, 0.4495, 0.4590, 0.4465, 0.4690, 0.5454, 0.5093, 0.4321])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

        # TODO: need to figure out correct text output
        print(text)

    def test_unidiffuser_default_text2img(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        unidiffuser_pipe = UniDiffuserPipeline(**components)
        unidiffuser_pipe = unidiffuser_pipe.to(device)
        unidiffuser_pipe.set_progress_bar_config(disable=None)

        # Set mode to 'text2img'
        unidiffuser_pipe.set_text_to_image_mode()
        assert unidiffuser_pipe.mode == "text2img"

        inputs = self.get_dummy_inputs(device)
        # Delete image for text-conditioned image generation
        del inputs["image"]
        image = unidiffuser_pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.3965, 0.4568, 0.4495, 0.4590, 0.4463, 0.4690, 0.5454, 0.5093, 0.4321])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_unidiffuser_default_img2text(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        unidiffuser_pipe = UniDiffuserPipeline(**components)
        unidiffuser_pipe = unidiffuser_pipe.to(device)
        unidiffuser_pipe.set_progress_bar_config(disable=None)

        # Set mode to 'img2text'
        unidiffuser_pipe.set_image_to_text_mode()
        assert unidiffuser_pipe.mode == "img2text"

        inputs = self.get_dummy_inputs(device)
        # Delete text for image-conditioned text generation
        del inputs["prompt"]
        text = unidiffuser_pipe(**inputs).text

        # TODO: need to figure out correct text output
        print(text)
        assert 0 == 1

    def test_unidiffuser_default_text(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        unidiffuser_pipe = UniDiffuserPipeline(**components)
        unidiffuser_pipe = unidiffuser_pipe.to(device)
        unidiffuser_pipe.set_progress_bar_config(disable=None)

        # Set mode to 'text'
        unidiffuser_pipe.set_text_mode()
        assert unidiffuser_pipe.mode == "text"

        inputs = self.get_dummy_inputs(device)
        # Delete prompt and image for unconditional ("marginal") text generation.
        del inputs["prompt"]
        del inputs["image"]
        text = unidiffuser_pipe(**inputs).text

        # TODO: need to figure out correct text output
        print(text)
        assert 0 == 1

    def test_unidiffuser_default_image(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        unidiffuser_pipe = UniDiffuserPipeline(**components)
        unidiffuser_pipe = unidiffuser_pipe.to(device)
        unidiffuser_pipe.set_progress_bar_config(disable=None)

        # Set mode to 'img'
        unidiffuser_pipe.set_image_mode()
        assert unidiffuser_pipe.mode == "img"

        inputs = self.get_dummy_inputs(device)
        # Delete prompt and image for unconditional ("marginal") text generation.
        del inputs["prompt"]
        del inputs["image"]
        image = unidiffuser_pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        print(image_slice.flatten())
        expected_slice = np.array([0.3967, 0.4568, 0.4495, 0.4590, 0.4463, 0.4690, 0.5454, 0.5093, 0.4321])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3


@slow
@require_torch_gpu
class UniDiffuserPipelineSlowTests(unittest.TestCase):
    pass
