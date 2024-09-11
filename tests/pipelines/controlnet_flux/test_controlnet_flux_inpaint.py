import gc
import random
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxControlNetInpaintPipeline,
    FluxTransformer2DModel,
)
from diffusers.models import FluxControlNetModel
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    require_torch_gpu,
    slow,
    torch_device,
)
from diffusers.utils.torch_utils import randn_tensor

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class FluxControlNetInpaintPipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = FluxControlNetInpaintPipeline

    params = frozenset(
        [
            "prompt",
            "image",
            "mask_image",
            "control_image",
            "height",
            "width",
            "strength",
            "guidance_scale",
            "prompt_embeds",
            "pooled_prompt_embeds",
        ]
    )
    batch_params = frozenset(["prompt", "image", "mask_image", "control_image"])

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
        )

        torch.manual_seed(0)
        controlnet = FluxControlNetModel(
            patch_size=1,
            in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
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
        tokenizer_2 = T5TokenizerFast.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=4,
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
            "controlnet": controlnet,
        }

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        mask_image = torch.ones((1, 1, 32, 32)).to(device)
        control_image = randn_tensor((1, 3, 32, 32), generator=generator, device=device, dtype=torch.float32)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "mask_image": mask_image,
            "control_image": control_image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 3.5,
            "output_type": "np",
            "controlnet_conditioning_scale": 0.5,
            "strength": 0.8,
        }

        return inputs

    def test_controlnet_inpaint_flux(self):
        components = self.get_dummy_components()
        flux_pipe = FluxControlNetInpaintPipeline(**components)
        flux_pipe = flux_pipe.to(torch_device)
        flux_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = flux_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)

        expected_slice = np.array([0.5182, 0.4976, 0.4718, 0.5249, 0.5039, 0.4751, 0.5168, 0.4980, 0.4738])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_attention_slicing_forward_pass(self):
        components = self.get_dummy_components()
        flux_pipe = FluxControlNetInpaintPipeline(**components)
        flux_pipe = flux_pipe.to(torch_device)
        flux_pipe.set_progress_bar_config(disable=None)

        flux_pipe.enable_attention_slicing()
        inputs = self.get_dummy_inputs(torch_device)
        output_sliced = flux_pipe(**inputs)
        image_sliced = output_sliced.images

        flux_pipe.disable_attention_slicing()
        inputs = self.get_dummy_inputs(torch_device)
        output = flux_pipe(**inputs)
        image = output.images

        assert np.abs(image_sliced.flatten() - image.flatten()).max() < 1e-3

    def test_inference_batch_single_identical(self):
        components = self.get_dummy_components()
        flux_pipe = FluxControlNetInpaintPipeline(**components)
        flux_pipe = flux_pipe.to(torch_device)
        flux_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)

        # make batch size 1
        inputs["prompt"] = [inputs["prompt"]]
        inputs["image"] = inputs["image"][:1]
        inputs["mask_image"] = inputs["mask_image"][:1]
        inputs["control_image"] = inputs["control_image"][:1]

        output = flux_pipe(**inputs)
        image = output.images

        inputs["prompt"] = inputs["prompt"] * 2
        inputs["image"] = torch.cat([inputs["image"], inputs["image"]])
        inputs["mask_image"] = torch.cat([inputs["mask_image"], inputs["mask_image"]])
        inputs["control_image"] = torch.cat([inputs["control_image"], inputs["control_image"]])

        output_batch = flux_pipe(**inputs)
        image_batch = output_batch.images

        assert np.abs(image_batch[0].flatten() - image[0].flatten()).max() < 1e-3
        assert np.abs(image_batch[1].flatten() - image[0].flatten()).max() < 1e-3

    def test_flux_controlnet_inpaint_prompt_embeds(self):
        components = self.get_dummy_components()
        flux_pipe = FluxControlNetInpaintPipeline(**components)
        flux_pipe = flux_pipe.to(torch_device)
        flux_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = flux_pipe(**inputs)
        image = output.images[0]

        inputs = self.get_dummy_inputs(torch_device)
        prompt = inputs.pop("prompt")

        (prompt_embeds, pooled_prompt_embeds, text_ids) = flux_pipe.encode_prompt(prompt, device=torch_device)
        inputs["prompt_embeds"] = prompt_embeds
        inputs["pooled_prompt_embeds"] = pooled_prompt_embeds
        output = flux_pipe(**inputs)
        image_from_embeds = output.images[0]

        assert np.abs(image - image_from_embeds).max() < 1e-3


@slow
@require_torch_gpu
class FluxControlNetInpaintPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_canny(self):
        controlnet = FluxControlNetModel.from_pretrained(
            "InstantX/FLUX.1-dev-Controlnet-Canny-alpha", torch_dtype=torch.bfloat16
        )
        pipe = FluxControlNetInpaintPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", controlnet=controlnet, torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "A girl in city, 25 years old, cool, futuristic"
        control_image = load_image(
            "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny-alpha/resolve/main/canny.jpg"
        )
        init_image = load_image(
            "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny-alpha/resolve/main/init_image.png"
        )
        mask_image = torch.ones((1, 1, init_image.height, init_image.width))

        output = pipe(
            prompt,
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
            controlnet_conditioning_scale=0.6,
            strength=0.7,
            num_inference_steps=3,
            guidance_scale=3.5,
            output_type="np",
            generator=generator,
        )

        image = output.images[0]

        assert image.shape == (1024, 1024, 3)

        image_slice = image[-3:, -3:, -1].flatten()
        expected_slice = np.array([0.3242, 0.3320, 0.3359, 0.3281, 0.3398, 0.3359, 0.3086, 0.3203, 0.3203])
        assert np.abs(image_slice - expected_slice).max() < 1e-2