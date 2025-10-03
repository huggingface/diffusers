import unittest

import torch
from PIL import Image
from transformers import AutoTokenizer, Gemma2Config, Gemma2Model

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    Lumina2AccessoryPipeline,
    Lumina2AccessoryTransformer2DModel,
)

from ...testing_utils import (
    enable_full_determinism,
    torch_device,
)
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class Lumina2AccessoryPipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = Lumina2AccessoryPipeline
    params = frozenset(
        [
            "prompt",
            "image",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]
    )
    batch_params = frozenset(
        [
            "prompt",
            "image",
            "negative_prompt",
        ]
    )
    image_params = frozenset(["image"])
    image_latents_params = frozenset(["latents"])
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "return_dict",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        ]
    )

    supports_dduf = False
    test_xformers_attention = False
    test_layerwise_casting = True

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = Lumina2AccessoryTransformer2DModel(
            sample_size=4,
            patch_size=2,
            in_channels=4,
            hidden_size=8,
            num_layers=2,
            num_attention_heads=1,
            num_kv_heads=1,
            multiple_of=16,
            ffn_dim_multiplier=None,
            norm_eps=1e-5,
            scaling_factor=1.0,
            axes_dim_rope=[4, 2, 2],
            cap_feat_dim=8,
        )

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
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/dummy-gemma")

        torch.manual_seed(0)
        config = Gemma2Config(
            head_dim=4,
            hidden_size=8,
            intermediate_size=8,
            num_attention_heads=2,
            num_hidden_layers=2,
            num_key_value_heads=2,
            sliding_window=2,
        )
        text_encoder = Gemma2Model(config)

        components = {
            "transformer": transformer,
            "vae": vae.eval(),
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "negative_prompt": "bad quality",
            "image": Image.new("RGB", (32, 32)),
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 4.0,
            "height": 32,
            "width": 32,
            "output_type": "np",
        }
        return inputs

    def test_lumina2_accessory_batch_inputs(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(torch_device)

        inputs = self.get_dummy_inputs(device=torch_device)

        inputs["prompt"] = ["A squirrel", "A cat"]
        inputs["image"] = [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))]

        output = pipe(**inputs)
        assert len(output.images) == 2

    def test_lumina2_accessory_image_output_shape(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        height_width_pairs = [(32, 32), (72, 57)]
        for height, width in height_width_pairs:
            expected_height = height - height % (pipe.vae_scale_factor * 2)
            expected_width = width - width % (pipe.vae_scale_factor * 2)

            inputs.update({"height": height, "width": width})
            image = pipe(**inputs).images[0]
            output_height, output_width, _ = image.shape
            assert (output_height, output_width) == (expected_height, expected_width)

    def test_lumina2_accessory_guidance_scale_effect(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(torch_device)

        inputs = self.get_dummy_inputs(device=torch_device)
        # run with default guidance_scale
        output1 = pipe(**inputs)

        # run with zero guidance_scale
        inputs["guidance_scale"] = 0.0
        output2 = pipe(**inputs)

        # outputs should not be exactly equal
        assert not (output1.images[0] == output2.images[0]).all()
