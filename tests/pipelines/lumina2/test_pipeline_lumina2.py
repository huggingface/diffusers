import gc
import unittest

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, GemmaConfig, GemmaForCausalLM

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    Lumina2Text2ImgPipeline,
    Lumina2Transformer2DModel,
)
from diffusers.utils.testing_utils import (
    nightly,
    numpy_cosine_similarity_distance,
    require_big_gpu_with_torch_cuda,
    torch_device,
)

from ..test_pipelines_common import PipelineTesterMixin


class Lumina2Text2ImgPipelinePipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = Lumina2Text2ImgPipeline
    params = frozenset(
        [
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]
    )
    batch_params = frozenset(["prompt", "negative_prompt"])
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

    test_xformers_attention = False
    test_layerwise_casting = True

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = Lumina2Transformer2DModel(
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
        config = GemmaConfig(
            head_dim=2,
            hidden_size=8,
            intermediate_size=37,
            num_attention_heads=4,
            num_hidden_layers=2,
            num_key_value_heads=4,
        )
        text_encoder = GemmaForCausalLM(config)

        components = {
            "transformer": transformer.eval(),
            "vae": vae.eval(),
            "scheduler": scheduler,
            "text_encoder": text_encoder.eval(),
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
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 32,
            "width": 32,
            "output_type": "np",
        }
        return inputs

    def test_lumina_prompt_embeds(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        output_with_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        prompt = inputs.pop("prompt")

        do_classifier_free_guidance = inputs["guidance_scale"] > 1
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = pipe.encode_prompt(
            prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            device=torch_device,
        )
        output_with_embeds = pipe(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            **inputs,
        ).images[0]

        max_diff = np.abs(output_with_prompt - output_with_embeds).max()
        assert max_diff < 1e-4


@nightly
@require_big_gpu_with_torch_cuda
@pytest.mark.big_gpu_with_torch_cuda
class Lumina2Text2ImgPipelineSlowTests(unittest.TestCase):
    pipeline_class = Lumina2Text2ImgPipeline
    repo_id = "Alpha-VLLM/Lumina-Image-2.0"

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        return {
            "prompt": "A photo of a cat",
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "generator": generator,
        }

    def test_lumina_inference(self):
        pipe = self.pipeline_class.from_pretrained(self.repo_id, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images[0]
        image_slice = image[0, :10, :10]
        expected_slice = np.array(
            [
                [0.17773438, 0.18554688, 0.22070312],
                [0.046875, 0.06640625, 0.10351562],
                [0.0, 0.0, 0.02148438],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), image_slice.flatten())

        assert max_diff < 1e-4
