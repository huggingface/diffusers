import gc
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, GemmaConfig, GemmaForCausalLM

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    LuminaNextDiT2DModel,
    LuminaPipeline,
)

from ...testing_utils import (
    backend_empty_cache,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..test_pipelines_common import PipelineTesterMixin


class LuminaPipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = LuminaPipeline
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

    supports_dduf = False
    test_layerwise_casting = True
    test_group_offloading = True

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = LuminaNextDiT2DModel(
            sample_size=4,
            patch_size=2,
            in_channels=4,
            hidden_size=4,
            num_layers=2,
            num_attention_heads=1,
            num_kv_heads=1,
            multiple_of=16,
            ffn_dim_multiplier=None,
            norm_eps=1e-5,
            learn_sigma=True,
            qk_norm=True,
            cross_attention_dim=8,
            scaling_factor=1.0,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL()

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
            "output_type": "np",
        }
        return inputs

    @unittest.skip("xformers attention processor does not exist for Lumina")
    def test_xformers_attention_forwardGenerator_pass(self):
        pass


@slow
@require_torch_accelerator
class LuminaPipelineSlowTests(unittest.TestCase):
    pipeline_class = LuminaPipeline
    repo_id = "Alpha-VLLM/Lumina-Next-SFT-diffusers"

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

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
        pipe.enable_model_cpu_offload(device=torch_device)

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
