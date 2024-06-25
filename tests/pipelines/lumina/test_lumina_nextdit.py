import gc
import unittest

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, LuminaNextDiT2DModel, LuminaText2ImgPipeline
from diffusers.utils.testing_utils import (
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..test_pipelines_common import PipelineTesterMixin


class LuminaText2ImgPipelinePipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = LuminaText2ImgPipeline
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

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = LuminaNextDiT2DModel(
            sample_size=32,
            patch_size=1,
            in_channels=4,
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            multiple_of=32,
            caption_dim=2048,
            qk_norm=True,
            learn_sigma=True,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL()

        scheduler = FlowMatchEulerDiscreteScheduler()
        text_encoder = AutoModel.from_pretrained("google/gemma-2b")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

        components = {
            "transformer": transformer.eval(),
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
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
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


@slow
@require_torch_gpu
class LuminaText2ImgPipelineSlowTests(unittest.TestCase):
    pipeline_class = LuminaText2ImgPipeline
    repo_id = "Alpha-VLLM/Lumina-Next-SFT-diffusers"

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
                [0.36132812, 0.30004883, 0.25830078],
                [0.36669922, 0.31103516, 0.23754883],
                [0.34814453, 0.29248047, 0.23583984],
                [0.35791016, 0.30981445, 0.23999023],
                [0.36328125, 0.31274414, 0.2607422],
                [0.37304688, 0.32177734, 0.26171875],
                [0.3671875, 0.31933594, 0.25756836],
                [0.36035156, 0.31103516, 0.2578125],
                [0.3857422, 0.33789062, 0.27563477],
                [0.3701172, 0.31982422, 0.265625],
            ],
            dtype=np.float32,
        )

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), image_slice.flatten())

        assert max_diff < 1e-4


if __name__ == "__main__":
    unittest.main()
