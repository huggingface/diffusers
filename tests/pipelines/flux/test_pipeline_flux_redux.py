import gc
import unittest

import numpy as np
import torch

from diffusers import FluxPipeline, FluxPriorReduxPipeline
from diffusers.utils import load_image

from ...testing_utils import (
    Expectations,
    backend_empty_cache,
    numpy_cosine_similarity_distance,
    require_big_accelerator,
    slow,
    torch_device,
)


@slow
@require_big_accelerator
class FluxReduxSlowTests(unittest.TestCase):
    pipeline_class = FluxPriorReduxPipeline
    repo_id = "black-forest-labs/FLUX.1-Redux-dev"
    base_pipeline_class = FluxPipeline
    base_repo_id = "black-forest-labs/FLUX.1-schnell"

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, seed=0):
        init_image = load_image(
            "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/style_ziggy/img5.png"
        )
        return {"image": init_image}

    def get_base_pipeline_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        return {
            "num_inference_steps": 2,
            "guidance_scale": 2.0,
            "output_type": "np",
            "generator": generator,
        }

    def test_flux_redux_inference(self):
        pipe_redux = self.pipeline_class.from_pretrained(self.repo_id, torch_dtype=torch.bfloat16)
        pipe_base = self.base_pipeline_class.from_pretrained(
            self.base_repo_id, torch_dtype=torch.bfloat16, text_encoder=None, text_encoder_2=None
        )
        pipe_redux.to(torch_device)
        pipe_base.enable_model_cpu_offload(device=torch_device)

        inputs = self.get_inputs(torch_device)
        base_pipeline_inputs = self.get_base_pipeline_inputs(torch_device)

        redux_pipeline_output = pipe_redux(**inputs)
        image = pipe_base(**base_pipeline_inputs, **redux_pipeline_output).images[0]

        image_slice = image[0, :10, :10]
        expected_slices = Expectations(
            {
                ("cuda", 7): np.array(
                    [
                        0.30078125,
                        0.37890625,
                        0.46875,
                        0.28125,
                        0.36914062,
                        0.47851562,
                        0.28515625,
                        0.375,
                        0.4765625,
                        0.28125,
                        0.375,
                        0.48046875,
                        0.27929688,
                        0.37695312,
                        0.47851562,
                        0.27734375,
                        0.38085938,
                        0.4765625,
                        0.2734375,
                        0.38085938,
                        0.47265625,
                        0.27539062,
                        0.37890625,
                        0.47265625,
                        0.27734375,
                        0.37695312,
                        0.47070312,
                        0.27929688,
                        0.37890625,
                        0.47460938,
                    ],
                    dtype=np.float32,
                ),
                ("xpu", 3): np.array(
                    [
                        0.20507812,
                        0.30859375,
                        0.3984375,
                        0.18554688,
                        0.30078125,
                        0.41015625,
                        0.19921875,
                        0.3125,
                        0.40625,
                        0.19726562,
                        0.3125,
                        0.41601562,
                        0.19335938,
                        0.31445312,
                        0.4140625,
                        0.1953125,
                        0.3203125,
                        0.41796875,
                        0.19726562,
                        0.32421875,
                        0.41992188,
                        0.19726562,
                        0.32421875,
                        0.41992188,
                        0.20117188,
                        0.32421875,
                        0.41796875,
                        0.203125,
                        0.32617188,
                        0.41796875,
                    ],
                    dtype=np.float32,
                ),
            }
        )
        expected_slice = expected_slices.get_expectation()

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), image_slice.flatten())

        assert max_diff < 1e-4
