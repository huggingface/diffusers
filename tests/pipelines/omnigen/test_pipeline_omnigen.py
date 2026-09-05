import gc

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, OmniGenPipeline, OmniGenTransformer2DModel

from ...testing_utils import (
    Expectations,
    backend_empty_cache,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


class OmniGenPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = OmniGenPipeline
    required_input_params_in_call_signature = frozenset(["prompt", "guidance_scale"])
    batch_input_params = frozenset(["prompt"])
    output_shape = (3, 16, 16)

    def get_dummy_components(self):
        torch.manual_seed(0)

        transformer = OmniGenTransformer2DModel(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=32,
            num_layers=1,
            in_channels=4,
            time_step_dim=4,
            rope_scaling={"long_factor": list(range(1, 3)), "short_factor": list(range(1, 3))},
        )

        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4, 4, 4, 4),
            layers_per_block=1,
            latent_channels=4,
            norm_num_groups=1,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        )

        scheduler = FlowMatchEulerDiscreteScheduler(invert_sigmas=True, num_train_timesteps=1)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "tokenizer": tokenizer,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 1,
            "guidance_scale": 3.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "height": 16,
            "width": 16,
        }


class TestOmniGenPipeline(OmniGenPipelineTesterConfig, PipelineTesterMixin):
    """Core pipeline tests for OmniGen. The old `test_inference` only asserted the output shape, which
    `PipelineTesterMixin.test_output` now covers against `output_shape`."""


class TestOmniGenPipelineMemory(OmniGenPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the OmniGen pipeline."""


@slow
@require_torch_accelerator
class TestOmniGenPipelineIntegration:
    pipeline_class = OmniGenPipeline
    repo_id = "shitao/OmniGen-v1-diffusers"

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
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
            "guidance_scale": 2.5,
            "output_type": "np",
            "generator": generator,
        }

    def test_omnigen_inference(self):
        pipe = self.pipeline_class.from_pretrained(self.repo_id, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()

        inputs = self.get_inputs(torch_device)

        image = pipe(**inputs).images[0]
        image_slice = image[0, :10, :10]

        expected_slices = Expectations(
            {
                ("xpu", 3): np.array(
                    [
                        [0.05859375, 0.05859375, 0.04492188],
                        [0.04882812, 0.04101562, 0.03320312],
                        [0.04882812, 0.04296875, 0.03125],
                        [0.04296875, 0.0390625, 0.03320312],
                        [0.04296875, 0.03710938, 0.03125],
                        [0.04492188, 0.0390625, 0.03320312],
                        [0.04296875, 0.03710938, 0.03125],
                        [0.04101562, 0.03710938, 0.02734375],
                        [0.04101562, 0.03515625, 0.02734375],
                        [0.04101562, 0.03515625, 0.02929688],
                    ],
                    dtype=np.float32,
                ),
                ("cuda", 7): np.array(
                    [
                        [0.1783447, 0.16772744, 0.14339337],
                        [0.17066911, 0.15521264, 0.13757327],
                        [0.17072496, 0.15531206, 0.13524258],
                        [0.16746324, 0.1564025, 0.13794944],
                        [0.16490817, 0.15258026, 0.13697758],
                        [0.16971767, 0.15826806, 0.13928896],
                        [0.16782972, 0.15547255, 0.13783783],
                        [0.16464645, 0.15281534, 0.13522372],
                        [0.16535294, 0.15301755, 0.13526791],
                        [0.16365296, 0.15092957, 0.13443318],
                    ],
                    dtype=np.float32,
                ),
                ("cuda", 8): np.array(
                    [
                        [0.0546875, 0.05664062, 0.04296875],
                        [0.046875, 0.04101562, 0.03320312],
                        [0.05078125, 0.04296875, 0.03125],
                        [0.04296875, 0.04101562, 0.03320312],
                        [0.0390625, 0.03710938, 0.02929688],
                        [0.04296875, 0.03710938, 0.03125],
                        [0.0390625, 0.03710938, 0.02929688],
                        [0.0390625, 0.03710938, 0.02734375],
                        [0.0390625, 0.03320312, 0.02734375],
                        [0.0390625, 0.03320312, 0.02734375],
                    ],
                    dtype=np.float32,
                ),
            }
        )
        expected_slice = expected_slices.get_expectation()

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), image_slice.flatten())

        assert max_diff < 1e-4
