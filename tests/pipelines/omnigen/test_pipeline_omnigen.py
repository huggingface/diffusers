import gc
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, OmniGenPipeline, OmniGenTransformer2DModel
from diffusers.utils.testing_utils import (
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..test_pipelines_common import PipelineTesterMixin


class OmniGenPipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = OmniGenPipeline
    params = frozenset(
        [
            "prompt",
            "guidance_scale",
        ]
    )
    batch_params = frozenset(["prompt", ])

    def get_dummy_components(self):
        torch.manual_seed(0)

        transformer_config = {
            "_name_or_path": "Phi-3-vision-128k-instruct",
            "architectures": [
                "Phi3ForCausalLM"
            ],
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 3072,
            "initializer_range": 0.02,
            "intermediate_size": 8192,
            "max_position_embeddings": 131072,
            "model_type": "phi3",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32,
            "original_max_position_embeddings": 4096,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "long_factor": [
                    1.0299999713897705,
                    1.0499999523162842,
                    1.0499999523162842,
                    1.0799999237060547,
                    1.2299998998641968,
                    1.2299998998641968,
                    1.2999999523162842,
                    1.4499999284744263,
                    1.5999999046325684,
                    1.6499998569488525,
                    1.8999998569488525,
                    2.859999895095825,
                    3.68999981880188,
                    5.419999599456787,
                    5.489999771118164,
                    5.489999771118164,
                    9.09000015258789,
                    11.579999923706055,
                    15.65999984741211,
                    15.769999504089355,
                    15.789999961853027,
                    18.360000610351562,
                    21.989999771118164,
                    23.079999923706055,
                    30.009998321533203,
                    32.35000228881836,
                    32.590003967285156,
                    35.56000518798828,
                    39.95000457763672,
                    53.840003967285156,
                    56.20000457763672,
                    57.95000457763672,
                    59.29000473022461,
                    59.77000427246094,
                    59.920005798339844,
                    61.190006256103516,
                    61.96000671386719,
                    62.50000762939453,
                    63.3700065612793,
                    63.48000717163086,
                    63.48000717163086,
                    63.66000747680664,
                    63.850006103515625,
                    64.08000946044922,
                    64.760009765625,
                    64.80001068115234,
                    64.81001281738281,
                    64.81001281738281
                ],
                "short_factor": [
                    1.05,
                    1.05,
                    1.05,
                    1.1,
                    1.1,
                    1.1,
                    1.2500000000000002,
                    1.2500000000000002,
                    1.4000000000000004,
                    1.4500000000000004,
                    1.5500000000000005,
                    1.8500000000000008,
                    1.9000000000000008,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.000000000000001,
                    2.1000000000000005,
                    2.1000000000000005,
                    2.2,
                    2.3499999999999996,
                    2.3499999999999996,
                    2.3499999999999996,
                    2.3499999999999996,
                    2.3999999999999995,
                    2.3999999999999995,
                    2.6499999999999986,
                    2.6999999999999984,
                    2.8999999999999977,
                    2.9499999999999975,
                    3.049999999999997,
                    3.049999999999997,
                    3.049999999999997
                ],
                "type": "su"
            },
            "rope_theta": 10000.0,
            "sliding_window": 131072,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.38.1",
            "use_cache": True,
            "vocab_size": 32064,
            "_attn_implementation": "sdpa"
        }
        transformer = OmniGenTransformer2DModel(
            transformer_config=transformer_config,
            patch_size=2,
            in_channels=4,
            pos_embed_max_size=192,
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
            up_block_types = ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        )

        scheduler = FlowMatchEulerDiscreteScheduler()
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        # tokenizer = AutoTokenizer.from_pretrained("Shitao/OmniGen-v1")

        components = {
            "transformer": transformer.eval(),
            "vae": vae.eval(),
            "scheduler": scheduler,
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
            "guidance_scale": 3.0,
            "output_type": "np",
            "height": 16,
            "width": 16,
            "use_kv_cache": False,
            "offload_kv_cache": False,
        }
        return inputs

    def test_inference(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        generated_image = pipe(**inputs).images[0]

        self.assertEqual(generated_image.shape, (16, 16, 3))




@slow
@require_torch_gpu
class OmniGenPipelineSlowTests(unittest.TestCase):
    pipeline_class = OmniGenPipeline
    repo_id = "Shitao/OmniGen-v1-diffusers"

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

        expected_slice = np.array(
            [[0.25806782, 0.28012177, 0.27807158],
            [0.25740036, 0.2677201,  0.26857468],
            [0.258638,   0.27035138, 0.26633185],
            [0.2541029,  0.2636156,  0.26373306],
            [0.24975497, 0.2608987,  0.2617477 ],
            [0.25102,    0.26192215, 0.262023  ],
            [0.24452701, 0.25664824, 0.259144  ],
            [0.2419573,  0.2574909,  0.25996095],
            [0.23953134, 0.25292695, 0.25652167],
            [0.23523712, 0.24710432, 0.25460982]],
            dtype=np.float32,
        )

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), image_slice.flatten())

        assert max_diff < 1e-4
