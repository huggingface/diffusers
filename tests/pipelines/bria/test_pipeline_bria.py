# Copyright 2024 Bria AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import tempfile
import unittest

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import T5EncoderModel, T5TokenizerFast

from diffusers import (
    AutoencoderKL,
    BriaTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.pipelines.bria import BriaPipeline

# from ..test_pipelines_common import PipelineTesterMixin, check_qkv_fused_layers_exist
from tests.pipelines.test_pipelines_common import PipelineTesterMixin, to_np

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)


enable_full_determinism()


class BriaPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = BriaPipeline
    params = frozenset(["prompt", "height", "width", "guidance_scale", "prompt_embeds"])
    batch_params = frozenset(["prompt"])
    test_xformers_attention = False

    # there is no xformers processor for Flux
    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = BriaTransformer2DModel(
            patch_size=1,
            in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=None,
            axes_dims_rope=[0, 4, 4],
        )

        torch.manual_seed(0)
        vae = AutoencoderKL(
            act_fn="silu",
            block_out_channels=(32,),
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=32,
            shift_factor=0,
            scaling_factor=0.13025,
            use_post_quant_conv=True,
            use_quant_conv=True,
            force_upcast=False,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        torch.manual_seed(0)
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer = T5TokenizerFast.from_pretrained("hf-internal-testing/tiny-random-t5")

        components = {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "transformer": transformer,
            "vae": vae,
            "image_encoder": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "negative_prompt": "bad, ugly",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 16,
            "width": 16,
            "max_sequence_length": 48,
            "output_type": "np",
        }
        return inputs

    def test_encode_prompt_works_in_isolation(self):
        pass

    def test_bria_different_prompts(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)
        output_same_prompt = pipe(**inputs).images[0]
        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt"] = "a different prompt"
        output_different_prompts = pipe(**inputs).images[0]
        max_diff = np.abs(output_same_prompt - output_different_prompts).max()
        assert max_diff > 1e-6

    def test_image_output_shape(self):
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

    @unittest.skipIf(torch_device not in ["cuda", "xpu"], reason="float16 requires CUDA or XPU")
    @require_torch_accelerator
    def test_save_load_float16(self, expected_max_diff=1e-2):
        components = self.get_dummy_components()
        for name, module in components.items():
            if hasattr(module, "half"):
                components[name] = module.to(torch_device).half()

        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir, torch_dtype=torch.float16)
            for component in pipe_loaded.components.values():
                if hasattr(component, "set_default_attn_processor"):
                    component.set_default_attn_processor()
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for name, component in pipe_loaded.components.items():
            if name == "vae":
                continue
            if hasattr(component, "dtype"):
                self.assertTrue(
                    component.dtype == torch.float16,
                    f"`{name}.dtype` switched from `float16` to {component.dtype} after loading.",
                )

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs)[0]
        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(
            max_diff, expected_max_diff, "The output of the fp16 pipeline changed after saving and loading."
        )

    def test_bria_image_output_shape(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        height_width_pairs = [(16, 16), (32, 32), (64, 64)]
        for height, width in height_width_pairs:
            expected_height = height - height % (pipe.vae_scale_factor * 2)
            expected_width = width - width % (pipe.vae_scale_factor * 2)

            inputs.update({"height": height, "width": width})
            image = pipe(**inputs).images[0]
            output_height, output_width, _ = image.shape
            assert (output_height, output_width) == (expected_height, expected_width)

    def test_to_dtype(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        model_dtypes = [component.dtype for component in components.values() if hasattr(component, "dtype")]
        self.assertTrue([dtype == torch.float32 for dtype in model_dtypes] == [True, True, True])

    def test_torch_dtype_dict(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            torch_dtype_dict = {"transformer": torch.bfloat16, "default": torch.float16}
            loaded_pipe = self.pipeline_class.from_pretrained(tmpdirname, torch_dtype=torch_dtype_dict)

            self.assertEqual(loaded_pipe.transformer.dtype, torch.bfloat16)
            self.assertEqual(loaded_pipe.text_encoder.dtype, torch.float16)
            self.assertEqual(loaded_pipe.vae.dtype, torch.float16)

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            torch_dtype_dict = {"default": torch.float16}
            loaded_pipe = self.pipeline_class.from_pretrained(tmpdirname, torch_dtype=torch_dtype_dict)

            self.assertEqual(loaded_pipe.transformer.dtype, torch.float16)
            self.assertEqual(loaded_pipe.text_encoder.dtype, torch.float16)
            self.assertEqual(loaded_pipe.vae.dtype, torch.float16)


@slow
@require_torch_accelerator
class BriaPipelineSlowTests(unittest.TestCase):
    pipeline_class = BriaPipeline
    repo_id = "briaai/BRIA-3.2"

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, seed=0):
        generator = torch.Generator(device="cpu").manual_seed(seed)

        prompt_embeds = torch.load(
            hf_hub_download(repo_id="diffusers/test-slices", repo_type="dataset", filename="flux/prompt_embeds.pt")
        ).to(torch_device)

        return {
            "prompt_embeds": prompt_embeds,
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "max_sequence_length": 256,
            "output_type": "np",
            "generator": generator,
        }

    def test_bria_inference_bf16(self):
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id, torch_dtype=torch.bfloat16, text_encoder=None, tokenizer=None
        )
        pipe.to(torch_device)

        inputs = self.get_inputs(torch_device)

        image = pipe(**inputs).images[0]
        image_slice = image[0, :10, :10].flatten()

        expected_slice = np.array(
            [
                0.59729785,
                0.6153719,
                0.595112,
                0.5884763,
                0.59366125,
                0.5795311,
                0.58325,
                0.58449626,
                0.57737637,
                0.58432233,
                0.5867875,
                0.57824117,
                0.5819089,
                0.5830988,
                0.57730293,
                0.57647324,
                0.5769151,
                0.57312685,
                0.57926565,
                0.5823928,
                0.57783926,
                0.57162863,
                0.575649,
                0.5745547,
                0.5740556,
                0.5799735,
                0.57799566,
                0.5715559,
                0.5771242,
                0.5773058,
            ],
            dtype=np.float32,
        )
        max_diff = numpy_cosine_similarity_distance(expected_slice, image_slice)
        self.assertLess(max_diff, 1e-4, f"Image slice is different from expected slice: {max_diff:.4f}")
