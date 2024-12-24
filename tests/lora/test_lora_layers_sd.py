# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    StableDiffusionPipeline,
)
from diffusers.utils.import_utils import is_accelerate_available
from diffusers.utils.testing_utils import (
    load_image,
    nightly,
    numpy_cosine_similarity_distance,
    require_peft_backend,
    require_torch_gpu,
    slow,
    torch_device,
)


sys.path.append(".")

from utils import PeftLoraLoaderMixinTests, check_if_lora_correctly_set  # noqa: E402


if is_accelerate_available():
    from accelerate.utils import release_memory


class StableDiffusionLoRATests(PeftLoraLoaderMixinTests, unittest.TestCase):
    pipeline_class = StableDiffusionPipeline
    scheduler_cls = DDIMScheduler
    scheduler_kwargs = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "clip_sample": False,
        "set_alpha_to_one": False,
        "steps_offset": 1,
    }
    unet_kwargs = {
        "block_out_channels": (32, 64),
        "layers_per_block": 2,
        "sample_size": 32,
        "in_channels": 4,
        "out_channels": 4,
        "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D"),
        "up_block_types": ("CrossAttnUpBlock2D", "UpBlock2D"),
        "cross_attention_dim": 32,
    }
    vae_kwargs = {
        "block_out_channels": [32, 64],
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
        "latent_channels": 4,
    }
    text_encoder_cls, text_encoder_id = CLIPTextModel, "peft-internal-testing/tiny-clip-text-2"
    tokenizer_cls, tokenizer_id = CLIPTokenizer, "peft-internal-testing/tiny-clip-text-2"

    @property
    def output_shape(self):
        return (1, 64, 64, 3)

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    # Keeping this test here makes sense because it doesn't look any integration
    # (value assertions on logits).
    @slow
    @require_torch_gpu
    def test_integration_move_lora_cpu(self):
        path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        lora_id = "takuma104/lora-test-text-encoder-lora-target"

        pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16)
        pipe.load_lora_weights(lora_id, adapter_name="adapter-1")
        pipe.load_lora_weights(lora_id, adapter_name="adapter-2")
        pipe = pipe.to(torch_device)

        self.assertTrue(
            check_if_lora_correctly_set(pipe.text_encoder),
            "Lora not correctly set in text encoder",
        )

        self.assertTrue(
            check_if_lora_correctly_set(pipe.unet),
            "Lora not correctly set in text encoder",
        )

        # We will offload the first adapter in CPU and check if the offloading
        # has been performed correctly
        pipe.set_lora_device(["adapter-1"], "cpu")

        for name, module in pipe.unet.named_modules():
            if "adapter-1" in name and not isinstance(module, (nn.Dropout, nn.Identity)):
                self.assertTrue(module.weight.device == torch.device("cpu"))
            elif "adapter-2" in name and not isinstance(module, (nn.Dropout, nn.Identity)):
                self.assertTrue(module.weight.device != torch.device("cpu"))

        for name, module in pipe.text_encoder.named_modules():
            if "adapter-1" in name and not isinstance(module, (nn.Dropout, nn.Identity)):
                self.assertTrue(module.weight.device == torch.device("cpu"))
            elif "adapter-2" in name and not isinstance(module, (nn.Dropout, nn.Identity)):
                self.assertTrue(module.weight.device != torch.device("cpu"))

        pipe.set_lora_device(["adapter-1"], 0)

        for n, m in pipe.unet.named_modules():
            if "adapter-1" in n and not isinstance(m, (nn.Dropout, nn.Identity)):
                self.assertTrue(m.weight.device != torch.device("cpu"))

        for n, m in pipe.text_encoder.named_modules():
            if "adapter-1" in n and not isinstance(m, (nn.Dropout, nn.Identity)):
                self.assertTrue(m.weight.device != torch.device("cpu"))

        pipe.set_lora_device(["adapter-1", "adapter-2"], torch_device)

        for n, m in pipe.unet.named_modules():
            if ("adapter-1" in n or "adapter-2" in n) and not isinstance(m, (nn.Dropout, nn.Identity)):
                self.assertTrue(m.weight.device != torch.device("cpu"))

        for n, m in pipe.text_encoder.named_modules():
            if ("adapter-1" in n or "adapter-2" in n) and not isinstance(m, (nn.Dropout, nn.Identity)):
                self.assertTrue(m.weight.device != torch.device("cpu"))

    @slow
    @require_torch_gpu
    def test_integration_move_lora_dora_cpu(self):
        from peft import LoraConfig

        path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        unet_lora_config = LoraConfig(
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            use_dora=True,
        )
        text_lora_config = LoraConfig(
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            use_dora=True,
        )

        pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16)
        pipe.unet.add_adapter(unet_lora_config, "adapter-1")
        pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")

        self.assertTrue(
            check_if_lora_correctly_set(pipe.text_encoder),
            "Lora not correctly set in text encoder",
        )

        self.assertTrue(
            check_if_lora_correctly_set(pipe.unet),
            "Lora not correctly set in text encoder",
        )

        for name, param in pipe.unet.named_parameters():
            if "lora_" in name:
                self.assertEqual(param.device, torch.device("cpu"))

        for name, param in pipe.text_encoder.named_parameters():
            if "lora_" in name:
                self.assertEqual(param.device, torch.device("cpu"))

        pipe.set_lora_device(["adapter-1"], torch_device)

        for name, param in pipe.unet.named_parameters():
            if "lora_" in name:
                self.assertNotEqual(param.device, torch.device("cpu"))

        for name, param in pipe.text_encoder.named_parameters():
            if "lora_" in name:
                self.assertNotEqual(param.device, torch.device("cpu"))


@slow
@nightly
@require_torch_gpu
@require_peft_backend
class LoraIntegrationTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_integration_logits_with_scale(self):
        path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        lora_id = "takuma104/lora-test-text-encoder-lora-target"

        pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float32)
        pipe.load_lora_weights(lora_id)
        pipe = pipe.to(torch_device)

        self.assertTrue(
            check_if_lora_correctly_set(pipe.text_encoder),
            "Lora not correctly set in text encoder",
        )

        prompt = "a red sks dog"

        images = pipe(
            prompt=prompt,
            num_inference_steps=15,
            cross_attention_kwargs={"scale": 0.5},
            generator=torch.manual_seed(0),
            output_type="np",
        ).images

        expected_slice_scale = np.array([0.307, 0.283, 0.310, 0.310, 0.300, 0.314, 0.336, 0.314, 0.321])
        predicted_slice = images[0, -3:, -3:, -1].flatten()

        max_diff = numpy_cosine_similarity_distance(expected_slice_scale, predicted_slice)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_integration_logits_no_scale(self):
        path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        lora_id = "takuma104/lora-test-text-encoder-lora-target"

        pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float32)
        pipe.load_lora_weights(lora_id)
        pipe = pipe.to(torch_device)

        self.assertTrue(
            check_if_lora_correctly_set(pipe.text_encoder),
            "Lora not correctly set in text encoder",
        )

        prompt = "a red sks dog"

        images = pipe(prompt=prompt, num_inference_steps=30, generator=torch.manual_seed(0), output_type="np").images

        expected_slice_scale = np.array([0.074, 0.064, 0.073, 0.0842, 0.069, 0.0641, 0.0794, 0.076, 0.084])
        predicted_slice = images[0, -3:, -3:, -1].flatten()

        max_diff = numpy_cosine_similarity_distance(expected_slice_scale, predicted_slice)

        assert max_diff < 1e-3

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_dreambooth_old_format(self):
        generator = torch.Generator("cpu").manual_seed(0)

        lora_model_id = "hf-internal-testing/lora_dreambooth_dog_example"

        base_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

        pipe = StableDiffusionPipeline.from_pretrained(base_model_id, safety_checker=None)
        pipe = pipe.to(torch_device)
        pipe.load_lora_weights(lora_model_id)

        images = pipe(
            "A photo of a sks dog floating in the river", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.7207, 0.6787, 0.6010, 0.7478, 0.6838, 0.6064, 0.6984, 0.6443, 0.5785])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-4

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_dreambooth_text_encoder_new_format(self):
        generator = torch.Generator().manual_seed(0)

        lora_model_id = "hf-internal-testing/lora-trained"

        base_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

        pipe = StableDiffusionPipeline.from_pretrained(base_model_id, safety_checker=None)
        pipe = pipe.to(torch_device)
        pipe.load_lora_weights(lora_model_id)

        images = pipe("A photo of a sks dog", output_type="np", generator=generator, num_inference_steps=2).images

        images = images[0, -3:, -3:, -1].flatten()

        expected = np.array([0.6628, 0.6138, 0.5390, 0.6625, 0.6130, 0.5463, 0.6166, 0.5788, 0.5359])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-4

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_a1111(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained("hf-internal-testing/Counterfeit-V2.5", safety_checker=None).to(
            torch_device
        )
        lora_model_id = "hf-internal-testing/civitai-light-shadow-lora"
        lora_filename = "light_and_shadow.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.3636, 0.3708, 0.3694, 0.3679, 0.3829, 0.3677, 0.3692, 0.3688, 0.3292])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_lycoris(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/Amixx", safety_checker=None, use_safetensors=True, variant="fp16"
        ).to(torch_device)
        lora_model_id = "hf-internal-testing/edgLycorisMugler-light"
        lora_filename = "edgLycorisMugler-light.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.6463, 0.658, 0.599, 0.6542, 0.6512, 0.6213, 0.658, 0.6485, 0.6017])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_a1111_with_model_cpu_offload(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained("hf-internal-testing/Counterfeit-V2.5", safety_checker=None)
        pipe.enable_model_cpu_offload()
        lora_model_id = "hf-internal-testing/civitai-light-shadow-lora"
        lora_filename = "light_and_shadow.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.3636, 0.3708, 0.3694, 0.3679, 0.3829, 0.3677, 0.3692, 0.3688, 0.3292])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_a1111_with_sequential_cpu_offload(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained("hf-internal-testing/Counterfeit-V2.5", safety_checker=None)
        pipe.enable_sequential_cpu_offload()
        lora_model_id = "hf-internal-testing/civitai-light-shadow-lora"
        lora_filename = "light_and_shadow.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.3636, 0.3708, 0.3694, 0.3679, 0.3829, 0.3677, 0.3692, 0.3688, 0.3292])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_kohya_sd_v15_with_higher_dimensions(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None
        ).to(torch_device)
        lora_model_id = "hf-internal-testing/urushisato-lora"
        lora_filename = "urushisato_v15.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.7165, 0.6616, 0.5833, 0.7504, 0.6718, 0.587, 0.6871, 0.6361, 0.5694])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_vanilla_funetuning(self):
        generator = torch.Generator().manual_seed(0)

        lora_model_id = "hf-internal-testing/sd-model-finetuned-lora-t4"

        base_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

        pipe = StableDiffusionPipeline.from_pretrained(base_model_id, safety_checker=None)
        pipe = pipe.to(torch_device)
        pipe.load_lora_weights(lora_model_id)

        images = pipe("A pokemon with blue eyes.", output_type="np", generator=generator, num_inference_steps=2).images

        images = images[0, -3:, -3:, -1].flatten()

        expected = np.array([0.7406, 0.699, 0.5963, 0.7493, 0.7045, 0.6096, 0.6886, 0.6388, 0.583])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-4

        pipe.unload_lora_weights()
        release_memory(pipe)

    def test_unload_kohya_lora(self):
        generator = torch.manual_seed(0)
        prompt = "masterpiece, best quality, mountain"
        num_inference_steps = 2

        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None
        ).to(torch_device)
        initial_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        initial_images = initial_images[0, -3:, -3:, -1].flatten()

        lora_model_id = "hf-internal-testing/civitai-colored-icons-lora"
        lora_filename = "Colored_Icons_by_vizsumit.safetensors"

        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        generator = torch.manual_seed(0)
        lora_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        lora_images = lora_images[0, -3:, -3:, -1].flatten()

        pipe.unload_lora_weights()
        generator = torch.manual_seed(0)
        unloaded_lora_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        unloaded_lora_images = unloaded_lora_images[0, -3:, -3:, -1].flatten()

        self.assertFalse(np.allclose(initial_images, lora_images))
        self.assertTrue(np.allclose(initial_images, unloaded_lora_images, atol=1e-3))

        release_memory(pipe)

    def test_load_unload_load_kohya_lora(self):
        # This test ensures that a Kohya-style LoRA can be safely unloaded and then loaded
        # without introducing any side-effects. Even though the test uses a Kohya-style
        # LoRA, the underlying adapter handling mechanism is format-agnostic.
        generator = torch.manual_seed(0)
        prompt = "masterpiece, best quality, mountain"
        num_inference_steps = 2

        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None
        ).to(torch_device)
        initial_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        initial_images = initial_images[0, -3:, -3:, -1].flatten()

        lora_model_id = "hf-internal-testing/civitai-colored-icons-lora"
        lora_filename = "Colored_Icons_by_vizsumit.safetensors"

        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        generator = torch.manual_seed(0)
        lora_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        lora_images = lora_images[0, -3:, -3:, -1].flatten()

        pipe.unload_lora_weights()
        generator = torch.manual_seed(0)
        unloaded_lora_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        unloaded_lora_images = unloaded_lora_images[0, -3:, -3:, -1].flatten()

        self.assertFalse(np.allclose(initial_images, lora_images))
        self.assertTrue(np.allclose(initial_images, unloaded_lora_images, atol=1e-3))

        # make sure we can load a LoRA again after unloading and they don't have
        # any undesired effects.
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        generator = torch.manual_seed(0)
        lora_images_again = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        lora_images_again = lora_images_again[0, -3:, -3:, -1].flatten()

        self.assertTrue(np.allclose(lora_images, lora_images_again, atol=1e-3))
        release_memory(pipe)

    def test_not_empty_state_dict(self):
        # Makes sure https://github.com/huggingface/diffusers/issues/7054 does not happen again
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to(torch_device)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        cached_file = hf_hub_download("hf-internal-testing/lcm-lora-test-sd-v1-5", "test_lora.safetensors")
        lcm_lora = load_file(cached_file)

        pipe.load_lora_weights(lcm_lora, adapter_name="lcm")
        self.assertTrue(lcm_lora != {})
        release_memory(pipe)

    def test_load_unload_load_state_dict(self):
        # Makes sure https://github.com/huggingface/diffusers/issues/7054 does not happen again
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to(torch_device)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        cached_file = hf_hub_download("hf-internal-testing/lcm-lora-test-sd-v1-5", "test_lora.safetensors")
        lcm_lora = load_file(cached_file)
        previous_state_dict = lcm_lora.copy()

        pipe.load_lora_weights(lcm_lora, adapter_name="lcm")
        self.assertDictEqual(lcm_lora, previous_state_dict)

        pipe.unload_lora_weights()
        pipe.load_lora_weights(lcm_lora, adapter_name="lcm")
        self.assertDictEqual(lcm_lora, previous_state_dict)

        release_memory(pipe)

    def test_sdv1_5_lcm_lora(self):
        pipe = DiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        pipe.to(torch_device)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        generator = torch.Generator("cpu").manual_seed(0)

        lora_model_id = "latent-consistency/lcm-lora-sdv1-5"
        pipe.load_lora_weights(lora_model_id)

        image = pipe(
            "masterpiece, best quality, mountain", generator=generator, num_inference_steps=4, guidance_scale=0.5
        ).images[0]

        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/lcm_lora/sdv15_lcm_lora.png"
        )

        image_np = pipe.image_processor.pil_to_numpy(image)
        expected_image_np = pipe.image_processor.pil_to_numpy(expected_image)

        max_diff = numpy_cosine_similarity_distance(image_np.flatten(), expected_image_np.flatten())
        assert max_diff < 1e-4

        pipe.unload_lora_weights()

        release_memory(pipe)

    def test_sdv1_5_lcm_lora_img2img(self):
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        pipe.to(torch_device)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/fantasy_landscape.png"
        )

        generator = torch.Generator("cpu").manual_seed(0)

        lora_model_id = "latent-consistency/lcm-lora-sdv1-5"
        pipe.load_lora_weights(lora_model_id)

        image = pipe(
            "snowy mountain",
            generator=generator,
            image=init_image,
            strength=0.5,
            num_inference_steps=4,
            guidance_scale=0.5,
        ).images[0]

        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/lcm_lora/sdv15_lcm_lora_img2img.png"
        )

        image_np = pipe.image_processor.pil_to_numpy(image)
        expected_image_np = pipe.image_processor.pil_to_numpy(expected_image)

        max_diff = numpy_cosine_similarity_distance(image_np.flatten(), expected_image_np.flatten())
        assert max_diff < 1e-4

        pipe.unload_lora_weights()

        release_memory(pipe)

    def test_sd_load_civitai_empty_network_alpha(self):
        """
        This test simply checks that loading a LoRA with an empty network alpha works fine
        See: https://github.com/huggingface/diffusers/issues/5606
        """
        pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
        pipeline.enable_sequential_cpu_offload()
        civitai_path = hf_hub_download("ybelkada/test-ahi-civitai", "ahi_lora_weights.safetensors")
        pipeline.load_lora_weights(civitai_path, adapter_name="ahri")

        images = pipeline(
            "ahri, masterpiece, league of legends",
            output_type="np",
            generator=torch.manual_seed(156),
            num_inference_steps=5,
        ).images
        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.0, 0.0, 0.0, 0.002557, 0.020954, 0.001792, 0.006581, 0.00591, 0.002995])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipeline.unload_lora_weights()
        release_memory(pipeline)
