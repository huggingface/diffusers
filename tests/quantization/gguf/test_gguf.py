import gc
import unittest

import numpy as np
import torch
import torch.nn as nn

from diffusers import (
    AuraFlowPipeline,
    AuraFlowTransformer2DModel,
    DiffusionPipeline,
    FluxControlPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
    HiDreamImageTransformer2DModel,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
    WanTransformer3DModel,
    WanVACETransformer3DModel,
)
from diffusers.utils import load_image

from ...testing_utils import (
    Expectations,
    backend_empty_cache,
    backend_max_memory_allocated,
    backend_reset_peak_memory_stats,
    enable_full_determinism,
    is_gguf_available,
    nightly,
    numpy_cosine_similarity_distance,
    require_accelerate,
    require_accelerator,
    require_big_accelerator,
    require_gguf_version_greater_or_equal,
    require_kernels_version_greater_or_equal,
    require_peft_backend,
    require_torch_version_greater,
    torch_device,
)
from ..test_torch_compile_utils import QuantCompileTests


if is_gguf_available():
    import gguf

    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter

enable_full_determinism()


@nightly
@require_accelerate
@require_accelerator
@require_gguf_version_greater_or_equal("0.10.0")
@require_kernels_version_greater_or_equal("0.9.0")
class GGUFCudaKernelsTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def test_cuda_kernels_vs_native(self):
        if torch_device != "cuda":
            self.skipTest("CUDA kernels test requires CUDA device")

        from diffusers.quantizers.gguf.utils import GGUFLinear, can_use_cuda_kernels

        if not can_use_cuda_kernels:
            self.skipTest("CUDA kernels not available (compute capability < 7 or kernels not installed)")

        test_quant_types = ["Q4_0", "Q4_K"]
        test_shape = (1, 64, 512)  # batch, seq_len, hidden_dim
        compute_dtype = torch.bfloat16

        for quant_type in test_quant_types:
            qtype = getattr(gguf.GGMLQuantizationType, quant_type)
            in_features, out_features = 512, 512

            torch.manual_seed(42)
            float_weight = torch.randn(out_features, in_features, dtype=torch.float32)
            quantized_data = gguf.quants.quantize(float_weight.numpy(), qtype)
            weight_data = torch.from_numpy(quantized_data).to(device=torch_device)
            weight = GGUFParameter(weight_data, quant_type=qtype)

            x = torch.randn(test_shape, dtype=compute_dtype, device=torch_device)

            linear = GGUFLinear(in_features, out_features, bias=True, compute_dtype=compute_dtype)
            linear.weight = weight
            linear.bias = nn.Parameter(torch.randn(out_features, dtype=compute_dtype))
            linear = linear.to(torch_device)

            with torch.no_grad():
                output_native = linear.forward_native(x)
                output_cuda = linear.forward_cuda(x)

            assert torch.allclose(output_native, output_cuda, 1e-2), (
                f"GGUF CUDA Kernel Output is different from Native Output for {quant_type}"
            )


@nightly
@require_big_accelerator
@require_accelerate
@require_gguf_version_greater_or_equal("0.10.0")
class GGUFSingleFileTesterMixin:
    ckpt_path = None
    model_cls = None
    torch_dtype = torch.bfloat16
    expected_memory_use_in_gb = 5

    def test_gguf_parameters(self):
        quant_storage_type = torch.uint8
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        model = self.model_cls.from_single_file(self.ckpt_path, quantization_config=quantization_config)

        for param_name, param in model.named_parameters():
            if isinstance(param, GGUFParameter):
                assert hasattr(param, "quant_type")
                assert param.dtype == quant_storage_type

    def test_gguf_linear_layers(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        model = self.model_cls.from_single_file(self.ckpt_path, quantization_config=quantization_config)

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and hasattr(module.weight, "quant_type"):
                assert module.weight.dtype == torch.uint8
                if module.bias is not None:
                    assert module.bias.dtype == self.torch_dtype

    def test_gguf_memory_usage(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)

        model = self.model_cls.from_single_file(
            self.ckpt_path, quantization_config=quantization_config, torch_dtype=self.torch_dtype
        )
        model.to(torch_device)
        assert (model.get_memory_footprint() / 1024**3) < self.expected_memory_use_in_gb
        inputs = self.get_dummy_inputs()

        backend_reset_peak_memory_stats(torch_device)
        backend_empty_cache(torch_device)
        with torch.no_grad():
            model(**inputs)
        max_memory = backend_max_memory_allocated(torch_device)
        assert (max_memory / 1024**3) < self.expected_memory_use_in_gb

    def test_keep_modules_in_fp32(self):
        r"""
        A simple tests to check if the modules under `_keep_in_fp32_modules` are kept in fp32.
        Also ensures if inference works.
        """
        _keep_in_fp32_modules = self.model_cls._keep_in_fp32_modules
        self.model_cls._keep_in_fp32_modules = ["proj_out"]

        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        model = self.model_cls.from_single_file(self.ckpt_path, quantization_config=quantization_config)

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name in model._keep_in_fp32_modules:
                    assert module.weight.dtype == torch.float32
        self.model_cls._keep_in_fp32_modules = _keep_in_fp32_modules

    def test_dtype_assignment(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        model = self.model_cls.from_single_file(self.ckpt_path, quantization_config=quantization_config)

        with self.assertRaises(ValueError):
            # Tries with a `dtype`
            model.to(torch.float16)

        with self.assertRaises(ValueError):
            # Tries with a `device` and `dtype`
            device_0 = f"{torch_device}:0"
            model.to(device=device_0, dtype=torch.float16)

        with self.assertRaises(ValueError):
            # Tries with a cast
            model.float()

        with self.assertRaises(ValueError):
            # Tries with a cast
            model.half()

        # This should work
        model.to(torch_device)

    def test_dequantize_model(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        model = self.model_cls.from_single_file(self.ckpt_path, quantization_config=quantization_config)
        model.dequantize()

        def _check_for_gguf_linear(model):
            has_children = list(model.children())
            if not has_children:
                return

            for name, module in model.named_children():
                if isinstance(module, nn.Linear):
                    assert not isinstance(module, GGUFLinear), f"{name} is still GGUFLinear"
                    assert not isinstance(module.weight, GGUFParameter), f"{name} weight is still GGUFParameter"

        for name, module in model.named_children():
            _check_for_gguf_linear(module)


class FluxGGUFSingleFileTests(GGUFSingleFileTesterMixin, unittest.TestCase):
    ckpt_path = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q2_K.gguf"
    diffusers_ckpt_path = "https://huggingface.co/sayakpaul/flux-diffusers-gguf/blob/main/model-Q4_0.gguf"
    torch_dtype = torch.bfloat16
    model_cls = FluxTransformer2DModel
    expected_memory_use_in_gb = 5

    def setUp(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def get_dummy_inputs(self):
        return {
            "hidden_states": torch.randn((1, 4096, 64), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "encoder_hidden_states": torch.randn(
                (1, 512, 4096),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "pooled_projections": torch.randn(
                (1, 768),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "timestep": torch.tensor([1]).to(torch_device, self.torch_dtype),
            "img_ids": torch.randn((4096, 3), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "txt_ids": torch.randn((512, 3), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "guidance": torch.tensor([3.5]).to(torch_device, self.torch_dtype),
        }

    def test_pipeline_inference(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        transformer = self.model_cls.from_single_file(
            self.ckpt_path, quantization_config=quantization_config, torch_dtype=self.torch_dtype
        )
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=self.torch_dtype
        )
        pipe.enable_model_cpu_offload()

        prompt = "a cat holding a sign that says hello"
        output = pipe(
            prompt=prompt, num_inference_steps=2, generator=torch.Generator("cpu").manual_seed(0), output_type="np"
        ).images[0]
        output_slice = output[:3, :3, :].flatten()
        expected_slice = np.array(
            [
                0.47265625,
                0.43359375,
                0.359375,
                0.47070312,
                0.421875,
                0.34375,
                0.46875,
                0.421875,
                0.34765625,
                0.46484375,
                0.421875,
                0.34179688,
                0.47070312,
                0.42578125,
                0.34570312,
                0.46875,
                0.42578125,
                0.3515625,
                0.45507812,
                0.4140625,
                0.33984375,
                0.4609375,
                0.41796875,
                0.34375,
                0.45898438,
                0.41796875,
                0.34375,
            ]
        )
        max_diff = numpy_cosine_similarity_distance(expected_slice, output_slice)
        assert max_diff < 1e-4

    def test_loading_gguf_diffusers_format(self):
        model = self.model_cls.from_single_file(
            self.diffusers_ckpt_path,
            subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            config="black-forest-labs/FLUX.1-dev",
        )
        model.to(torch_device)
        model(**self.get_dummy_inputs())


class SD35LargeGGUFSingleFileTests(GGUFSingleFileTesterMixin, unittest.TestCase):
    ckpt_path = "https://huggingface.co/city96/stable-diffusion-3.5-large-gguf/blob/main/sd3.5_large-Q4_0.gguf"
    torch_dtype = torch.bfloat16
    model_cls = SD3Transformer2DModel
    expected_memory_use_in_gb = 5

    def setUp(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def get_dummy_inputs(self):
        return {
            "hidden_states": torch.randn((1, 16, 64, 64), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "encoder_hidden_states": torch.randn(
                (1, 512, 4096),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "pooled_projections": torch.randn(
                (1, 2048),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "timestep": torch.tensor([1]).to(torch_device, self.torch_dtype),
        }

    def test_pipeline_inference(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        transformer = self.model_cls.from_single_file(
            self.ckpt_path, quantization_config=quantization_config, torch_dtype=self.torch_dtype
        )
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large", transformer=transformer, torch_dtype=self.torch_dtype
        )
        pipe.enable_model_cpu_offload()

        prompt = "a cat holding a sign that says hello"
        output = pipe(
            prompt=prompt,
            num_inference_steps=2,
            generator=torch.Generator("cpu").manual_seed(0),
            output_type="np",
        ).images[0]
        output_slice = output[:3, :3, :].flatten()
        expected_slices = Expectations(
            {
                ("xpu", 3): np.array(
                    [
                        0.16796875,
                        0.27929688,
                        0.28320312,
                        0.11328125,
                        0.27539062,
                        0.26171875,
                        0.10742188,
                        0.26367188,
                        0.26171875,
                        0.1484375,
                        0.2734375,
                        0.296875,
                        0.13476562,
                        0.2890625,
                        0.30078125,
                        0.1171875,
                        0.28125,
                        0.28125,
                        0.16015625,
                        0.31445312,
                        0.30078125,
                        0.15625,
                        0.32421875,
                        0.296875,
                        0.14453125,
                        0.30859375,
                        0.2890625,
                    ]
                ),
                ("cuda", 7): np.array(
                    [
                        0.17578125,
                        0.27539062,
                        0.27734375,
                        0.11914062,
                        0.26953125,
                        0.25390625,
                        0.109375,
                        0.25390625,
                        0.25,
                        0.15039062,
                        0.26171875,
                        0.28515625,
                        0.13671875,
                        0.27734375,
                        0.28515625,
                        0.12109375,
                        0.26757812,
                        0.265625,
                        0.16210938,
                        0.29882812,
                        0.28515625,
                        0.15625,
                        0.30664062,
                        0.27734375,
                        0.14648438,
                        0.29296875,
                        0.26953125,
                    ]
                ),
            }
        )
        expected_slice = expected_slices.get_expectation()
        max_diff = numpy_cosine_similarity_distance(expected_slice, output_slice)
        assert max_diff < 1e-4


class SD35MediumGGUFSingleFileTests(GGUFSingleFileTesterMixin, unittest.TestCase):
    ckpt_path = "https://huggingface.co/city96/stable-diffusion-3.5-medium-gguf/blob/main/sd3.5_medium-Q3_K_M.gguf"
    torch_dtype = torch.bfloat16
    model_cls = SD3Transformer2DModel
    expected_memory_use_in_gb = 2

    def setUp(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def get_dummy_inputs(self):
        return {
            "hidden_states": torch.randn((1, 16, 64, 64), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "encoder_hidden_states": torch.randn(
                (1, 512, 4096),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "pooled_projections": torch.randn(
                (1, 2048),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "timestep": torch.tensor([1]).to(torch_device, self.torch_dtype),
        }

    def test_pipeline_inference(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        transformer = self.model_cls.from_single_file(
            self.ckpt_path, quantization_config=quantization_config, torch_dtype=self.torch_dtype
        )
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium", transformer=transformer, torch_dtype=self.torch_dtype
        )
        pipe.enable_model_cpu_offload()

        prompt = "a cat holding a sign that says hello"
        output = pipe(
            prompt=prompt, num_inference_steps=2, generator=torch.Generator("cpu").manual_seed(0), output_type="np"
        ).images[0]
        output_slice = output[:3, :3, :].flatten()
        expected_slice = np.array(
            [
                0.625,
                0.6171875,
                0.609375,
                0.65625,
                0.65234375,
                0.640625,
                0.6484375,
                0.640625,
                0.625,
                0.6484375,
                0.63671875,
                0.6484375,
                0.66796875,
                0.65625,
                0.65234375,
                0.6640625,
                0.6484375,
                0.6328125,
                0.6640625,
                0.6484375,
                0.640625,
                0.67578125,
                0.66015625,
                0.62109375,
                0.671875,
                0.65625,
                0.62109375,
            ]
        )
        max_diff = numpy_cosine_similarity_distance(expected_slice, output_slice)
        assert max_diff < 1e-4


class AuraFlowGGUFSingleFileTests(GGUFSingleFileTesterMixin, unittest.TestCase):
    ckpt_path = "https://huggingface.co/city96/AuraFlow-v0.3-gguf/blob/main/aura_flow_0.3-Q2_K.gguf"
    torch_dtype = torch.bfloat16
    model_cls = AuraFlowTransformer2DModel
    expected_memory_use_in_gb = 4

    def setUp(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def get_dummy_inputs(self):
        return {
            "hidden_states": torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "encoder_hidden_states": torch.randn(
                (1, 512, 2048),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "timestep": torch.tensor([1]).to(torch_device, self.torch_dtype),
        }

    def test_pipeline_inference(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        transformer = self.model_cls.from_single_file(
            self.ckpt_path, quantization_config=quantization_config, torch_dtype=self.torch_dtype
        )
        pipe = AuraFlowPipeline.from_pretrained(
            "fal/AuraFlow-v0.3", transformer=transformer, torch_dtype=self.torch_dtype
        )
        pipe.enable_model_cpu_offload()

        prompt = "a pony holding a sign that says hello"
        output = pipe(
            prompt=prompt, num_inference_steps=2, generator=torch.Generator("cpu").manual_seed(0), output_type="np"
        ).images[0]
        output_slice = output[:3, :3, :].flatten()
        expected_slice = np.array(
            [
                0.46484375,
                0.546875,
                0.64453125,
                0.48242188,
                0.53515625,
                0.59765625,
                0.47070312,
                0.5078125,
                0.5703125,
                0.42773438,
                0.50390625,
                0.5703125,
                0.47070312,
                0.515625,
                0.57421875,
                0.45898438,
                0.48632812,
                0.53515625,
                0.4453125,
                0.5078125,
                0.56640625,
                0.47851562,
                0.5234375,
                0.57421875,
                0.48632812,
                0.5234375,
                0.56640625,
            ]
        )
        max_diff = numpy_cosine_similarity_distance(expected_slice, output_slice)
        assert max_diff < 1e-4


@require_peft_backend
@nightly
@require_big_accelerator
@require_accelerate
@require_gguf_version_greater_or_equal("0.10.0")
class FluxControlLoRAGGUFTests(unittest.TestCase):
    def test_lora_loading(self):
        ckpt_path = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q2_K.gguf"
        transformer = FluxTransformer2DModel.from_single_file(
            ckpt_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        pipe = FluxControlPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        ).to(torch_device)
        pipe.load_lora_weights("black-forest-labs/FLUX.1-Canny-dev-lora")

        prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."
        control_image = load_image(
            "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/control_image_robot_canny.png"
        )

        output = pipe(
            prompt=prompt,
            control_image=control_image,
            height=256,
            width=256,
            num_inference_steps=10,
            guidance_scale=30.0,
            output_type="np",
            generator=torch.manual_seed(0),
        ).images

        out_slice = output[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.8047, 0.8359, 0.8711, 0.6875, 0.7070, 0.7383, 0.5469, 0.5820, 0.6641])

        max_diff = numpy_cosine_similarity_distance(expected_slice, out_slice)
        self.assertTrue(max_diff < 1e-3)


class HiDreamGGUFSingleFileTests(GGUFSingleFileTesterMixin, unittest.TestCase):
    ckpt_path = "https://huggingface.co/city96/HiDream-I1-Dev-gguf/blob/main/hidream-i1-dev-Q2_K.gguf"
    torch_dtype = torch.bfloat16
    model_cls = HiDreamImageTransformer2DModel
    expected_memory_use_in_gb = 8

    def get_dummy_inputs(self):
        return {
            "hidden_states": torch.randn((1, 16, 128, 128), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "encoder_hidden_states_t5": torch.randn(
                (1, 128, 4096),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "encoder_hidden_states_llama3": torch.randn(
                (32, 1, 128, 4096),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "pooled_embeds": torch.randn(
                (1, 2048),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "timesteps": torch.tensor([1]).to(torch_device, self.torch_dtype),
        }


class WanGGUFTexttoVideoSingleFileTests(GGUFSingleFileTesterMixin, unittest.TestCase):
    ckpt_path = "https://huggingface.co/city96/Wan2.1-T2V-14B-gguf/blob/main/wan2.1-t2v-14b-Q3_K_S.gguf"
    torch_dtype = torch.bfloat16
    model_cls = WanTransformer3DModel
    expected_memory_use_in_gb = 9

    def get_dummy_inputs(self):
        return {
            "hidden_states": torch.randn((1, 16, 2, 64, 64), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "encoder_hidden_states": torch.randn(
                (1, 512, 4096),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "timestep": torch.tensor([1]).to(torch_device, self.torch_dtype),
        }


class WanGGUFImagetoVideoSingleFileTests(GGUFSingleFileTesterMixin, unittest.TestCase):
    ckpt_path = "https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/blob/main/wan2.1-i2v-14b-480p-Q3_K_S.gguf"
    torch_dtype = torch.bfloat16
    model_cls = WanTransformer3DModel
    expected_memory_use_in_gb = 9

    def get_dummy_inputs(self):
        return {
            "hidden_states": torch.randn((1, 36, 2, 64, 64), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "encoder_hidden_states": torch.randn(
                (1, 512, 4096),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "encoder_hidden_states_image": torch.randn(
                (1, 257, 1280), generator=torch.Generator("cpu").manual_seed(0)
            ).to(torch_device, self.torch_dtype),
            "timestep": torch.tensor([1]).to(torch_device, self.torch_dtype),
        }


class WanVACEGGUFSingleFileTests(GGUFSingleFileTesterMixin, unittest.TestCase):
    ckpt_path = "https://huggingface.co/QuantStack/Wan2.1_14B_VACE-GGUF/blob/main/Wan2.1_14B_VACE-Q3_K_S.gguf"
    torch_dtype = torch.bfloat16
    model_cls = WanVACETransformer3DModel
    expected_memory_use_in_gb = 9

    def get_dummy_inputs(self):
        return {
            "hidden_states": torch.randn((1, 16, 2, 64, 64), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "encoder_hidden_states": torch.randn(
                (1, 512, 4096),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "control_hidden_states": torch.randn(
                (1, 96, 2, 64, 64),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "control_hidden_states_scale": torch.randn(
                (8,),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "timestep": torch.tensor([1]).to(torch_device, self.torch_dtype),
        }


@require_torch_version_greater("2.7.1")
class GGUFCompileTests(QuantCompileTests, unittest.TestCase):
    torch_dtype = torch.bfloat16
    gguf_ckpt = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q2_K.gguf"

    @property
    def quantization_config(self):
        return GGUFQuantizationConfig(compute_dtype=self.torch_dtype)

    def _init_pipeline(self, *args, **kwargs):
        transformer = FluxTransformer2DModel.from_single_file(
            self.gguf_ckpt, quantization_config=self.quantization_config, torch_dtype=self.torch_dtype
        )
        pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=self.torch_dtype
        )
        return pipe
