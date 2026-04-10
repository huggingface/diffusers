import gc
import tempfile
import unittest

from diffusers import AutoRoundConfig, FluxTransformer2DModel, FluxPipeline
from diffusers.utils import is_auto_round_available, is_torch_available
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    backend_reset_peak_memory_stats,
    enable_full_determinism,
    nightly,
    numpy_cosine_similarity_distance,
    require_accelerate,
    require_big_accelerator,
    require_auto_round_version_greater_or_equal,
    require_torch_cuda_compatibility,
    torch_device,
)


if is_torch_available():
    import torch

    from ..utils import get_memory_consumption_stat


def _is_gptqmodel_available(min_version="5.8.0"):
    """Check if gptqmodel is installed with a minimum version."""
    try:
        import importlib.metadata

        from packaging import version

        gptqmodel_version = importlib.metadata.version("gptqmodel")
        return version.parse(gptqmodel_version) >= version.parse(min_version)
    except importlib.metadata.PackageNotFoundError:
        return False


enable_full_determinism()


@nightly
@require_big_accelerator
@require_accelerate
@require_auto_round_version_greater_or_equal("0.13.0")
class AutoRoundBaseTesterMixin:
    """Base test mixin for AutoRound quantized models.

    AutoRound is a weight-only quantization method (W4A16). It supports multiple inference
    backends depending on the hardware:
    - CPU:  `auto_round:torch_zp` backend
    - CUDA: `auto_round:tritonv2_zp` backend
    - CUDA + GPTQModel>=5.8.0: `gptqmodel:marlin_zp` backend (best performance)

    When `backend="auto"`, AutoRound selects the best available backend automatically.

    Key differences from ModelOpt tests:
    - Only pre-quantized model loading is supported (no on-the-fly quantization).
    - `is_trainable` returns False, so no LoRA training test.
    - No `test_dtype_assignment` (AutoRound doesn't restrict dtype changes).
    - `requires_calibration = True` means we always load pre-quantized checkpoints.
    """

    # TODO: Replace with a real tiny AutoRound-quantized checkpoint on the Hub.
    # This should be a small model that has been quantized with AutoRound and uploaded
    # in the standard format (qweight, scales, qzeros, g_idx).
    model_id = "hf-internal-testing/tiny-flux-pipe-autoround-w4g128"
    model_cls = FluxTransformer2DModel
    pipeline_cls = FluxPipeline
    torch_dtype = torch.float16
    expected_memory_reduction = 0.0
    modules_to_not_convert = ""
    _test_torch_compile = False

    def setUp(self):
        backend_reset_peak_memory_stats(torch_device)
        backend_empty_cache(torch_device)
        gc.collect()

    def tearDown(self):
        backend_reset_peak_memory_stats(torch_device)
        backend_empty_cache(torch_device)
        gc.collect()

    def get_dummy_init_kwargs(self):
        """Returns the default AutoRoundConfig kwargs for W4A16 quantization.

        Subclasses override this to specify backend, group_size, sym, etc.
        """
        return {
            "bits": 4,
            "group_size": 128,
            "sym": False,
        }

    def get_dummy_model_init_kwargs(self):
        """Returns kwargs for model_cls.from_pretrained() with AutoRound quantization."""
        return {
            "pretrained_model_name_or_path": self.model_id,
            "torch_dtype": self.torch_dtype,
            "quantization_config": AutoRoundConfig(**self.get_dummy_init_kwargs()),
            "subfolder": "transformer",
        }

    def get_dummy_inputs(self):
        """Creates dummy inputs matching the model's expected forward signature.

        TODO: Adjust input shapes to match the tiny test checkpoint's config.
        """
        batch_size = 1
        seq_len = 16
        height = width = 32
        num_latent_channels = 4
        caption_channels = 8

        torch.manual_seed(0)
        hidden_states = torch.randn((batch_size, num_latent_channels, height, width)).to(
            torch_device, dtype=self.torch_dtype
        )
        encoder_hidden_states = torch.randn((batch_size, seq_len, caption_channels)).to(
            torch_device, dtype=self.torch_dtype
        )
        timestep = torch.tensor([1.0]).to(torch_device, dtype=self.torch_dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
        }

    def test_autoround_layers(self):
        """Verify that eligible nn.Linear layers have been replaced with AutoRound QuantLinear layers.

        After loading a pre-quantized model, all target linear layers should have been
        converted to AutoRound's QuantLinear by `convert_hf_model`.
        """
        from auto_round.inference.convert_model import QuantLinear

        model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())
        has_quantized_layer = False
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                has_quantized_layer = True
        assert has_quantized_layer, "No QuantLinear layers found in the model — quantization may have failed."

    def test_autoround_memory_usage(self):
        """Compare peak memory between unquantized and AutoRound-quantized model.

        The quantized model should use significantly less memory due to 4-bit weight packing.
        `expected_memory_reduction` defines the minimum ratio (unquantized / quantized).
        """
        inputs = self.get_dummy_inputs()
        inputs = {
            k: v.to(device=torch_device, dtype=self.torch_dtype) for k, v in inputs.items() if not isinstance(v, bool)
        }

        unquantized_model = self.model_cls.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, subfolder="transformer"
        )
        unquantized_model.to(torch_device)
        unquantized_model_memory = get_memory_consumption_stat(unquantized_model, inputs)

        quantized_model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())
        quantized_model.to(torch_device)
        quantized_model_memory = get_memory_consumption_stat(quantized_model, inputs)

        assert unquantized_model_memory / quantized_model_memory >= self.expected_memory_reduction

    def test_modules_to_not_convert(self):
        """Verify that modules listed in `modules_to_not_convert` remain as standard nn.Linear."""
        from auto_round.inference.convert_model import QuantLinear

        init_kwargs = self.get_dummy_model_init_kwargs()
        quantization_config_kwargs = self.get_dummy_init_kwargs()
        quantization_config_kwargs.update({"modules_to_not_convert": self.modules_to_not_convert})
        quantization_config = AutoRoundConfig(**quantization_config_kwargs)
        init_kwargs.update({"quantization_config": quantization_config})

        model = self.model_cls.from_pretrained(**init_kwargs)
        model.to(torch_device)

        for name, module in model.named_modules():
            if name in self.modules_to_not_convert:
                assert not isinstance(
                    module, QuantLinear
                ), f"Module '{name}' should NOT have been quantized but is a QuantLinear."

    def test_serialization(self):
        """Test round-trip save and load of an AutoRound quantized model."""
        model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())
        inputs = self.get_dummy_inputs()

        model.to(torch_device)
        with torch.no_grad():
            model_output = model(**inputs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            saved_model = self.model_cls.from_pretrained(
                tmp_dir,
                torch_dtype=self.torch_dtype,
            )

        saved_model.to(torch_device)
        with torch.no_grad():
            saved_model_output = saved_model(**inputs)

        assert torch.allclose(model_output.sample, saved_model_output.sample, rtol=1e-5, atol=1e-5)

    def test_torch_compile(self):
        """Test that the quantized model works with torch.compile."""
        if not self._test_torch_compile:
            return

        model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())
        compiled_model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)

        model.to(torch_device)
        with torch.no_grad():
            model_output = model(**self.get_dummy_inputs()).sample

        compiled_model.to(torch_device)
        with torch.no_grad():
            compiled_model_output = compiled_model(**self.get_dummy_inputs()).sample

        model_output = model_output.detach().float().cpu().numpy()
        compiled_model_output = compiled_model_output.detach().float().cpu().numpy()

        max_diff = numpy_cosine_similarity_distance(model_output.flatten(), compiled_model_output.flatten())
        assert max_diff < 1e-3

    def test_model_cpu_offload(self):
        """Test that the quantized model works with pipeline CPU offload."""
        init_kwargs = self.get_dummy_init_kwargs()
        transformer = self.model_cls.from_pretrained(
            self.model_id,
            quantization_config=AutoRoundConfig(**init_kwargs),
            subfolder="transformer",
            torch_dtype=self.torch_dtype,
        )
        pipe = self.pipeline_cls.from_pretrained(self.model_id, transformer=transformer, torch_dtype=self.torch_dtype)
        pipe.enable_model_cpu_offload(device=torch_device)
        _ = pipe("a cat holding a sign that says hello", num_inference_steps=2)


# ============================================================================
# Backend: auto (auto-select best available backend)
# ============================================================================


class AutoRoundW4G128AsymAutoBackendTest(AutoRoundBaseTesterMixin, unittest.TestCase):
    """W4A16, group_size=128, asymmetric, backend='auto' (default — auto-selects best backend)."""

    expected_memory_reduction = 0.55

    def get_dummy_init_kwargs(self):
        return {
            "bits": 4,
            "group_size": 128,
            "sym": False,
            "backend": "auto",
        }


class AutoRoundW4G128SymAutoBackendTest(AutoRoundBaseTesterMixin, unittest.TestCase):
    """W4A16, group_size=128, symmetric, backend='auto'."""

    expected_memory_reduction = 0.55

    def get_dummy_init_kwargs(self):
        return {
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "backend": "auto",
        }


class AutoRoundW4G32AsymAutoBackendTest(AutoRoundBaseTesterMixin, unittest.TestCase):
    """W4A16, group_size=32, asymmetric, backend='auto' (finer granularity)."""

    expected_memory_reduction = 0.50

    def get_dummy_init_kwargs(self):
        return {
            "bits": 4,
            "group_size": 32,
            "sym": False,
            "backend": "auto",
        }


# ============================================================================
# Backend: auto_round:tritonv2_zp (CUDA, Triton-based kernel)
# ============================================================================


@require_torch_cuda_compatibility(7.0)
class AutoRoundW4G128AsymTritonTest(AutoRoundBaseTesterMixin, unittest.TestCase):
    """W4A16, group_size=128, asymmetric, backend='auto_round:tritonv2_zp' (CUDA Triton kernel)."""

    expected_memory_reduction = 0.55

    def get_dummy_init_kwargs(self):
        return {
            "bits": 4,
            "group_size": 128,
            "sym": False,
            "backend": "auto_round:tritonv2_zp",
        }


@require_torch_cuda_compatibility(7.0)
class AutoRoundW4G128SymTritonTest(AutoRoundBaseTesterMixin, unittest.TestCase):
    """W4A16, group_size=128, symmetric, backend='auto_round:tritonv2_zp'."""

    expected_memory_reduction = 0.55

    def get_dummy_init_kwargs(self):
        return {
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "backend": "auto_round:tritonv2_zp",
        }


# ============================================================================
# Backend: gptqmodel:marlin_zp (CUDA, requires GPTQModel>=5.8.0, best perf)
# ============================================================================


@unittest.skipUnless(_is_gptqmodel_available("5.8.0"), "Test requires gptqmodel>=5.8.0")
@require_torch_cuda_compatibility(8.0)
class AutoRoundW4G128AsymMarlinTest(AutoRoundBaseTesterMixin, unittest.TestCase):
    """W4A16, group_size=128, asymmetric, backend='gptqmodel:marlin_zp' (best CUDA performance)."""

    _test_torch_compile = True
    expected_memory_reduction = 0.55

    def get_dummy_init_kwargs(self):
        return {
            "bits": 4,
            "group_size": 128,
            "sym": False,
            "backend": "gptqmodel:marlin_zp",
        }


@unittest.skipUnless(_is_gptqmodel_available("5.8.0"), "Test requires gptqmodel>=5.8.0")
@require_torch_cuda_compatibility(8.0)
class AutoRoundW4G128SymMarlinTest(AutoRoundBaseTesterMixin, unittest.TestCase):
    """W4A16, group_size=128, symmetric, backend='gptqmodel:marlin_zp'."""

    _test_torch_compile = True
    expected_memory_reduction = 0.55

    def get_dummy_init_kwargs(self):
        return {
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "backend": "gptqmodel:marlin_zp",
        }


# ============================================================================
# Backend: auto_round:torch_zp (CPU, pure PyTorch kernel)
# ============================================================================


class AutoRoundW4G128AsymTorchCPUTest(AutoRoundBaseTesterMixin, unittest.TestCase):
    """W4A16, group_size=128, asymmetric, backend='auto_round:torch_zp' (CPU)."""

    expected_memory_reduction = 0.50

    def get_dummy_init_kwargs(self):
        return {
            "bits": 4,
            "group_size": 128,
            "sym": False,
            "backend": "auto_round:torch_zp",
        }
