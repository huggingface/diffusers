import gc
import tempfile
import unittest

from diffusers import AutoRoundConfig, ZImageTransformer2DModel, ZImagePipeline
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
    model_id = "INCModel/Z-Image-tiny-for-testing-W4A16-AutoRound"
    model_cls = ZImageTransformer2DModel
    pipeline_cls = ZImagePipeline
    torch_dtype = torch.bfloat16
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
        """Creates dummy inputs matching ZImageTransformer2DModel.forward() signature.

        ZImageTransformer2DModel expects:
          - x: list of (C, F, H, W) tensors, one per batch item
          - t: 1-D timestep tensor of shape (batch_size,)
          - cap_feats: list of (seq_len, cap_feat_dim) tensors, one per batch item

        Dimensions are chosen to match the tiny test checkpoint
        (in_channels=16, cap_feat_dim=512, patch_size=2, f_patch_size=1).
        """
        batch_size = 1
        in_channels = 16      # matches tiny model config
        cap_feat_dim = 512    # matches tiny model config
        height = width = 8    # must be divisible by patch_size=2
        frames = 1            # must be divisible by f_patch_size=1
        seq_len = 16          # caption token count (will be padded to multiple of 32)

        torch.manual_seed(0)
        x = [
            torch.randn((in_channels, frames, height, width)).to(torch_device, dtype=self.torch_dtype)
            for _ in range(batch_size)
        ]
        cap_feats = [
            torch.randn((seq_len, cap_feat_dim)).to(torch_device, dtype=self.torch_dtype)
            for _ in range(batch_size)
        ]
        t = torch.tensor([0.5] * batch_size).to(torch_device, dtype=self.torch_dtype)

        return {"x": x, "cap_feats": cap_feats, "t": t}

    def test_autoround_memory_usage(self):
        """Compare peak memory between unquantized and AutoRound-quantized model.

        The quantized model should use significantly less memory due to 4-bit weight packing.
        `expected_memory_reduction` defines the minimum ratio (unquantized / quantized).
        """
        inputs = self.get_dummy_inputs()
        # x and cap_feats are lists of tensors; move each element individually.
        inputs = {
            k: [t.to(device=torch_device, dtype=self.torch_dtype) for t in v]
            if isinstance(v, list)
            else v.to(device=torch_device, dtype=self.torch_dtype)
            for k, v in inputs.items()
            if not isinstance(v, bool)
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
        from auto_round.inference.convert_model import dynamic_import_inference_linear

        init_kwargs = self.get_dummy_model_init_kwargs()
        quantization_config_kwargs = self.get_dummy_init_kwargs()
        quantization_config_kwargs.update({"modules_to_not_convert": self.modules_to_not_convert})
        quantization_config = AutoRoundConfig(**quantization_config_kwargs)
        init_kwargs.update({"quantization_config": quantization_config})

        model = self.model_cls.from_pretrained(**init_kwargs)
        model.to(torch_device)

        # Resolve the actual backend used after model loading.
        # When backend='auto', AutoRound selects the best available backend
        # per-layer during convert_hf_model(); 'auto' itself is not a valid
        # argument to dynamic_import_inference_linear.
        used_backends = getattr(model.hf_quantizer, "used_backends", [])
        resolved_backend = (
            used_backends[0]
            if used_backends
            else quantization_config_kwargs.get("backend", "auto_round:torch_zp")
        )
        quant_linear_cls = dynamic_import_inference_linear(resolved_backend, quantization_config_kwargs)

        for name, module in model.named_modules():
            if name in self.modules_to_not_convert:
                assert not isinstance(
                    module, quant_linear_cls
                ), f"Module '{name}' should NOT have been quantized but is a {quant_linear_cls}."

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

        # model_output.sample is a list of per-item tensors
        for out, saved_out in zip(model_output.sample, saved_model_output.sample):
            assert torch.allclose(out, saved_out, rtol=1e-5, atol=1e-5)

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

        # model_output is a list of per-item tensors; stack for comparison
        model_output = torch.stack([o.detach().float().cpu() for o in model_output]).numpy()
        compiled_model_output = torch.stack([o.detach().float().cpu() for o in compiled_model_output]).numpy()

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
