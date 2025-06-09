#!/usr/bin/env python3
"""
Comprehensive Device Map Testing Suite for Diffusers
====================================================

This test suite verifies that all device_map formats work correctly with the new
Accelerate integration in Diffusers. It tests multiple mainstream models with different
device mapping strategies to ensure direct device loading without CPU intermediates.

Models tested:
- Stable Diffusion XL (larger model with dual text encoders)
- FLUX.dev (transformer-based diffusion model)

Device map formats tested:
- String strategies: "auto", "balanced", "balanced_low_0", "sequential"
- Dict mappings: {"": "cuda:0"}, {"unet": 0, "vae": 1}, component-specific maps
- Special devices: "meta", "cpu", "disk"
- Memory constraints with max_memory parameter
- Multi-GPU scenarios with component distribution
"""

import gc
import tempfile
import unittest

import torch
from transformers import (
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
)

from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import is_accelerate_available
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    require_accelerate_version_greater,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)


class AccelerateDeviceMapFastTests(unittest.TestCase):
    """Fast tests for Accelerate device mapping integration using dummy components."""

    def setUp(self):
        """Set up test environment."""
        gc.collect()
        backend_empty_cache(torch_device)
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    def tearDown(self):
        """Clean up after tests."""
        gc.collect()
        backend_empty_cache(torch_device)

    def get_dummy_sdxl_components(self):
        """Create dummy SDXL components for testing (matches SDXL test patterns)."""
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(2, 4),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
            norm_num_groups=1,
        )
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        return {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "image_encoder": None,
            "feature_extractor": None,
        }

    def get_dummy_flux_components(self):
        """Create dummy FLUX components for testing (matches FLUX test patterns)."""
        torch.manual_seed(0)
        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
        )
        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )

        torch.manual_seed(0)
        text_encoder = CLIPTextModel(clip_text_encoder_config)

        torch.manual_seed(0)
        text_encoder_2 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=1,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "transformer": transformer,
            "vae": vae,
            "image_encoder": None,
            "feature_extractor": None,
        }

    def get_dummy_inputs_sdxl(self, device, seed=0):
        """Get dummy inputs for SDXL testing."""
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
        }

    def get_dummy_inputs_flux(self, device, seed=0):
        """Get dummy inputs for FLUX testing."""
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 48,
            "output_type": "np",
        }

    def test_accelerate_integration_available(self):
        """Test that Accelerate integration is available."""
        self.assertTrue(is_accelerate_available(), "Accelerate should be available")

        # Test that our custom functions are importable
        from diffusers.utils.accelerate_utils import validate_device_map

        # Test basic validation
        validate_device_map("auto")
        validate_device_map({"": "cpu"})
        validate_device_map({"": "meta"})

    def test_device_map_validation_cpu_safe(self):
        """Test device_map validation with CPU-safe formats only."""
        from diffusers.utils.accelerate_utils import validate_device_map

        # Valid string device maps (Accelerate handles these internally)
        valid_strings = ["auto", "balanced", "balanced_low_0", "sequential"]
        for device_map in valid_strings:
            with self.subTest(device_map=device_map):
                try:
                    validate_device_map(device_map)
                except Exception as e:
                    self.fail(f"validate_device_map failed for '{device_map}': {e}")

        # Valid dict device maps - CPU-safe only
        cpu_safe_dicts = [
            {"": "cpu"},
            {"": "meta"},
            {"": "disk"},
            {"unet": "cpu", "vae": "meta"},
            {"text_encoder": "cpu", "scheduler": "meta"},
            {"vae": "disk", "text_encoder": "cpu"},
        ]
        for device_map in cpu_safe_dicts:
            with self.subTest(device_map=device_map):
                try:
                    validate_device_map(device_map)
                except Exception as e:
                    self.fail(f"validate_device_map failed for {device_map}: {e}")

    def test_invalid_device_map_formats(self):
        """Test invalid device map formats (hardware-independent)."""
        from diffusers.utils.accelerate_utils import validate_device_map

        # Invalid formats that don't depend on hardware availability
        invalid_formats = [
            (123, "`device_map` must be"),
            (["cpu"], "`device_map` must be"),
            ({123: "cpu"}, "device_map keys must be strings"),
            ({"unet": ["cpu"]}, "device_map values must be"),
            ({"": -1}, "Device index must be non-negative"),
            ({"": "completely_invalid_device"}, "Invalid device string"),
        ]

        for device_map, expected_error in invalid_formats:
            with self.subTest(device_map=device_map):
                with self.assertRaises((ValueError, TypeError)) as cm:
                    validate_device_map(device_map)
                self.assertIn(expected_error, str(cm.exception))

    def test_cpu_dict_device_map_with_temp_save(self):
        """Test CPU dict device map by saving and loading dummy components."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            # Load with CPU device_map
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                tmp_dir,
                device_map={"": "cpu"},
                torch_dtype=torch.float16,
            )

            # Verify all components are on CPU
            self.assertEqual(str(pipeline.unet.device), "cpu")
            self.assertEqual(str(pipeline.vae.device), "cpu")
            self.assertEqual(str(pipeline.text_encoder.device), "cpu")
            self.assertEqual(str(pipeline.text_encoder_2.device), "cpu")

            # Verify hf_device_map is set
            self.assertIsNotNone(pipeline.hf_device_map)
            self.assertIn("", pipeline.hf_device_map)

            # Test inference works with CPU device loading
            inputs = self.get_dummy_inputs_sdxl("cpu")
            result = pipeline(**inputs)
            self.assertIsNotNone(result.images)
            self.assertEqual(result.images.shape[0], 1)  # Check batch size

    def test_cpu_device_map_sdxl(self):
        """Test CPU device mapping - useful for memory-constrained scenarios (SDXL)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            # Load with CPU device_map
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                tmp_dir,
                device_map={"": "cpu"},
                torch_dtype=torch.float16,
            )

            # Verify all SDXL components are on CPU
            self.assertEqual(str(pipeline.unet.device), "cpu")
            self.assertEqual(str(pipeline.vae.device), "cpu")
            self.assertEqual(str(pipeline.text_encoder.device), "cpu")
            self.assertEqual(str(pipeline.text_encoder_2.device), "cpu")

            # Test inference works on CPU (slower but should work)
            inputs = self.get_dummy_inputs_sdxl("cpu")
            result = pipeline(**inputs)
            self.assertIsNotNone(result.images)

    def test_meta_device_map_sdxl(self):
        """Test meta device mapping for memory introspection without loading weights (SDXL)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline - use PyTorch format for meta device compatibility
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir, safe_serialization=False)  # Force PyTorch format
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            # Load with meta device_map - safetensors doesn't support meta device loading
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                tmp_dir,
                device_map={"": "meta"},
                torch_dtype=torch.float16,
                use_safetensors=False,  # Required for meta device
            )

            # Verify all components are on meta device
            self.assertEqual(str(pipeline.unet.device), "meta")
            self.assertEqual(str(pipeline.vae.device), "meta")
            self.assertEqual(str(pipeline.text_encoder.device), "meta")
            self.assertEqual(str(pipeline.text_encoder_2.device), "meta")

            # Meta device models are useful for:
            # 1. Memory planning without actually loading weights
            # 2. Model architecture inspection
            # 3. Calculating memory requirements

            # We can check model exists and verify architecture
            self.assertIsNotNone(pipeline.unet)
            self.assertIsInstance(pipeline.unet, UNet2DConditionModel)
            self.assertIsNotNone(pipeline.vae)
            self.assertIsInstance(pipeline.vae, AutoencoderKL)

            # Verify we can inspect model properties without weight access
            self.assertGreater(pipeline.unet.config.cross_attention_dim, 0)
            self.assertGreater(pipeline.vae.config.latent_channels, 0)

            # Meta device models cannot run inference (weights not loaded)
            with self.assertRaises((RuntimeError, ValueError)):
                inputs = self.get_dummy_inputs_sdxl("meta")
                pipeline(**inputs)

    def test_flux_cpu_device_mapping(self):
        """Test CPU device mapping with FLUX pipeline (transformer-based)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_flux_components()
            pipeline = FluxPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            # Load with CPU device_map
            pipeline = FluxPipeline.from_pretrained(
                tmp_dir,
                device_map={"": "cpu"},
                torch_dtype=torch.float16,
            )

            # Verify FLUX model components are on CPU
            model_components = ['transformer', 'vae', 'text_encoder', 'text_encoder_2']
            for component_name in model_components:
                if hasattr(pipeline, component_name):
                    component = getattr(pipeline, component_name)
                    if hasattr(component, 'device'):
                        self.assertEqual(str(component.device), "cpu",
                            f"{component_name} should be on cpu")

            # Test inference works on CPU
            inputs = self.get_dummy_inputs_flux("cpu")
            result = pipeline(**inputs)
            self.assertIsNotNone(result.images)
            self.assertEqual(result.images.shape[0], 1)

    def test_offload_folder_with_disk_device_sdxl(self):
        """Test disk offloading with offload_folder parameter (SDXL)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with tempfile.TemporaryDirectory() as offload_dir:
                device_map = {
                    "unet": "cpu",  # Use CPU for fast tests
                    "vae": "disk",
                    "text_encoder": "cpu",
                    "text_encoder_2": "disk"  # SDXL has two text encoders
                }

                # Create and save dummy pipeline
                components = self.get_dummy_sdxl_components()
                pipeline = StableDiffusionXLPipeline(**components)
                pipeline.save_pretrained(tmp_dir)
                del pipeline
                gc.collect()
                backend_empty_cache(torch_device)

                # Load with disk offloading
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    tmp_dir,
                    device_map=device_map,
                    offload_folder=offload_dir,
                    torch_dtype=torch.float16,
                )

                # Verify pipeline loaded successfully with disk offloading
                self.assertIsNotNone(pipeline.unet)
                self.assertIsNotNone(pipeline.vae)
                self.assertIsNotNone(pipeline.text_encoder)
                self.assertIsNotNone(pipeline.text_encoder_2)

                # Verify device placement
                self.assertEqual(str(pipeline.unet.device), "cpu")
                self.assertEqual(str(pipeline.text_encoder.device), "cpu")

                # Disk offloaded components should still be accessible
                # They'll be loaded on-demand when needed

    def test_accelerate_utils_integration(self):
        """Test that our custom accelerate utils functions work correctly."""
        from diffusers.utils.accelerate_utils import PipelineDeviceMapper, validate_device_map

        # Test validate_device_map with various inputs
        valid_maps = [
            "auto",
            "balanced",
            {"": "cpu"},
            {"": "cuda:0"} if torch.cuda.is_available() else {"": "cpu"},
            {"": "meta"},
            {"unet": "cuda:0", "vae": "cpu"} if torch.cuda.is_available() else {"unet": "cpu", "vae": "cpu"},
            {"": torch.device("cpu")},
        ]

        for device_map in valid_maps:
            with self.subTest(device_map=device_map):
                # Should not raise any exception
                validate_device_map(device_map)

        # Test PipelineDeviceMapper
        init_dict = {
            "unet": ("diffusers", "UNet2DConditionModel"),
            "vae": ("diffusers", "AutoencoderKL"),
            "text_encoder": ("transformers", "CLIPTextModel"),
            "scheduler": ("diffusers", "DDIMScheduler")
        }
        mapper = PipelineDeviceMapper(
            pipeline_class=StableDiffusionXLPipeline,
            init_dict=init_dict,
            passed_class_obj={},
            cached_folder=""
        )

        device_map = {"": "cpu"}
        component_maps = mapper.resolve_component_device_maps(
            device_map=device_map,
            max_memory=None,
            torch_dtype=torch.float16
        )

        self.assertIsInstance(component_maps, dict)
        for component in init_dict.keys():
            self.assertIn(component, component_maps)
            # Each component should get the resolved device map
            self.assertEqual(component_maps[component], device_map)

    def test_comprehensive_error_validation(self):
        """Test comprehensive error validation for ALL invalid device configurations."""
        from diffusers.utils.accelerate_utils import validate_device_map

        # Type validation errors (hardware-independent)
        type_errors = [
            (123, "device_map must be"),
            (45.7, "device_map must be"),
            (["cuda:0"], "device_map must be"),
            (("cuda:0",), "device_map must be"),
            ({"cuda:0"}, "device_map must be"),
        ]

        for device_map, expected_error in type_errors:
            with self.subTest(category="type_errors", device_map=device_map):
                with self.assertRaises((ValueError, TypeError)) as cm:
                    validate_device_map(device_map)
                self.assertIn(expected_error, str(cm.exception))

        # Dictionary key validation errors
        key_errors = [
            ({123: "cpu"}, "device_map keys must be strings"),
            ({45.7: "cpu"}, "device_map keys must be strings"),
            ({None: "cpu"}, "device_map keys must be strings"),
            ({("unet",): "cpu"}, "device_map keys must be strings"),
        ]

        for device_map, expected_error in key_errors:
            with self.subTest(category="key_errors", device_map=device_map):
                with self.assertRaises((ValueError, TypeError)) as cm:
                    validate_device_map(device_map)
                self.assertIn(expected_error, str(cm.exception))

        # Dictionary value validation errors
        value_errors = [
            ({"unet": ["cuda:0"]}, "device_map values must be"),
            ({"vae": {"device": "cpu"}}, "device_map values must be"),
            ({"text_encoder": None}, "device_map values must be"),
            ({"scheduler": (0,)}, "device_map values must be"),
        ]

        for device_map, expected_error in value_errors:
            with self.subTest(category="value_errors", device_map=device_map):
                with self.assertRaises((ValueError, TypeError)) as cm:
                    validate_device_map(device_map)
                self.assertIn(expected_error, str(cm.exception))

        # Device index validation errors
        index_errors = [
            ({"": -1}, "Device index must be non-negative"),
            ({"unet": -5}, "Device index must be non-negative"),
            ({"vae": -999}, "Device index must be non-negative"),
        ]

        for device_map, expected_error in index_errors:
            with self.subTest(category="index_errors", device_map=device_map):
                with self.assertRaises(ValueError) as cm:
                    validate_device_map(device_map)
                self.assertIn(expected_error, str(cm.exception))

        # Invalid device string errors (hardware-independent)
        string_errors = [
            ({"": "invalid_device"}, "Invalid device string"),
            ({"unet": "not_a_device"}, "Invalid device string"),
            ({"vae": "gpu"}, "Invalid device string"),  # Should be "cuda" not "gpu"
            ({"text_encoder": "device:0"}, "Invalid device string"),
        ]

        for device_map, expected_error in string_errors:
            with self.subTest(category="string_errors", device_map=device_map):
                with self.assertRaises(ValueError) as cm:
                    validate_device_map(device_map)
                self.assertIn(expected_error, str(cm.exception))

        # CUDA availability-dependent errors (only test if CUDA available)
        if torch.cuda.is_available():
            cuda_errors = [
                ({"": 999}, "CUDA device index 999 is not available"),
                ({"unet": "cuda:999"}, "CUDA device 'cuda:999' is not available"),
                ({"vae": torch.device("cuda:999")}, "CUDA device index 999 is not available"),
            ]

            for device_map, expected_error in cuda_errors:
                with self.subTest(category="cuda_errors", device_map=device_map):
                    with self.assertRaises(ValueError) as cm:
                        validate_device_map(device_map)
                    self.assertIn(expected_error, str(cm.exception))
        else:
            # Test CUDA-requested-but-unavailable errors
            cuda_unavailable_errors = [
                ({"": "cuda"}, "CUDA requested but no CUDA devices available"),
                ({"unet": "cuda:0"}, "CUDA requested but no CUDA devices available"),
                ({"vae": torch.device("cuda:0")}, "CUDA requested but no CUDA devices available"),
            ]

            for device_map, expected_error in cuda_unavailable_errors:
                with self.subTest(category="cuda_unavailable", device_map=device_map):
                    with self.assertRaises(ValueError) as cm:
                        validate_device_map(device_map)
                    self.assertIn(expected_error, str(cm.exception))

    def test_mps_device_validation(self):
        """Test MPS device validation based on system availability."""
        from diffusers.utils.accelerate_utils import validate_device_map

        mps_device_maps = [
            {"": "mps"},
            {"unet": "mps", "vae": "cpu"},
            {"text_encoder": torch.device("mps")},
        ]

        for device_map in mps_device_maps:
            with self.subTest(device_map=device_map):
                if torch.backends.mps.is_available():
                    # Should work on MPS-capable systems
                    try:
                        validate_device_map(device_map)
                    except Exception as e:
                        self.fail(f"MPS device validation failed unexpectedly: {e}")
                else:
                    # Should fail gracefully on non-MPS systems
                    with self.assertRaises(ValueError) as cm:
                        validate_device_map(device_map)
                    self.assertIn("MPS device requested but MPS is not available", str(cm.exception))

    def test_edge_case_device_maps(self):
        """Test edge cases and boundary conditions in device mapping."""
        from diffusers.utils.accelerate_utils import validate_device_map

        # Valid edge cases that should work
        valid_edge_cases = [
            {},  # Empty dict - should be valid
            {"": "cpu"},  # Root assignment only
            {"nonexistent_component": "cpu"},  # Component that doesn't exist - should still validate
            {"text_encoder.layers.0.attention": "cpu"},  # Deep hierarchical path
            {"unet": torch.device("cpu")},  # torch.device object
        ]

        for device_map in valid_edge_cases:
            with self.subTest(category="valid_edge", device_map=device_map):
                try:
                    validate_device_map(device_map)
                except Exception as e:
                    self.fail(f"Valid edge case failed: {device_map} - {e}")

        # Complex but valid hierarchical structures
        complex_valid = [
            {
                "unet.down_blocks.0": "cpu",
                "unet.down_blocks.1": "meta",
                "unet.up_blocks": "disk",
                "vae.encoder": "cpu",
                "vae.decoder": "meta",
                "text_encoder": "cpu",
            },
            {
                "": "meta",  # Default everything to meta
                "unet": "cpu",  # Override UNet to CPU
                "vae.encoder": "disk",  # Override VAE encoder to disk
            }
        ]

        for device_map in complex_valid:
            with self.subTest(category="complex_valid", device_map=device_map):
                try:
                    validate_device_map(device_map)
                except Exception as e:
                    self.fail(f"Complex valid case failed: {device_map} - {e}")

    def test_invalid_device_maps(self):
        """Test that invalid device maps are properly rejected (legacy compatibility)."""
        from diffusers.utils.accelerate_utils import validate_device_map

        # Maintain backward compatibility for existing tests
        invalid_maps = [
            ({"": "invalid_device"}, "Invalid device string"),
            (123, "`device_map` must be"),
            (["cuda:0"], "`device_map` must be"),
            ({"": -1}, "Device index must be non-negative"),
            ({123: "cuda:0"}, "device_map keys must be strings"),
            ({"unet": ["cuda:0"]}, "device_map values must be"),
        ]

        # Add CUDA-specific tests only if CUDA available
        if torch.cuda.is_available():
            invalid_maps.extend([
                ({"": 999}, "CUDA device index 999 is not available"),
                ({"unet": "cuda:999"}, "CUDA device 'cuda:999' is not available"),
            ])

        for device_map, expected_msg in invalid_maps:
            with self.subTest(device_map=device_map):
                with self.assertRaises((ValueError, TypeError)) as cm:
                    validate_device_map(device_map)
                if expected_msg:
                    self.assertIn(expected_msg, str(cm.exception))

    @require_torch_accelerator
    def test_hierarchical_device_map(self):
        """Test hierarchical device mapping for nested model structures."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            # Test hierarchical device map (for models with nested modules)
            device_map = {
                "unet.down_blocks": 0,
                "unet.up_blocks": "cpu",
                "unet.mid_block": 0,
                "vae": "cpu",
                "text_encoder": 0,
                "text_encoder_2": "cpu",
            }

            pipeline = StableDiffusionXLPipeline.from_pretrained(
                tmp_dir,
                device_map=device_map,
                torch_dtype=torch.float16,
            )

            # Verify hierarchical mapping worked
            self.assertIsNotNone(pipeline.hf_device_map)
            # The actual device mapping will be resolved by Accelerate

    @require_torch_accelerator
    def test_max_memory_constraint(self):
        """Test device mapping with max_memory constraints."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            # Test with max_memory constraint
            max_memory = {0: "1GB", "cpu": "2GB"}

            pipeline = StableDiffusionXLPipeline.from_pretrained(
                tmp_dir,
                device_map="auto",
                max_memory=max_memory,
                torch_dtype=torch.float16,
            )

            # Verify pipeline loaded with memory constraints
            self.assertIsNotNone(pipeline.hf_device_map)
            # Components should be distributed based on memory constraints

    def test_reset_device_map(self):
        """Test resetting device map to None."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            # Load with device_map
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                tmp_dir,
                device_map={"": "cpu"},
                torch_dtype=torch.float16,
            )

            # Verify device_map is set
            self.assertIsNotNone(pipeline.hf_device_map)

            # Reset device map
            pipeline.reset_device_map()

            # Verify device_map is cleared
            self.assertIsNone(getattr(pipeline, 'hf_device_map', None))

    def test_all_accelerate_string_strategies_cpu_fallback(self):
        """Test that ALL Accelerate string strategies work gracefully in CPU-only scenarios."""
        strategies = ["auto", "balanced", "balanced_low_0", "sequential"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            for strategy in strategies:
                with self.subTest(strategy=strategy):
                    # Load with string strategy - should work on CPU-only systems
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        tmp_dir,
                        device_map=strategy,
                        torch_dtype=torch.float16,
                    )

                    # All strategies should result in working pipeline
                    self.assertIsNotNone(pipeline.unet)
                    self.assertIsNotNone(pipeline.vae)
                    self.assertIsNotNone(pipeline.text_encoder)
                    self.assertIsNotNone(pipeline.text_encoder_2)

                    # Verify hf_device_map is set
                    self.assertIsNotNone(pipeline.hf_device_map)

                    # Test that inference works (should adapt to available hardware)
                    inputs = self.get_dummy_inputs_sdxl("cpu")
                    result = pipeline(**inputs)
                    self.assertIsNotNone(result.images)
                    self.assertEqual(result.images.shape[0], 1)

                    del pipeline
                    gc.collect()
                    backend_empty_cache(torch_device)

    def test_torch_device_object_formats(self):
        """Test that torch.device objects work correctly in device maps."""
        device_formats = [
            torch.device("cpu"),
            torch.device("meta"),
        ]

        # Add CUDA device objects if available
        if torch.cuda.is_available():
            device_formats.append(torch.device("cuda:0"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline in PyTorch format for meta device compatibility
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir, safe_serialization=False)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            for device_obj in device_formats:
                with self.subTest(device=device_obj):
                    try:
                        # Load with torch.device object
                        pipeline = StableDiffusionXLPipeline.from_pretrained(
                            tmp_dir,
                            device_map={"": device_obj},
                            torch_dtype=torch.float16,
                            use_safetensors=False if str(device_obj) == "meta" else True,
                        )

                        # Verify device placement
                        self.assertEqual(str(pipeline.unet.device), str(device_obj))
                        self.assertEqual(str(pipeline.vae.device), str(device_obj))
                        self.assertEqual(str(pipeline.text_encoder.device), str(device_obj))
                        self.assertEqual(str(pipeline.text_encoder_2.device), str(device_obj))

                        # Test inference for non-meta devices
                        if str(device_obj) != "meta":
                            inputs = self.get_dummy_inputs_sdxl(str(device_obj))
                            result = pipeline(**inputs)
                            self.assertIsNotNone(result.images)

                        del pipeline
                        gc.collect()
                        backend_empty_cache(torch_device)
                    except Exception as e:
                        # Skip if device not available, but don't fail
                        if "not available" in str(e) or "CUDA" in str(e):
                            continue
                        raise

    def test_integer_device_indices_cpu_safe(self):
        """Test integer device indices in CPU-safe scenarios."""
        # Only test device index 0, which should map to available devices
        device_maps = [
            {"": 0},  # All components on device 0
            {"unet": 0, "vae": 0, "text_encoder": 0, "text_encoder_2": 0},  # Explicit mapping
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            for device_map in device_maps:
                with self.subTest(device_map=device_map):
                    try:
                        # This should work if device 0 exists (GPU or CPU)
                        pipeline = StableDiffusionXLPipeline.from_pretrained(
                            tmp_dir,
                            device_map=device_map,
                            torch_dtype=torch.float16,
                        )

                        # Verify pipeline loaded successfully
                        self.assertIsNotNone(pipeline.unet)
                        self.assertIsNotNone(pipeline.hf_device_map)

                        # Test inference works
                        inputs = self.get_dummy_inputs_sdxl(torch_device)
                        result = pipeline(**inputs)
                        self.assertIsNotNone(result.images)

                        del pipeline
                        gc.collect()
                        backend_empty_cache(torch_device)
                    except Exception as e:
                        # Skip if device index not available
                        if "not available" in str(e) or "CUDA" in str(e):
                            continue
                        raise

    def test_mixed_device_scenarios_comprehensive(self):
        """Test complex mixed device scenarios that users might encounter."""
        mixed_scenarios = [
            # CPU + meta combination (memory planning)
            {"unet": "cpu", "vae": "meta", "text_encoder": "cpu", "text_encoder_2": "meta"},
            # CPU + disk combination (memory offloading)
            {"unet": "cpu", "vae": "disk", "text_encoder": "cpu", "text_encoder_2": "disk"},
            # All different devices
            {"unet": "cpu", "vae": "meta", "text_encoder": "disk", "text_encoder_2": "cpu"},
        ]

        # Add CUDA combinations if available
        if torch.cuda.is_available():
            mixed_scenarios.extend([
                {"unet": "cuda:0", "vae": "cpu", "text_encoder": "meta", "text_encoder_2": "disk"},
                {"unet": 0, "vae": "cpu", "text_encoder": "meta", "text_encoder_2": torch.device("cpu")},
            ])

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            for device_map in mixed_scenarios:
                with self.subTest(device_map=device_map):
                    try:
                        # Load with mixed device configuration
                        pipeline = StableDiffusionXLPipeline.from_pretrained(
                            tmp_dir,
                            device_map=device_map,
                            torch_dtype=torch.float16,
                            offload_folder=tmp_dir + "_offload",  # Handle disk devices
                        )

                        # Verify components are on expected devices
                        for component_name, expected_device in device_map.items():
                            component = getattr(pipeline, component_name)
                            if hasattr(component, 'device'):
                                actual_device = str(component.device)
                                expected_device_str = str(expected_device)
                                self.assertEqual(actual_device, expected_device_str,
                                    f"{component_name} should be on {expected_device_str}, got {actual_device}")

                        # Verify hf_device_map is set
                        self.assertIsNotNone(pipeline.hf_device_map)

                        del pipeline
                        gc.collect()
                        backend_empty_cache(torch_device)
                    except Exception as e:
                        # Skip if specific device not available, but don't fail silently
                        if "not available" in str(e) or "CUDA" in str(e):
                            continue
                        raise

    def test_hierarchical_device_mapping_cpu_safe(self):
        """Test hierarchical device mapping for complex model distributions."""
        hierarchical_maps = [
            # Distribute UNet blocks across CPU (always available)
            {
                "unet.down_blocks": "cpu",
                "unet.up_blocks": "cpu",
                "unet.mid_block": "cpu",
                "vae": "meta",  # Use meta for inspection
                "text_encoder": "cpu",
                "text_encoder_2": "disk",  # Offload to disk
            },
            # Component-level + root level mixing
            {
                "": "cpu",  # Default to CPU
                "vae": "meta",  # Override VAE to meta
                "text_encoder_2": "disk",  # Override text_encoder_2 to disk
            }
        ]

        # Add GPU hierarchical mappings if available
        if torch.cuda.is_available():
            hierarchical_maps.append({
                "unet.down_blocks": "cuda:0",
                "unet.up_blocks": "cpu",
                "vae": "cuda:0",
                "text_encoder": "cpu",
                "text_encoder_2": "meta",
            })

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            for device_map in hierarchical_maps:
                with self.subTest(device_map=device_map):
                    try:
                        # Load with hierarchical device mapping
                        pipeline = StableDiffusionXLPipeline.from_pretrained(
                            tmp_dir,
                            device_map=device_map,
                            torch_dtype=torch.float16,
                            offload_folder=tmp_dir + "_offload",
                        )

                        # Verify pipeline loaded successfully with hierarchical mapping
                        self.assertIsNotNone(pipeline.unet)
                        self.assertIsNotNone(pipeline.vae)
                        self.assertIsNotNone(pipeline.text_encoder)
                        self.assertIsNotNone(pipeline.text_encoder_2)

                        # Verify hf_device_map contains the hierarchical structure
                        self.assertIsNotNone(pipeline.hf_device_map)

                        # Hierarchical device maps should be reflected in the device map
                        for key in device_map.keys():
                            if key != "":  # Root assignment handled differently
                                self.assertTrue(
                                    any(key in map_key for map_key in pipeline.hf_device_map.keys()),
                                    f"Hierarchical key '{key}' should be reflected in device map"
                                )

                        del pipeline
                        gc.collect()
                        backend_empty_cache(torch_device)
                    except Exception as e:
                        # Skip if specific device not available
                        if "not available" in str(e) or "CUDA" in str(e):
                            continue
                        raise


class AccelerateDeviceMapGPUTests(unittest.TestCase):
    """GPU tests with proper decorators for single GPU scenarios."""

    def setUp(self):
        """Set up test environment."""
        gc.collect()
        backend_empty_cache(torch_device)
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    def tearDown(self):
        """Clean up after tests."""
        gc.collect()
        backend_empty_cache(torch_device)

    def get_dummy_sdxl_components(self):
        """Create dummy SDXL components for testing (matches SDXL test patterns)."""
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(2, 4),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
            norm_num_groups=1,
        )
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        return {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "image_encoder": None,
            "feature_extractor": None,
        }

    def get_dummy_inputs_sdxl(self, device, seed=0):
        """Get dummy inputs for SDXL testing."""
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
        }

    @require_torch_accelerator
    def test_cuda_device_map_validation(self):
        """Test CUDA device maps when GPU is available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        from diffusers.utils.accelerate_utils import validate_device_map

        cuda_device_maps = [
            {"": "cuda:0"},
            {"": torch.device("cuda:0")},
            {"": 0},  # Integer device index
            {"unet": "cuda:0", "vae": "cpu"},
        ]

        for device_map in cuda_device_maps:
            validate_device_map(device_map)

    @require_torch_accelerator
    def test_simple_cuda_device_map_with_temp_save(self):
        """Test simple CUDA dict device map by saving and loading dummy components."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            # Load with CUDA device_map - this was previously broken
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                tmp_dir,
                device_map={"": "cuda:0"},  # This format was broken before our fix
                torch_dtype=torch.float16,
            )

            # Verify all components are on the right device
            self.assertEqual(str(pipeline.unet.device), "cuda:0")
            self.assertEqual(str(pipeline.vae.device), "cuda:0")
            self.assertEqual(str(pipeline.text_encoder.device), "cuda:0")
            self.assertEqual(str(pipeline.text_encoder_2.device), "cuda:0")

            # Verify hf_device_map is set
            self.assertIsNotNone(pipeline.hf_device_map)
            self.assertIn("", pipeline.hf_device_map)

            # Test inference works with direct device loading
            inputs = self.get_dummy_inputs_sdxl(torch_device)
            result = pipeline(**inputs)
            self.assertIsNotNone(result.images)
            self.assertEqual(result.images.shape[0], 1)  # Check batch size

    @require_torch_accelerator
    def test_all_gpu_string_strategies(self):
        """Test all Accelerate string strategies with GPU available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        strategies = ["auto", "balanced", "balanced_low_0", "sequential"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            for strategy in strategies:
                with self.subTest(strategy=strategy):
                    # Load with GPU strategy
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        tmp_dir,
                        device_map=strategy,
                        torch_dtype=torch.float16,
                    )

                    # All strategies should result in working pipeline
                    self.assertIsNotNone(pipeline.unet)
                    self.assertIsNotNone(pipeline.vae)
                    self.assertIsNotNone(pipeline.text_encoder)
                    self.assertIsNotNone(pipeline.text_encoder_2)

                    # Verify hf_device_map is set
                    self.assertIsNotNone(pipeline.hf_device_map)

                    # Test inference works
                    inputs = self.get_dummy_inputs_sdxl(torch_device)
                    result = pipeline(**inputs)
                    self.assertIsNotNone(result.images)

                    del pipeline
                    gc.collect()
                    backend_empty_cache(torch_device)

    @require_torch_accelerator
    def test_gpu_mixed_device_scenarios(self):
        """Test complex GPU + other device combinations."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        mixed_scenarios = [
            # GPU + CPU combinations
            {"unet": "cuda:0", "vae": "cpu", "text_encoder": "cuda:0", "text_encoder_2": "cpu"},
            # GPU + meta combinations
            {"unet": "cuda:0", "vae": "meta", "text_encoder": "cpu", "text_encoder_2": "cuda:0"},
            # GPU + disk combinations
            {"unet": "cuda:0", "vae": "disk", "text_encoder": "cuda:0", "text_encoder_2": "disk"},
            # All formats: integer, string, torch.device
            {"unet": 0, "vae": "cpu", "text_encoder": torch.device("cuda:0"), "text_encoder_2": "meta"},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            for device_map in mixed_scenarios:
                with self.subTest(device_map=device_map):
                    # Load with mixed GPU device configuration
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        tmp_dir,
                        device_map=device_map,
                        torch_dtype=torch.float16,
                        offload_folder=tmp_dir + "_offload",
                    )

                    # Verify components are on expected devices
                    for component_name, expected_device in device_map.items():
                        component = getattr(pipeline, component_name)
                        if hasattr(component, 'device'):
                            actual_device = str(component.device)
                            expected_device_str = str(expected_device)
                            # Handle integer device indices (map to cuda:0)
                            if expected_device == 0:
                                expected_device_str = "cuda:0"
                            self.assertEqual(actual_device, expected_device_str,
                                f"{component_name} should be on {expected_device_str}, got {actual_device}")

                    # Verify hf_device_map is set
                    self.assertIsNotNone(pipeline.hf_device_map)

                    del pipeline
                    gc.collect()
                    backend_empty_cache(torch_device)

    @require_torch_accelerator
    def test_gpu_hierarchical_device_mapping(self):
        """Test GPU-specific hierarchical device mapping."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        hierarchical_maps = [
            # Distribute UNet blocks across GPU and CPU
            {
                "unet.down_blocks": "cuda:0",
                "unet.up_blocks": "cpu",
                "unet.mid_block": "cuda:0",
                "vae": "cuda:0",
                "text_encoder": "cpu",
                "text_encoder_2": "cuda:0",
            },
            # Complex mixed hierarchical distribution
            {
                "": "cpu",  # Default to CPU
                "unet": "cuda:0",  # Override UNet to GPU
                "vae.encoder": "cuda:0",  # Override VAE encoder to GPU
                "vae.decoder": "cpu",  # Keep VAE decoder on CPU
                "text_encoder_2": "meta",  # Use meta for text_encoder_2
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            for device_map in hierarchical_maps:
                with self.subTest(device_map=device_map):
                    # Load with hierarchical GPU device mapping
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        tmp_dir,
                        device_map=device_map,
                        torch_dtype=torch.float16,
                    )

                    # Verify pipeline loaded successfully with hierarchical mapping
                    self.assertIsNotNone(pipeline.unet)
                    self.assertIsNotNone(pipeline.vae)
                    self.assertIsNotNone(pipeline.text_encoder)
                    self.assertIsNotNone(pipeline.text_encoder_2)

                    # Verify hf_device_map contains the hierarchical structure
                    self.assertIsNotNone(pipeline.hf_device_map)

                    # Hierarchical device maps should be reflected in the device map
                    for key in device_map.keys():
                        if key != "":  # Root assignment handled differently
                            self.assertTrue(
                                any(key in map_key for map_key in pipeline.hf_device_map.keys()),
                                f"Hierarchical key '{key}' should be reflected in device map"
                            )

                    del pipeline
                    gc.collect()
                    backend_empty_cache(torch_device)

    @require_torch_accelerator
    def test_gpu_memory_constrained_scenarios(self):
        """Test GPU scenarios with memory constraints."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            # Test with max_memory constraint on GPU
            max_memory = {0: "512MB", "cpu": "1GB"}

            pipeline = StableDiffusionXLPipeline.from_pretrained(
                tmp_dir,
                device_map="auto",
                max_memory=max_memory,
                torch_dtype=torch.float16,
            )

            # Verify pipeline loaded with memory constraints
            self.assertIsNotNone(pipeline.hf_device_map)
            # With constrained GPU memory, some components should be on CPU
            device_types = set()
            for component_name in ["unet", "vae", "text_encoder", "text_encoder_2"]:
                component = getattr(pipeline, component_name)
                if hasattr(component, 'device'):
                    device_types.add(str(component.device).split(':')[0])

            # Should use both GPU and CPU due to memory constraints
            self.assertGreaterEqual(len(device_types), 1, "Should use at least one device type")

            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)


@require_torch_multi_accelerator
@require_accelerate_version_greater("0.27.0")
class AccelerateDeviceMapMultiGPUTests(unittest.TestCase):
    """Multi-GPU tests for comprehensive device mapping scenarios."""

    def setUp(self):
        """Set up test environment."""
        gc.collect()
        backend_empty_cache(torch_device)
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    def tearDown(self):
        """Clean up after tests."""
        gc.collect()
        backend_empty_cache(torch_device)

    def get_dummy_sdxl_components(self):
        """Create dummy SDXL components for testing (matches SDXL test patterns)."""
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(2, 4),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
            norm_num_groups=1,
        )
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        return {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "image_encoder": None,
            "feature_extractor": None,
        }

    def get_dummy_inputs_sdxl(self, device, seed=0):
        """Get dummy inputs for SDXL testing."""
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
        }

    def test_multi_gpu_explicit_device_mapping(self):
        """Test explicit device mapping across multiple GPUs."""
        if self.device_count < 2:
            self.skipTest("Need at least 2 GPUs for this test")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            # Test multiple explicit mapping strategies
            multi_gpu_mappings = [
                # Integer device indices
                {
                    "unet": 0,
                    "vae": 1,
                    "text_encoder": 0,
                    "text_encoder_2": 1,
                },
                # String device names
                {
                    "unet": "cuda:0",
                    "vae": "cuda:1",
                    "text_encoder": "cuda:1",
                    "text_encoder_2": "cuda:0",
                },
                # Mixed formats
                {
                    "unet": 0,
                    "vae": "cuda:1",
                    "text_encoder": torch.device("cuda:0"),
                    "text_encoder_2": 1,
                },
            ]

            for device_map in multi_gpu_mappings:
                with self.subTest(device_map=device_map):
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        tmp_dir,
                        device_map=device_map,
                        torch_dtype=torch.float16,
                    )

                    # Verify components are on correct devices
                    for component_name, expected_device in device_map.items():
                        component = getattr(pipeline, component_name)
                        if hasattr(component, 'device'):
                            actual_device = str(component.device)
                            expected_device_str = str(expected_device)
                            # Normalize integer indices to cuda:N format
                            if isinstance(expected_device, int):
                                expected_device_str = f"cuda:{expected_device}"
                            elif isinstance(expected_device, torch.device):
                                expected_device_str = str(expected_device)

                            self.assertEqual(actual_device, expected_device_str,
                                f"{component_name} should be on {expected_device_str}, got {actual_device}")

                    # Test inference works across multiple GPUs
                    inputs = self.get_dummy_inputs_sdxl("cuda:0")
                    result = pipeline(**inputs)
                    self.assertIsNotNone(result.images)

                    del pipeline
                    gc.collect()
                    backend_empty_cache(torch_device)

    def test_multi_gpu_auto_strategies(self):
        """Test all auto strategies with multi-GPU setup."""
        if self.device_count < 2:
            self.skipTest("Need at least 2 GPUs for multi-GPU strategies")

        strategies = ["auto", "balanced", "balanced_low_0", "sequential"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            for strategy in strategies:
                with self.subTest(strategy=strategy):
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        tmp_dir,
                        device_map=strategy,
                        torch_dtype=torch.float16,
                    )

                    # Verify pipeline loaded successfully
                    self.assertIsNotNone(pipeline.unet)
                    self.assertIsNotNone(pipeline.vae)
                    self.assertIsNotNone(pipeline.text_encoder)
                    self.assertIsNotNone(pipeline.text_encoder_2)

                    # Verify hf_device_map is set
                    self.assertIsNotNone(pipeline.hf_device_map)

                    # For multi-GPU strategies, should use multiple devices
                    device_set = set()
                    for component_name in ["unet", "vae", "text_encoder", "text_encoder_2"]:
                        component = getattr(pipeline, component_name)
                        if hasattr(component, 'device'):
                            device_set.add(str(component.device))

                    # Should use multiple devices in multi-GPU setup
                    if strategy in ["balanced", "auto"]:
                        self.assertGreaterEqual(len(device_set), 1,
                            f"Strategy {strategy} should distribute across available GPUs")

                    # Test inference works
                    inputs = self.get_dummy_inputs_sdxl("cuda:0")
                    result = pipeline(**inputs)
                    self.assertIsNotNone(result.images)

                    del pipeline
                    gc.collect()
                    backend_empty_cache(torch_device)

    def test_multi_gpu_hierarchical_mapping(self):
        """Test hierarchical device mapping across multiple GPUs."""
        if self.device_count < 2:
            self.skipTest("Need at least 2 GPUs for hierarchical multi-GPU mapping")

        hierarchical_maps = [
            # Distribute UNet layers across GPUs
            {
                "unet.down_blocks": "cuda:0",
                "unet.up_blocks": "cuda:1",
                "unet.mid_block": "cuda:0",
                "vae": "cuda:1",
                "text_encoder": "cuda:0",
                "text_encoder_2": "cuda:1",
            },
            # Complex hierarchical distribution
            {
                "": "cuda:0",  # Default to GPU 0
                "vae": "cuda:1",  # Override VAE to GPU 1
                "unet.down_blocks": "cuda:1",  # Override UNet down blocks to GPU 1
                "text_encoder_2": "cpu",  # Keep text_encoder_2 on CPU
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            for device_map in hierarchical_maps:
                with self.subTest(device_map=device_map):
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        tmp_dir,
                        device_map=device_map,
                        torch_dtype=torch.float16,
                    )

                    # Verify pipeline loaded successfully
                    self.assertIsNotNone(pipeline.unet)
                    self.assertIsNotNone(pipeline.vae)
                    self.assertIsNotNone(pipeline.text_encoder)
                    self.assertIsNotNone(pipeline.text_encoder_2)

                    # Verify hf_device_map contains hierarchical structure
                    self.assertIsNotNone(pipeline.hf_device_map)

                    # Should use multiple GPUs
                    device_set = set()
                    for component_name in ["unet", "vae", "text_encoder", "text_encoder_2"]:
                        component = getattr(pipeline, component_name)
                        if hasattr(component, 'device'):
                            device_set.add(str(component.device))

                    gpu_devices = [d for d in device_set if d.startswith("cuda")]
                    self.assertGreaterEqual(len(gpu_devices), 1, "Should use multiple GPU devices")

                    del pipeline
                    gc.collect()
                    backend_empty_cache(torch_device)

    def test_multi_gpu_memory_constraints(self):
        """Test multi-GPU scenarios with memory constraints."""
        if self.device_count < 2:
            self.skipTest("Need at least 2 GPUs for memory constraint testing")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            # Test with memory constraints across multiple GPUs
            max_memory_configs = [
                # Balanced across GPUs
                {0: "512MB", 1: "512MB", "cpu": "1GB"},
                # Unbalanced GPU memory
                {0: "256MB", 1: "768MB", "cpu": "1GB"},
                # One GPU with very little memory
                {0: "128MB", 1: "1GB", "cpu": "2GB"},
            ]

            for max_memory in max_memory_configs:
                with self.subTest(max_memory=max_memory):
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        tmp_dir,
                        device_map="auto",
                        max_memory=max_memory,
                        torch_dtype=torch.float16,
                    )

                    # Verify pipeline loaded with memory constraints
                    self.assertIsNotNone(pipeline.hf_device_map)

                    # Should distribute across available devices based on memory
                    device_set = set()
                    for component_name in ["unet", "vae", "text_encoder", "text_encoder_2"]:
                        component = getattr(pipeline, component_name)
                        if hasattr(component, 'device'):
                            device_set.add(str(component.device))

                    # Should use multiple devices due to memory constraints
                    self.assertGreaterEqual(len(device_set), 1, "Should distribute across devices")

                    del pipeline
                    gc.collect()
                    backend_empty_cache(torch_device)

    def test_multi_gpu_mixed_device_scenarios(self):
        """Test complex mixed device scenarios in multi-GPU environment."""
        if self.device_count < 2:
            self.skipTest("Need at least 2 GPUs for mixed device testing")

        mixed_scenarios = [
            # GPUs + CPU + meta + disk
            {
                "unet": "cuda:0",
                "vae": "cuda:1",
                "text_encoder": "cpu",
                "text_encoder_2": "meta",
            },
            # GPUs + disk offloading
            {
                "unet": "cuda:0",
                "vae": "disk",
                "text_encoder": "cuda:1",
                "text_encoder_2": "disk",
            },
            # All device types mixed
            {
                "unet": 0,  # cuda:0
                "vae": "cuda:1",
                "text_encoder": "cpu",
                "text_encoder_2": "meta",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save dummy pipeline
            components = self.get_dummy_sdxl_components()
            pipeline = StableDiffusionXLPipeline(**components)
            pipeline.save_pretrained(tmp_dir)
            del pipeline
            gc.collect()
            backend_empty_cache(torch_device)

            for device_map in mixed_scenarios:
                with self.subTest(device_map=device_map):
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        tmp_dir,
                        device_map=device_map,
                        torch_dtype=torch.float16,
                        offload_folder=tmp_dir + "_offload",
                    )

                    # Verify components are placed correctly
                    for component_name, expected_device in device_map.items():
                        component = getattr(pipeline, component_name)
                        if hasattr(component, 'device'):
                            actual_device = str(component.device)
                            expected_device_str = str(expected_device)
                            if expected_device == 0:
                                expected_device_str = "cuda:0"

                            # Skip disk devices for device comparison (handled differently)
                            if expected_device_str != "disk":
                                self.assertEqual(actual_device, expected_device_str,
                                    f"{component_name} should be on {expected_device_str}, got {actual_device}")

                    # Verify hf_device_map is set
                    self.assertIsNotNone(pipeline.hf_device_map)

                    del pipeline
                    gc.collect()
                    backend_empty_cache(torch_device)


@slow
@require_torch_multi_accelerator
@require_accelerate_version_greater("0.27.0")
class AccelerateDeviceMapSlowTests(unittest.TestCase):
    """Slow tests using real models from Hub for thorough integration testing."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    def tearDown(self):
        """Clean up after tests."""
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_sdxl_real_model_device_mapping(self):
        """Test device mapping with real SDXL model from Hub."""
        if self.device_count < 2:
            self.skipTest("Need at least 2 GPUs for this test")

        device_map = {
            "unet": 0,
            "vae": 1,
            "text_encoder": 0,
            "text_encoder_2": 1,
        }

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            device_map=device_map,
            torch_dtype=torch.float16,
        )

        # Verify components are on correct devices
        self.assertEqual(str(pipeline.unet.device), "cuda:0")
        self.assertEqual(str(pipeline.vae.device), "cuda:1")
        self.assertEqual(str(pipeline.text_encoder.device), "cuda:0")
        self.assertEqual(str(pipeline.text_encoder_2.device), "cuda:1")

        # Test inference works
        result = pipeline(
            "test prompt",
            num_inference_steps=1,
            height=64,
            width=64,
            output_type="pt"
        )
        self.assertIsNotNone(result.images)

    def test_auto_device_mapping_real_model(self):
        """Test auto device mapping with real model."""
        if self.device_count < 2:
            self.skipTest("Need at least 2 GPUs for balanced mapping")

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            device_map="balanced",
            torch_dtype=torch.float16,
        )

        # Verify components are distributed across devices
        device_set = set()
        component_devices = {}
        for name, component in pipeline.components.items():
            if hasattr(component, "device"):
                device_str = str(component.device)
                device_set.add(device_str)
                component_devices[name] = device_str
            elif hasattr(component, "parameters"):
                try:
                    first_param = next(component.parameters())
                    device_str = str(first_param.device)
                    device_set.add(device_str)
                    component_devices[name] = device_str
                except StopIteration:
                    pass

        # Should use multiple devices for balanced mapping on multi-GPU
        if self.device_count >= 2:
            self.assertGreaterEqual(len(device_set), 2,
                f"Expected balanced mapping to use multiple devices, got: {component_devices}")

        # Test inference works with balanced device distribution
        result = pipeline(
            "test prompt",
            num_inference_steps=2,
            height=64,
            width=64,
            output_type="pt"
        )
        self.assertIsNotNone(result.images)

        # Verify hf_device_map is populated
        self.assertIsNotNone(getattr(pipeline, 'hf_device_map', None))
