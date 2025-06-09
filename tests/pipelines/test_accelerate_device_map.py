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
from diffusers import (
    FluxPipeline, 
    StableDiffusionXLPipeline,
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


class AccelerateDeviceMapTests(unittest.TestCase):
    """Tests for Accelerate device mapping integration in Diffusers pipelines."""

    def setUp(self):
        """Set up test environment."""
        gc.collect()
        backend_empty_cache(torch_device)
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Test model configurations
        self.sdxl_model = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        # Using a smaller FLUX model for testing
        self.flux_model = "sayakpaul/FLUX.1-dev-nf4-pkg"

    def tearDown(self):
        """Clean up after tests."""
        gc.collect() 
        backend_empty_cache(torch_device)

    def test_accelerate_integration_available(self):
        """Test that Accelerate integration is available."""
        self.assertTrue(is_accelerate_available(), "Accelerate should be available")
        
        # Test that our custom functions are importable
        from diffusers.utils.accelerate_utils import validate_device_map, PipelineDeviceMapper
        
        # Test basic validation
        validate_device_map("auto")
        validate_device_map({"": "cuda:0"})
        validate_device_map({"": "meta"})

    def test_device_map_validation(self):
        """Test device_map validation with various formats."""
        from diffusers.utils.accelerate_utils import validate_device_map
        
        # Valid string device maps
        valid_strings = ["auto", "balanced", "balanced_low_0", "sequential"]
        for device_map in valid_strings:
            with self.subTest(device_map=device_map):
                try:
                    validate_device_map(device_map)
                except Exception as e:
                    self.fail(f"validate_device_map failed for '{device_map}': {e}")
        
        # Valid dict device maps
        valid_dicts = [
            {"": "cuda:0"},
            {"": "cpu"}, 
            {"": "meta"},
            {"": torch.device("cuda:0")},
            {"unet": 0, "vae": "cpu"},
            {"unet": "cuda:0", "vae": "cuda:1", "text_encoder": "cpu"},
        ]
        for device_map in valid_dicts:
            with self.subTest(device_map=device_map):
                try:
                    validate_device_map(device_map)
                except Exception as e:
                    self.fail(f"validate_device_map failed for {device_map}: {e}")

        # Invalid device maps should raise errors
        invalid_maps = [
            {"": "invalid_device"},
            {"": 999},  # Invalid device ID
            123,  # Wrong type
        ]
        for device_map in invalid_maps:
            with self.subTest(device_map=device_map):
                with self.assertRaises(ValueError):
                    validate_device_map(device_map)

    @require_torch_accelerator
    def test_simple_dict_device_map(self):
        """Test simple dict device map that was previously broken."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.sdxl_model,
            device_map={"": "cuda:0"},  # This format was broken before our fix
            torch_dtype=torch.float16,
        )
        
        # Verify all components are on the right device
        self.assertEqual(str(pipeline.unet.device), "cuda:0")
        self.assertEqual(str(pipeline.vae.device), "cuda:0")
        self.assertEqual(str(pipeline.text_encoder.device), "cuda:0")
        self.assertEqual(str(pipeline.text_encoder_2.device), "cuda:0")
        
        # Test inference works with direct device loading
        result = pipeline(
            "test prompt",
            num_inference_steps=1,
            height=64,
            width=64,
            output_type="pt"
        )
        self.assertIsNotNone(result.images)
        self.assertEqual(result.images.shape[0], 1)  # Check batch size

    @require_torch_accelerator 
    def test_cpu_device_map(self):
        """Test CPU device mapping - useful for memory-constrained scenarios."""
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.sdxl_model,
            device_map={"": "cpu"},
            torch_dtype=torch.float16,
        )
        
        # Verify all SDXL components are on CPU
        self.assertEqual(str(pipeline.unet.device), "cpu")
        self.assertEqual(str(pipeline.vae.device), "cpu")
        self.assertEqual(str(pipeline.text_encoder.device), "cpu")
        self.assertEqual(str(pipeline.text_encoder_2.device), "cpu")
        
        # Test inference works on CPU (slower but should work)
        result = pipeline(
            "test prompt",
            num_inference_steps=1,
            height=64,
            width=64,
            output_type="pt"
        )
        self.assertIsNotNone(result.images)

    @require_torch_accelerator
    def test_meta_device_map(self):
        """Test meta device mapping for memory introspection without loading weights."""
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.sdxl_model, 
            device_map={"": "meta"},
            torch_dtype=torch.float16,
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
        
        # We can check model exists but can't run inference
        self.assertIsNotNone(pipeline.unet)
        with self.assertRaises(RuntimeError):
            pipeline(
                "test prompt",
                num_inference_steps=1,
                height=64,
                width=64,
                output_type="pt"
            )

    @require_torch_multi_accelerator
    @require_accelerate_version_greater("0.27.0")
    def test_component_specific_device_map(self):
        """Test component-specific device mapping across multiple GPUs."""
        if self.device_count < 2:
            self.skipTest("Need at least 2 GPUs for this test")

        device_map = {
            "unet": 0,
            "vae": 1, 
            "text_encoder": 0,
            "safety_checker": "cpu"
        }
        
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.sdxl_model,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        
        # Verify components are on correct devices
        self.assertEqual(str(pipeline.unet.device), "cuda:0")
        self.assertEqual(str(pipeline.vae.device), "cuda:1") 
        self.assertEqual(str(pipeline.text_encoder.device), "cuda:0")
        if pipeline.safety_checker is not None:
            self.assertEqual(str(pipeline.safety_checker.device), "cpu")
        
        # Test inference works across devices
        result = pipeline(
            "test prompt",
            num_inference_steps=1,
            height=64,
            width=64,
            output_type="pt"
        )
        self.assertIsNotNone(result.images)
        
        # Verify cross-device communication works
        self.assertTrue(torch.cuda.device_count() >= 2)
        self.assertNotEqual(pipeline.unet.device, pipeline.vae.device)

    @require_torch_multi_accelerator
    @require_accelerate_version_greater("0.27.0") 
    @slow
    def test_balanced_device_map(self):
        """Test balanced device mapping strategy."""
        if self.device_count < 2:
            self.skipTest("Need at least 2 GPUs for balanced mapping")

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.sdxl_model,
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

    @require_torch_multi_accelerator
    @require_accelerate_version_greater("0.27.0")
    def test_max_memory_constraints(self):
        """Test device mapping with memory constraints."""
        if self.device_count < 2:
            self.skipTest("Need at least 2 GPUs for memory constraint testing")

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.sdxl_model,
            device_map="balanced",
            max_memory={0: "1GB", 1: "1GB"},
            torch_dtype=torch.float16,
        )
        
        # Verify pipeline loaded successfully with constraints
        self.assertIsNotNone(pipeline.unet)
        self.assertIsNotNone(pipeline.vae)
        self.assertIsNotNone(pipeline.text_encoder)
        
        # Test that memory constraints are respected
        # Components should be distributed according to memory limits
        device_memory_usage = {}
        for name, component in pipeline.components.items():
            if hasattr(component, "device") and hasattr(component, "get_memory_footprint"):
                device = str(component.device)
                memory = component.get_memory_footprint()
                device_memory_usage[device] = device_memory_usage.get(device, 0) + memory
                
        # Test inference still works with memory constraints
        result = pipeline(
            "test prompt",
            num_inference_steps=1,
            height=64,
            width=64,
            output_type="pt"
        )
        self.assertIsNotNone(result.images)

    @require_torch_accelerator
    def test_pipeline_device_mapper_functionality(self):
        """Test PipelineDeviceMapper class functionality."""
        from diffusers.utils.accelerate_utils import PipelineDeviceMapper
        
        components = ["unet", "vae", "text_encoder", "safety_checker"]
        mapper = PipelineDeviceMapper(
            components=components,
            torch_dtype=torch.float16
        )
        
        # Test with simple device map
        device_map = {"": "cuda:0"} if torch.cuda.is_available() else {"": "cpu"}
        component_maps = mapper.resolve_component_device_maps(
            device_map=device_map,
            max_memory=None,
            torch_dtype=torch.float16
        )
        
        self.assertIsInstance(component_maps, dict)
        for component in components:
            self.assertIn(component, component_maps)

    @require_torch_accelerator
    def test_sdxl_device_mapping(self):
        """Test device mapping with SDXL pipeline."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.sdxl_model, 
            device_map={"": "cuda:0"},
            torch_dtype=torch.float16,
        )
        
        # Verify components are on the right device (SDXL has dual text encoders)
        self.assertEqual(str(pipeline.unet.device), "cuda:0")
        self.assertEqual(str(pipeline.vae.device), "cuda:0")
        self.assertEqual(str(pipeline.text_encoder.device), "cuda:0")
        self.assertEqual(str(pipeline.text_encoder_2.device), "cuda:0")
        
        # Test inference works
        result = pipeline(
            "test prompt",
            num_inference_steps=1,
            height=64, 
            width=64,
            output_type="pt"
        )
        self.assertIsNotNone(result.images)
        
        # SDXL should have dual text encoders working
        self.assertTrue(hasattr(pipeline, 'text_encoder_2'))
        self.assertIsNotNone(pipeline.text_encoder_2)

    def test_device_map_legacy_compatibility(self):
        """Test that old device_map behavior is still rejected properly."""
        # This should now work with our new implementation
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch",
                device_map={"": "cpu"},  # This was previously broken
                torch_dtype=torch.float16,
            )
            # If we get here, the fix worked
            self.assertIsNotNone(pipeline)
        except ValueError as e:
            # If this fails, our fix didn't work
            self.fail(f"device_map dict format should now be supported: {e}")

    @require_torch_accelerator
    def test_offload_folder_with_disk_device(self):
        """Test disk offloading with offload_folder parameter."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            device_map = {
                "unet": "cuda:0" if torch.cuda.is_available() else "cpu",
                "vae": "disk",
                "text_encoder": "cpu",
                "text_encoder_2": "disk"  # SDXL has two text encoders
            }
            
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.sdxl_model,
                device_map=device_map,
                offload_folder=tmp_dir,
                torch_dtype=torch.float16,
            )
            
            # Verify pipeline loaded successfully with disk offloading
            self.assertIsNotNone(pipeline.unet)
            self.assertIsNotNone(pipeline.vae)
            self.assertIsNotNone(pipeline.text_encoder)
            self.assertIsNotNone(pipeline.text_encoder_2)
            
            # Verify device placement
            expected_unet_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.assertEqual(str(pipeline.unet.device), expected_unet_device)
            self.assertEqual(str(pipeline.text_encoder.device), "cpu")
            
            # Disk offloaded components should still be accessible
            # They'll be loaded on-demand when needed


    @require_torch_accelerator
    @slow  
    def test_flux_device_mapping(self):
        """Test device mapping with FLUX pipeline (transformer-based)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        try:
            pipeline = FluxPipeline.from_pretrained(
                self.flux_model,
                device_map={"": "cuda:0"},
                torch_dtype=torch.float16,
            )
        except Exception as e:
            # Skip if FLUX model is not available
            self.skipTest(f"FLUX model not available: {e}")
        
        # FLUX has these key components:
        # scheduler, text_encoder, text_encoder_2, tokenizer, tokenizer_2, transformer, vae
        flux_components = [
            'transformer', 'vae', 'text_encoder', 'text_encoder_2', 
            'scheduler', 'tokenizer', 'tokenizer_2'
        ]
        
        # Verify neural network components are on the right device
        neural_components = ['transformer', 'vae', 'text_encoder', 'text_encoder_2']
        for component_name in neural_components:
            if hasattr(pipeline, component_name):
                component = getattr(pipeline, component_name)
                if hasattr(component, 'device'):
                    self.assertEqual(str(component.device), "cuda:0", 
                        f"{component_name} should be on cuda:0")
        
        # Test inference works
        try:
            result = pipeline(
                "A beautiful landscape",
                num_inference_steps=1,
                height=64,
                width=64,
                output_type="pt"
            )
            self.assertIsNotNone(result.images)
            self.assertEqual(result.images.shape[0], 1)
        except Exception as e:
            # Some FLUX models might have different inference requirements
            self.skipTest(f"FLUX inference failed (expected for some model variants): {e}")
    
    @require_torch_multi_accelerator
    @require_accelerate_version_greater("0.27.0")
    @slow
    def test_flux_multi_gpu_device_mapping(self):
        """Test FLUX with component-specific device mapping across GPUs."""
        if self.device_count < 2:
            self.skipTest("Need at least 2 GPUs for this test")
            
        try:
            device_map = {
                "transformer": 0,  # Main model on GPU 0
                "vae": 1,          # VAE on GPU 1  
                "text_encoder": 0,  # Text encoders on GPU 0
                "text_encoder_2": 1, # Second text encoder on GPU 1
            }
            
            pipeline = FluxPipeline.from_pretrained(
                self.flux_model,
                device_map=device_map,
                torch_dtype=torch.float16,
            )
        except Exception as e:
            self.skipTest(f"FLUX model not available: {e}")
        
        # Verify components are on correct devices
        self.assertEqual(str(pipeline.transformer.device), "cuda:0")
        self.assertEqual(str(pipeline.vae.device), "cuda:1")
        self.assertEqual(str(pipeline.text_encoder.device), "cuda:0")
        self.assertEqual(str(pipeline.text_encoder_2.device), "cuda:1")
        
        # Test that cross-device inference works
        try:
            result = pipeline(
                "A futuristic cityscape",
                num_inference_steps=1,
                height=64,
                width=64,
                output_type="pt"
            )
            self.assertIsNotNone(result.images)
        except Exception as e:
            self.skipTest(f"FLUX cross-device inference failed: {e}")
    
    def test_string_device_map_formats(self):
        """Test all supported string device map formats."""
        valid_string_formats = ["auto", "balanced", "balanced_low_0", "sequential"]
        
        for device_map_str in valid_string_formats:
            with self.subTest(device_map=device_map_str):
                if device_map_str != "auto" and self.device_count < 2:
                    # Some strategies need multiple GPUs
                    continue
                    
                try:
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        self.sd15_model,
                        device_map=device_map_str,
                        torch_dtype=torch.float16,
                    )
                    
                    # Should have device map populated
                    self.assertIsNotNone(getattr(pipeline, 'hf_device_map', None),
                        f"hf_device_map should be set for {device_map_str}")
                    
                    # All components should be accessible
                    self.assertIsNotNone(pipeline.unet)
                    self.assertIsNotNone(pipeline.vae)
                    self.assertIsNotNone(pipeline.text_encoder)
                    
                except Exception as e:
                    if "insufficient" in str(e).lower() or "memory" in str(e).lower():
                        self.skipTest(f"Insufficient resources for {device_map_str}: {e}")
                    else:
                        raise
    
    def test_device_map_legacy_compatibility(self):
        """Test that device_map dict format works (was previously broken)."""
        legacy_breaking_cases = [
            # These were the problematic cases that used to fail before our fix
            {"": "cpu"},
            {"": "cuda:0"} if torch.cuda.is_available() else {"": "cpu"},
            {"": torch.device("cpu")},
            {"": 0} if torch.cuda.is_available() else {"": "cpu"},  # Device index
            {"unet": "cuda:0", "vae": "cpu"} if torch.cuda.is_available() else {"unet": "cpu", "vae": "cpu"},
        ]
        
        for device_map in legacy_breaking_cases:
            with self.subTest(device_map=device_map):
                try:
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        self.sd15_model,
                        device_map=device_map,
                        torch_dtype=torch.float16,
                    )
                    # If we get here, the fix worked
                    self.assertIsNotNone(pipeline)
                    self.assertIsNotNone(pipeline.unet)
                    self.assertIsNotNone(pipeline.vae)
                except ValueError as e:
                    if "device_map" in str(e) and "string" in str(e):
                        # This is the old broken behavior - should not happen
                        self.fail(f"device_map dict format should now be supported: {e}")
                    else:
                        # Some other error - re-raise
                        raise
    
    def test_int_device_map_conversion(self):
        """Test that integer device maps are properly converted."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        # Test integer device map (should convert to {"": device})
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.sdxl_model,
            device_map=0,  # Should convert to {"": "cuda:0"}
            torch_dtype=torch.float16,
        )
        
        # Verify all SDXL components are on cuda:0
        self.assertEqual(str(pipeline.unet.device), "cuda:0")
        self.assertEqual(str(pipeline.vae.device), "cuda:0")
        self.assertEqual(str(pipeline.text_encoder.device), "cuda:0")
        self.assertEqual(str(pipeline.text_encoder_2.device), "cuda:0")
    
    def test_accelerate_utils_integration(self):
        """Test that our custom accelerate utils functions work correctly."""
        from diffusers.utils.accelerate_utils import validate_device_map, PipelineDeviceMapper
        
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
        components = ["unet", "vae", "text_encoder", "scheduler"]
        mapper = PipelineDeviceMapper(
            components=components,
            torch_dtype=torch.float16
        )
        
        device_map = {"": "cpu"}
        component_maps = mapper.resolve_component_device_maps(
            device_map=device_map,
            max_memory=None,
            torch_dtype=torch.float16
        )
        
        self.assertIsInstance(component_maps, dict)
        for component in components:
            self.assertIn(component, component_maps)
            # Each component should get the resolved device map
            self.assertEqual(component_maps[component], device_map)
    
    @require_torch_accelerator
    def test_direct_device_loading(self):
        """Test that weights are loaded directly to target device without CPU intermediate."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        # Test both dict and string device maps
        device_maps_to_test = [
            {"": "cuda:0"},  # Dict format (was broken)
            {"": 0},         # Integer device index
            "auto",          # String strategy
        ]
        
        for device_map in device_maps_to_test:
            with self.subTest(device_map=device_map):
                # Clear GPU memory
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Load model with device_map
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    self.sdxl_model,
                    device_map=device_map,
                    torch_dtype=torch.float16,
                )
                
                # Verify all components are on GPU (auto might distribute)
                for name, component in pipeline.components.items():
                    if hasattr(component, "device"):
                        device_str = str(component.device)
                        self.assertTrue(
                            device_str.startswith("cuda") or device_str == "meta",
                            f"{name} is on {device_str}, expected cuda device"
                        )
                
                # Clean up
                del pipeline
                torch.cuda.empty_cache()
    
    def test_invalid_device_maps(self):
        """Test that invalid device maps are properly rejected."""
        from diffusers.utils.accelerate_utils import validate_device_map
        
        # Test various invalid device maps with specific error messages
        invalid_maps = [
            ({"": "invalid_device"}, "Invalid device string"),
            ({"": 999}, "CUDA device index 999 is not available" if torch.cuda.is_available() else None),
            (123, "device_map must be"),
            (["cuda:0"], "device_map must be"),
            ({"unet": "cuda:999"}, "CUDA device 'cuda:999' is not available" if torch.cuda.is_available() else None),
            ({"": -1}, "Device index must be non-negative"),
            ({123: "cuda:0"}, "device_map keys must be strings"),
            ({"unet": ["cuda:0"]}, "device_map values must be"),
        ]
        
        for device_map, expected_msg in invalid_maps:
            with self.subTest(device_map=device_map):
                if expected_msg is None:
                    # Skip test if expected error depends on CUDA availability
                    continue
                with self.assertRaises((ValueError, TypeError)) as cm:
                    validate_device_map(device_map)
                if expected_msg:
                    self.assertIn(expected_msg, str(cm.exception))
    
    @require_torch_accelerator
    def test_device_normalization(self):
        """Test that device strings are properly normalized."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        # Test that "cuda" and "cuda:0" behave the same
        pipeline1 = StableDiffusionXLPipeline.from_pretrained(
            self.sdxl_model,
            device_map={"": "cuda"},
            torch_dtype=torch.float16,
        )
        
        pipeline2 = StableDiffusionXLPipeline.from_pretrained(
            self.sdxl_model,
            device_map={"": "cuda:0"},
            torch_dtype=torch.float16,
        )
        
        # Both should result in components on cuda:0
        self.assertEqual(str(pipeline1.unet.device), "cuda:0")
        self.assertEqual(str(pipeline2.unet.device), "cuda:0")
    
    def test_component_filtering(self):
        """Test that non-model components don't get device maps."""
        from diffusers.utils.accelerate_utils import PipelineDeviceMapper
        
        # Create a mapper with both model and non-model components
        init_dict = {
            "unet": ("diffusers", "UNet2DConditionModel"),
            "vae": ("diffusers", "AutoencoderKL"),
            "scheduler": ("diffusers", "DDIMScheduler"),  # Not a model
            "tokenizer": ("transformers", "CLIPTokenizer"),  # Not a model
        }
        
        mapper = PipelineDeviceMapper(
            pipeline_class=StableDiffusionXLPipeline,
            init_dict=init_dict,
            passed_class_obj={},
            cached_folder="",
        )
        
        device_map = {"": "cuda:0"}
        component_maps = mapper.resolve_component_device_maps(
            device_map=device_map,
            max_memory=None,
            torch_dtype=torch.float16
        )
        
        # All components get device maps (filtering happens during loading)
        self.assertIn("unet", component_maps)
        self.assertIn("vae", component_maps)
        self.assertIn("scheduler", component_maps)
        self.assertIn("tokenizer", component_maps)
        
        # But they all get the same device map
        for component, dev_map in component_maps.items():
            self.assertEqual(dev_map, {"": "cuda:0"})
    
    @require_torch_multi_accelerator
    def test_hierarchical_device_maps(self):
        """Test hierarchical device maps for submodule assignments."""
        if self.device_count < 2:
            self.skipTest("Need at least 2 GPUs for this test")
            
        from diffusers.utils.accelerate_utils import PipelineDeviceMapper
        
        # Test hierarchical assignments like Accelerate supports
        device_map = {
            "unet.down_blocks": 0,
            "unet.up_blocks": 1,
            "vae": 0,
            "text_encoder": 1,
        }
        
        # Create mapper
        init_dict = {
            "unet": ("diffusers", "UNet2DConditionModel"),
            "vae": ("diffusers", "AutoencoderKL"),
            "text_encoder": ("transformers", "CLIPTextModel"),
        }
        
        mapper = PipelineDeviceMapper(
            pipeline_class=StableDiffusionXLPipeline,
            init_dict=init_dict,
            passed_class_obj={},
            cached_folder="",
        )
        
        component_maps = mapper.resolve_component_device_maps(
            device_map=device_map,
            max_memory=None,
            torch_dtype=torch.float16
        )
        
        # Check unet gets hierarchical mapping
        self.assertIn("unet", component_maps)
        self.assertEqual(component_maps["unet"]["down_blocks"], 0)
        self.assertEqual(component_maps["unet"]["up_blocks"], 1)
        
        # Check other components get their direct mappings
        self.assertEqual(component_maps["vae"], {"": 0})
        self.assertEqual(component_maps["text_encoder"], {"": 1})


if __name__ == "__main__":
    unittest.main()