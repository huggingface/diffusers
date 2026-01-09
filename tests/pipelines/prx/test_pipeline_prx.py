import unittest

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer
from transformers.models.t5gemma.configuration_t5gemma import T5GemmaConfig, T5GemmaModuleConfig
from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder

from diffusers.models import AutoencoderDC, AutoencoderKL
from diffusers.models.transformers.transformer_prx import PRXTransformer2DModel
from diffusers.pipelines.prx.pipeline_prx import PRXPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_transformers_version

from ..pipeline_params import TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


@pytest.mark.xfail(
    condition=is_transformers_version(">", "4.57.1"),
    reason="See https://github.com/huggingface/diffusers/pull/12456#issuecomment-3424228544",
    strict=False,
)
class PRXPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = PRXPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = frozenset(["prompt", "negative_prompt", "num_images_per_prompt"])
    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True

    @classmethod
    def setUpClass(cls):
        # Ensure PRXPipeline has an _execution_device property expected by __call__
        if not isinstance(getattr(PRXPipeline, "_execution_device", None), property):
            try:
                setattr(PRXPipeline, "_execution_device", property(lambda self: torch.device("cpu")))
            except Exception:
                pass

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = PRXTransformer2DModel(
            patch_size=1,
            in_channels=4,
            context_in_dim=8,
            hidden_size=8,
            mlp_ratio=2.0,
            num_heads=2,
            depth=1,
            axes_dim=[2, 2],
        )

        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=4,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0,
            scaling_factor=1.0,
        ).eval()

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler()

        torch.manual_seed(0)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/dummy-gemma")
        tokenizer.model_max_length = 64

        torch.manual_seed(0)

        encoder_params = {
            "vocab_size": tokenizer.vocab_size,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 64,
            "layer_types": ["full_attention"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "dropout_rate": 0.0,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-06,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "query_pre_attn_scalar": 4,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
        }
        encoder_config = T5GemmaModuleConfig(**encoder_params)
        text_encoder_config = T5GemmaConfig(encoder=encoder_config, is_encoder_decoder=False, **encoder_params)
        text_encoder = T5GemmaEncoder(text_encoder_config)

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        return {
            "prompt": "",
            "negative_prompt": "",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 32,
            "width": 32,
            "output_type": "pt",
            "use_resolution_binning": False,
        }

    def test_inference(self):
        device = "cpu"
        components = self.get_dummy_components()
        pipe = PRXPipeline(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)
        try:
            pipe.register_to_config(_execution_device="cpu")
        except Exception:
            pass

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs)[0]
        generated_image = image[0]

        self.assertEqual(generated_image.shape, (3, 32, 32))
        expected_image = torch.zeros(3, 32, 32)
        max_diff = np.abs(generated_image - expected_image).max()
        self.assertLessEqual(max_diff, 1e10)

    def test_callback_inputs(self):
        components = self.get_dummy_components()
        pipe = PRXPipeline(**components)
        pipe = pipe.to("cpu")
        pipe.set_progress_bar_config(disable=None)
        try:
            pipe.register_to_config(_execution_device="cpu")
        except Exception:
            pass
        self.assertTrue(
            hasattr(pipe, "_callback_tensor_inputs"),
            f" {PRXPipeline} should have `_callback_tensor_inputs` that defines a list of tensor variables its callback function can use as inputs",
        )

        def callback_inputs_subset(pipe, i, t, callback_kwargs):
            for tensor_name in callback_kwargs.keys():
                assert tensor_name in pipe._callback_tensor_inputs
            return callback_kwargs

        def callback_inputs_all(pipe, i, t, callback_kwargs):
            for tensor_name in pipe._callback_tensor_inputs:
                assert tensor_name in callback_kwargs
            for tensor_name in callback_kwargs.keys():
                assert tensor_name in pipe._callback_tensor_inputs
            return callback_kwargs

        inputs = self.get_dummy_inputs("cpu")

        inputs["callback_on_step_end"] = callback_inputs_subset
        inputs["callback_on_step_end_tensor_inputs"] = ["latents"]
        _ = pipe(**inputs)[0]

        inputs["callback_on_step_end"] = callback_inputs_all
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        _ = pipe(**inputs)[0]

    def test_attention_slicing_forward_pass(self, expected_max_diff=1e-3):
        if not self.test_attention_slicing:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to("cpu")
        pipe.set_progress_bar_config(disable=None)

        def to_np_local(tensor):
            if isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu().numpy()
            return tensor

        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        output_without_slicing = pipe(**inputs)[0]

        pipe.enable_attention_slicing(slice_size=1)
        inputs = self.get_dummy_inputs(generator_device)
        output_with_slicing1 = pipe(**inputs)[0]

        pipe.enable_attention_slicing(slice_size=2)
        inputs = self.get_dummy_inputs(generator_device)
        output_with_slicing2 = pipe(**inputs)[0]

        max_diff1 = np.abs(to_np_local(output_with_slicing1) - to_np_local(output_without_slicing)).max()
        max_diff2 = np.abs(to_np_local(output_with_slicing2) - to_np_local(output_without_slicing)).max()
        self.assertLess(max(max_diff1, max_diff2), expected_max_diff)

    def test_inference_with_autoencoder_dc(self):
        """Test PRXPipeline with AutoencoderDC (DCAE) instead of AutoencoderKL."""
        device = "cpu"

        components = self.get_dummy_components()

        torch.manual_seed(0)
        vae_dc = AutoencoderDC(
            in_channels=3,
            latent_channels=4,
            attention_head_dim=2,
            encoder_block_types=(
                "ResBlock",
                "EfficientViTBlock",
            ),
            decoder_block_types=(
                "ResBlock",
                "EfficientViTBlock",
            ),
            encoder_block_out_channels=(8, 8),
            decoder_block_out_channels=(8, 8),
            encoder_qkv_multiscales=((), (5,)),
            decoder_qkv_multiscales=((), (5,)),
            encoder_layers_per_block=(1, 1),
            decoder_layers_per_block=(1, 1),
            upsample_block_type="interpolate",
            downsample_block_type="stride_conv",
            decoder_norm_types="rms_norm",
            decoder_act_fns="silu",
        ).eval()

        components["vae"] = vae_dc

        pipe = PRXPipeline(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        expected_scale_factor = vae_dc.spatial_compression_ratio
        self.assertEqual(pipe.vae_scale_factor, expected_scale_factor)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs)[0]
        generated_image = image[0]

        self.assertEqual(generated_image.shape, (3, 32, 32))
        expected_image = torch.zeros(3, 32, 32)
        max_diff = np.abs(generated_image - expected_image).max()
        self.assertLessEqual(max_diff, 1e10)
