import inspect
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AnimateDiffPAGPipeline,
    AnimateDiffPipeline,
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LCMScheduler,
    MotionAdapter,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    UNetMotionModel,
)
from diffusers.models.attention import FreeNoiseTransformerBlock
from diffusers.utils import is_xformers_available
from diffusers.utils.testing_utils import torch_device

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import (
    IPAdapterTesterMixin,
    PipelineFromPipeTesterMixin,
    PipelineTesterMixin,
    SDFunctionTesterMixin,
)


def to_np(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    return tensor


class AnimateDiffPAGPipelineFastTests(
    IPAdapterTesterMixin, SDFunctionTesterMixin, PipelineTesterMixin, PipelineFromPipeTesterMixin, unittest.TestCase
):
    pipeline_class = AnimateDiffPAGPipeline
    params = TEXT_TO_IMAGE_PARAMS.union({"pag_scale", "pag_adaptive_scale"})
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "return_dict",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        ]
    )

    def get_dummy_components(self):
        cross_attention_dim = 8
        block_out_channels = (8, 8)

        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=block_out_channels,
            layers_per_block=2,
            sample_size=8,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=cross_attention_dim,
            norm_num_groups=2,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            clip_sample=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=block_out_channels,
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=2,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=cross_attention_dim,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        motion_adapter = MotionAdapter(
            block_out_channels=block_out_channels,
            motion_layers_per_block=2,
            motion_norm_num_groups=2,
            motion_num_attention_heads=4,
        )

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "motion_adapter": motion_adapter,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "feature_extractor": None,
            "image_encoder": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 7.5,
            "pag_scale": 3.0,
            "output_type": "pt",
        }
        return inputs

    def test_from_pipe_consistent_config(self):
        assert self.original_pipeline_class == StableDiffusionPipeline
        original_repo = "hf-internal-testing/tinier-stable-diffusion-pipe"
        original_kwargs = {"requires_safety_checker": False}

        # create original_pipeline_class(sd)
        pipe_original = self.original_pipeline_class.from_pretrained(original_repo, **original_kwargs)

        # original_pipeline_class(sd) -> pipeline_class
        pipe_components = self.get_dummy_components()
        pipe_additional_components = {}
        for name, component in pipe_components.items():
            if name not in pipe_original.components:
                pipe_additional_components[name] = component

        pipe = self.pipeline_class.from_pipe(pipe_original, **pipe_additional_components)

        # pipeline_class -> original_pipeline_class(sd)
        original_pipe_additional_components = {}
        for name, component in pipe_original.components.items():
            if name not in pipe.components or not isinstance(component, pipe.components[name].__class__):
                original_pipe_additional_components[name] = component

        pipe_original_2 = self.original_pipeline_class.from_pipe(pipe, **original_pipe_additional_components)

        # compare the config
        original_config = {k: v for k, v in pipe_original.config.items() if not k.startswith("_")}
        original_config_2 = {k: v for k, v in pipe_original_2.config.items() if not k.startswith("_")}
        assert original_config_2 == original_config

    def test_motion_unet_loading(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        assert isinstance(pipe.unet, UNetMotionModel)

    @unittest.skip("Attention slicing is not enabled in this pipeline")
    def test_attention_slicing_forward_pass(self):
        pass

    def test_ip_adapter(self):
        expected_pipe_slice = None

        if torch_device == "cpu":
            expected_pipe_slice = np.array(
                [
                    0.5068,
                    0.5294,
                    0.4926,
                    0.4810,
                    0.4188,
                    0.5935,
                    0.5295,
                    0.3947,
                    0.5300,
                    0.4706,
                    0.3950,
                    0.4737,
                    0.4072,
                    0.3227,
                    0.5481,
                    0.4864,
                    0.4518,
                    0.5315,
                    0.5979,
                    0.5374,
                    0.3503,
                    0.5275,
                    0.6067,
                    0.4914,
                    0.5440,
                    0.4775,
                    0.5538,
                ]
            )
        return super().test_ip_adapter(expected_pipe_slice=expected_pipe_slice)

    def test_dict_tuple_outputs_equivalent(self):
        expected_slice = None
        if torch_device == "cpu":
            expected_slice = np.array([0.5295, 0.3947, 0.5300, 0.4864, 0.4518, 0.5315, 0.5440, 0.4775, 0.5538])
        return super().test_dict_tuple_outputs_equivalent(expected_slice=expected_slice)

    @unittest.skipIf(torch_device != "cuda", reason="CUDA and CPU are required to switch devices")
    def test_to_device(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        pipe.to("cpu")
        # pipeline creates a new motion UNet under the hood. So we need to check the device from pipe.components
        model_devices = [
            component.device.type for component in pipe.components.values() if hasattr(component, "device")
        ]
        self.assertTrue(all(device == "cpu" for device in model_devices))

        output_cpu = pipe(**self.get_dummy_inputs("cpu"))[0]
        self.assertTrue(np.isnan(output_cpu).sum() == 0)

        pipe.to("cuda")
        model_devices = [
            component.device.type for component in pipe.components.values() if hasattr(component, "device")
        ]
        self.assertTrue(all(device == "cuda" for device in model_devices))

        output_cuda = pipe(**self.get_dummy_inputs("cuda"))[0]
        self.assertTrue(np.isnan(to_np(output_cuda)).sum() == 0)

    def test_to_dtype(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        # pipeline creates a new motion UNet under the hood. So we need to check the dtype from pipe.components
        model_dtypes = [component.dtype for component in pipe.components.values() if hasattr(component, "dtype")]
        self.assertTrue(all(dtype == torch.float32 for dtype in model_dtypes))

        pipe.to(dtype=torch.float16)
        model_dtypes = [component.dtype for component in pipe.components.values() if hasattr(component, "dtype")]
        self.assertTrue(all(dtype == torch.float16 for dtype in model_dtypes))

    def test_prompt_embeds(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        pipe.to(torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        inputs.pop("prompt")
        inputs["prompt_embeds"] = torch.randn((1, 4, pipe.text_encoder.config.hidden_size), device=torch_device)
        pipe(**inputs)

    def test_free_init(self):
        components = self.get_dummy_components()
        pipe: AnimateDiffPAGPipeline = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        pipe.to(torch_device)

        inputs_normal = self.get_dummy_inputs(torch_device)
        frames_normal = pipe(**inputs_normal).frames[0]

        pipe.enable_free_init(
            num_iters=2,
            use_fast_sampling=True,
            method="butterworth",
            order=4,
            spatial_stop_frequency=0.25,
            temporal_stop_frequency=0.25,
        )
        inputs_enable_free_init = self.get_dummy_inputs(torch_device)
        frames_enable_free_init = pipe(**inputs_enable_free_init).frames[0]

        pipe.disable_free_init()
        inputs_disable_free_init = self.get_dummy_inputs(torch_device)
        frames_disable_free_init = pipe(**inputs_disable_free_init).frames[0]

        sum_enabled = np.abs(to_np(frames_normal) - to_np(frames_enable_free_init)).sum()
        max_diff_disabled = np.abs(to_np(frames_normal) - to_np(frames_disable_free_init)).max()
        self.assertGreater(
            sum_enabled, 1e1, "Enabling of FreeInit should lead to results different from the default pipeline results"
        )
        self.assertLess(
            max_diff_disabled,
            1e-3,
            "Disabling of FreeInit should lead to results similar to the default pipeline results",
        )

    def test_free_init_with_schedulers(self):
        components = self.get_dummy_components()
        pipe: AnimateDiffPAGPipeline = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        pipe.to(torch_device)

        inputs_normal = self.get_dummy_inputs(torch_device)
        frames_normal = pipe(**inputs_normal).frames[0]

        schedulers_to_test = [
            DPMSolverMultistepScheduler.from_config(
                components["scheduler"].config,
                timestep_spacing="linspace",
                beta_schedule="linear",
                algorithm_type="dpmsolver++",
                steps_offset=1,
                clip_sample=False,
            ),
            LCMScheduler.from_config(
                components["scheduler"].config,
                timestep_spacing="linspace",
                beta_schedule="linear",
                steps_offset=1,
                clip_sample=False,
            ),
        ]
        components.pop("scheduler")

        for scheduler in schedulers_to_test:
            components["scheduler"] = scheduler
            pipe: AnimateDiffPAGPipeline = self.pipeline_class(**components)
            pipe.set_progress_bar_config(disable=None)
            pipe.to(torch_device)

            pipe.enable_free_init(num_iters=2, use_fast_sampling=False)

            inputs = self.get_dummy_inputs(torch_device)
            frames_enable_free_init = pipe(**inputs).frames[0]
            sum_enabled = np.abs(to_np(frames_normal) - to_np(frames_enable_free_init)).sum()

            self.assertGreater(
                sum_enabled,
                1e1,
                "Enabling of FreeInit should lead to results different from the default pipeline results",
            )

    def test_free_noise_blocks(self):
        components = self.get_dummy_components()
        pipe: AnimateDiffPAGPipeline = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        pipe.to(torch_device)

        pipe.enable_free_noise()
        for block in pipe.unet.down_blocks:
            for motion_module in block.motion_modules:
                for transformer_block in motion_module.transformer_blocks:
                    self.assertTrue(
                        isinstance(transformer_block, FreeNoiseTransformerBlock),
                        "Motion module transformer blocks must be an instance of `FreeNoiseTransformerBlock` after enabling FreeNoise.",
                    )

        pipe.disable_free_noise()
        for block in pipe.unet.down_blocks:
            for motion_module in block.motion_modules:
                for transformer_block in motion_module.transformer_blocks:
                    self.assertFalse(
                        isinstance(transformer_block, FreeNoiseTransformerBlock),
                        "Motion module transformer blocks must not be an instance of `FreeNoiseTransformerBlock` after disabling FreeNoise.",
                    )

    def test_free_noise(self):
        components = self.get_dummy_components()
        pipe: AnimateDiffPAGPipeline = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        pipe.to(torch_device)

        inputs_normal = self.get_dummy_inputs(torch_device)
        frames_normal = pipe(**inputs_normal).frames[0]

        for context_length in [8, 9]:
            for context_stride in [4, 6]:
                pipe.enable_free_noise(context_length, context_stride)

                inputs_enable_free_noise = self.get_dummy_inputs(torch_device)
                frames_enable_free_noise = pipe(**inputs_enable_free_noise).frames[0]

                pipe.disable_free_noise()

                inputs_disable_free_noise = self.get_dummy_inputs(torch_device)
                frames_disable_free_noise = pipe(**inputs_disable_free_noise).frames[0]

                sum_enabled = np.abs(to_np(frames_normal) - to_np(frames_enable_free_noise)).sum()
                max_diff_disabled = np.abs(to_np(frames_normal) - to_np(frames_disable_free_noise)).max()
                self.assertGreater(
                    sum_enabled,
                    1e1,
                    "Enabling of FreeNoise should lead to results different from the default pipeline results",
                )
                self.assertLess(
                    max_diff_disabled,
                    1e-4,
                    "Disabling of FreeNoise should lead to results similar to the default pipeline results",
                )

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_without_offload = pipe(**inputs).frames[0]
        output_without_offload = (
            output_without_offload.cpu() if torch.is_tensor(output_without_offload) else output_without_offload
        )

        pipe.enable_xformers_memory_efficient_attention()
        inputs = self.get_dummy_inputs(torch_device)
        output_with_offload = pipe(**inputs).frames[0]
        output_with_offload = (
            output_with_offload.cpu() if torch.is_tensor(output_with_offload) else output_without_offload
        )

        max_diff = np.abs(to_np(output_with_offload) - to_np(output_without_offload)).max()
        self.assertLess(max_diff, 1e-4, "XFormers attention should not affect the inference results")

    def test_vae_slicing(self):
        return super().test_vae_slicing(image_count=2)

    def test_pag_disable_enable(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        # base pipeline (expect same output when pag is disabled)
        components.pop("pag_applied_layers", None)
        pipe_sd = AnimateDiffPipeline(**components)
        pipe_sd = pipe_sd.to(device)
        pipe_sd.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["pag_scale"]
        assert (
            "pag_scale" not in inspect.signature(pipe_sd.__call__).parameters
        ), f"`pag_scale` should not be a call parameter of the base pipeline {pipe_sd.__class__.__name__}."
        out = pipe_sd(**inputs).frames[0, -3:, -3:, -1]

        components = self.get_dummy_components()

        # pag disabled with pag_scale=0.0
        pipe_pag = self.pipeline_class(**components)
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["pag_scale"] = 0.0
        out_pag_disabled = pipe_pag(**inputs).frames[0, -3:, -3:, -1]

        # pag enabled
        pipe_pag = self.pipeline_class(**components)
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        out_pag_enabled = pipe_pag(**inputs).frames[0, -3:, -3:, -1]

        assert np.abs(out.flatten() - out_pag_disabled.flatten()).max() < 1e-3
        assert np.abs(out.flatten() - out_pag_enabled.flatten()).max() > 1e-3

    def test_pag_applied_layers(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        # base pipeline
        components.pop("pag_applied_layers", None)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        # pag_applied_layers = ["mid","up","down"] should apply to all self-attention layers
        # Note that for motion modules in AnimateDiff, both attn1 and attn2 are self-attention
        all_self_attn_layers = [
            k for k in pipe.unet.attn_processors.keys() if "attn1" in k or ("motion_modules" in k and "attn2" in k)
        ]
        original_attn_procs = pipe.unet.attn_processors
        pag_layers = [
            "down",
            "mid",
            "up",
        ]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_layers)

        # pag_applied_layers = ["mid"], or ["mid_block.0"] should apply to all self-attention layers in mid_block, i.e.
        # mid_block.motion_modules.0.transformer_blocks.0.attn1.processor
        # mid_block.attentions.0.transformer_blocks.0.attn1.processor
        all_self_attn_mid_layers = [
            "mid_block.attentions.0.transformer_blocks.0.attn1.processor",
            "mid_block.motion_modules.0.transformer_blocks.0.attn1.processor",
            "mid_block.motion_modules.0.transformer_blocks.0.attn2.processor",
        ]
        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["mid"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_mid_layers)

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["mid_block"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_mid_layers)

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["mid_block.(attentions|motion_modules)"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_mid_layers)

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["mid_block.attentions.1"]
        with self.assertRaises(ValueError):
            pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)

        # pag_applied_layers = "down" should apply to all self-attention layers in down_blocks
        # down_blocks.1.(attentions|motion_modules).0.transformer_blocks.0.attn1.processor
        # down_blocks.1.(attentions|motion_modules).0.transformer_blocks.1.attn1.processor
        # down_blocks.1.(attentions|motion_modules).0.transformer_blocks.0.attn1.processor

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["down"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert len(pipe.pag_attn_processors) == 10

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["down_blocks.0"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert (len(pipe.pag_attn_processors)) == 6

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["blocks.1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert len(pipe.pag_attn_processors) == 10

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["motion_modules.42"]
        with self.assertRaises(ValueError):
            pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
