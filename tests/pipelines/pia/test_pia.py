import random
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LCMScheduler,
    MotionAdapter,
    PIAPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    UNetMotionModel,
)
from diffusers.utils import is_xformers_available, logging
from diffusers.utils.testing_utils import floats_tensor, require_accelerator, torch_device

from ..test_pipelines_common import IPAdapterTesterMixin, PipelineFromPipeTesterMixin, PipelineTesterMixin


def to_np(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    return tensor


class PIAPipelineFastTests(IPAdapterTesterMixin, PipelineTesterMixin, PipelineFromPipeTesterMixin, unittest.TestCase):
    pipeline_class = PIAPipeline
    params = frozenset(
        [
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
            "cross_attention_kwargs",
        ]
    )
    batch_params = frozenset(["prompt", "image", "generator"])
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
    test_layerwise_casting = True
    test_group_offloading = True

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
        torch.manual_seed(0)
        motion_adapter = MotionAdapter(
            block_out_channels=block_out_channels,
            motion_layers_per_block=2,
            motion_norm_num_groups=2,
            motion_num_attention_heads=4,
            conv_in_channels=9,
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

        image = floats_tensor((1, 3, 8, 8), rng=random.Random(seed)).to(device)
        inputs = {
            "image": image,
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 7.5,
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

    def test_ip_adapter(self):
        expected_pipe_slice = None

        if torch_device == "cpu":
            expected_pipe_slice = np.array(
                [
                    0.5475,
                    0.5769,
                    0.4873,
                    0.5064,
                    0.4445,
                    0.5876,
                    0.5453,
                    0.4102,
                    0.5247,
                    0.5370,
                    0.3406,
                    0.4322,
                    0.3991,
                    0.3756,
                    0.5438,
                    0.4780,
                    0.5087,
                    0.5248,
                    0.6243,
                    0.5506,
                    0.3491,
                    0.5440,
                    0.6111,
                    0.5122,
                    0.5326,
                    0.5180,
                    0.5538,
                ]
            )
        return super().test_ip_adapter(expected_pipe_slice=expected_pipe_slice)

    def test_dict_tuple_outputs_equivalent(self):
        expected_slice = None
        if torch_device == "cpu":
            expected_slice = np.array([0.5476, 0.4092, 0.5289, 0.4755, 0.5092, 0.5186, 0.5403, 0.5287, 0.5467])
        return super().test_dict_tuple_outputs_equivalent(expected_slice=expected_slice)

    @unittest.skip("Attention slicing is not enabled in this pipeline")
    def test_attention_slicing_forward_pass(self):
        pass

    def test_inference_batch_single_identical(
        self,
        batch_size=2,
        expected_max_diff=1e-4,
        additional_params_copy_to_batched_inputs=["num_inference_steps"],
    ):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for components in pipe.components.values():
            if hasattr(components, "set_default_attn_processor"):
                components.set_default_attn_processor()

        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs(torch_device)
        # Reset generator in case it is has been used in self.get_dummy_inputs
        inputs["generator"] = self.get_generator(0)

        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=diffusers.logging.FATAL)

        # batchify inputs
        batched_inputs = {}
        batched_inputs.update(inputs)

        for name in self.batch_params:
            if name not in inputs:
                continue

            value = inputs[name]
            if name == "prompt":
                len_prompt = len(value)
                batched_inputs[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]
                batched_inputs[name][-1] = 100 * "very long"

            else:
                batched_inputs[name] = batch_size * [value]

        if "generator" in inputs:
            batched_inputs["generator"] = [self.get_generator(i) for i in range(batch_size)]

        if "batch_size" in inputs:
            batched_inputs["batch_size"] = batch_size

        for arg in additional_params_copy_to_batched_inputs:
            batched_inputs[arg] = inputs[arg]

        output = pipe(**inputs)
        output_batch = pipe(**batched_inputs)

        assert output_batch[0].shape[0] == batch_size

        max_diff = np.abs(to_np(output_batch[0][0]) - to_np(output[0][0])).max()
        assert max_diff < expected_max_diff

    @require_accelerator
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

        pipe.to(torch_device)
        model_devices = [
            component.device.type for component in pipe.components.values() if hasattr(component, "device")
        ]
        self.assertTrue(all(device == torch_device for device in model_devices))

        output_device = pipe(**self.get_dummy_inputs(torch_device))[0]
        self.assertTrue(np.isnan(to_np(output_device)).sum() == 0)

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
        pipe = self.pipeline_class(**components)
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
            1e-4,
            "Disabling of FreeInit should lead to results similar to the default pipeline results",
        )

    def test_free_init_with_schedulers(self):
        components = self.get_dummy_components()
        pipe: PIAPipeline = self.pipeline_class(**components)
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
            pipe: PIAPipeline = self.pipeline_class(**components)
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

    def test_encode_prompt_works_in_isolation(self):
        extra_required_param_value_dict = {
            "device": torch.device(torch_device).type,
            "num_images_per_prompt": 1,
            "do_classifier_free_guidance": self.get_dummy_inputs(device=torch_device).get("guidance_scale", 1.0) > 1.0,
        }
        return super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict)
