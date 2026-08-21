import gc

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxControlPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.utils import load_image, logging

from ...models.testing_utils.lora import check_if_lora_correctly_set
from ...testing_utils import (
    CaptureLogger,
    assert_tensors_close,
    backend_empty_cache,
    nightly,
    numpy_cosine_similarity_distance,
    require_big_accelerator,
    require_peft_backend,
    require_torch_accelerator,
    torch_device,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    LoraTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
    check_qkv_fused_layers_exist,
)


class FluxControlPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = FluxControlPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "prompt_embeds", "pooled_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])
    output_shape = (3, 8, 8)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=8,
            out_channels=4,
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
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder_2 = T5EncoderModel(config)

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
        }

    def get_dummy_inputs(self):
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "control_image": Image.new("RGB", (16, 16), 0),
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 48,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }
        return inputs


class TestFluxControlPipeline(FluxControlPipelineTesterConfig, PipelineTesterMixin):
    def test_flux_different_prompts(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs()
        inputs["prompt_2"] = "a different prompt"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = (output_same_prompt - output_different_prompts).abs().max()

        # Outputs should be different here
        # For some reasons, they don't show large differences
        assert max_diff > 1e-6, "Outputs should be different for different prompts."

    def test_fused_qkv_projections(self):
        pipe = self.get_pipeline().to(torch_device)

        image_slice = self.run_pipe(pipe)[0, -3:, -3:, -1]

        # TODO (sayakpaul): will refactor this once `fuse_qkv_projections()` has been added
        # to the pipeline level.
        pipe.transformer.fuse_qkv_projections()
        assert check_qkv_fused_layers_exist(pipe.transformer, ["to_qkv"]), (
            "Something wrong with the fused attention layers. Expected all the attention projections to be fused."
        )

        image_slice_fused = self.run_pipe(pipe)[0, -3:, -3:, -1]

        pipe.transformer.unfuse_qkv_projections()
        image_slice_disabled = self.run_pipe(pipe)[0, -3:, -3:, -1]

        assert_tensors_close(
            image_slice_fused,
            image_slice,
            atol=1e-3,
            rtol=1e-3,
            msg="Fusion of QKV projections shouldn't affect the outputs.",
        )
        assert_tensors_close(
            image_slice_disabled,
            image_slice_fused,
            atol=1e-3,
            rtol=1e-3,
            msg="Outputs, with QKV projection fusion enabled, shouldn't change when fused QKV projections are disabled.",
        )
        assert_tensors_close(
            image_slice_disabled,
            image_slice,
            atol=1e-2,
            rtol=1e-2,
            msg="Original outputs should match when fused QKV projections are disabled.",
        )

    def test_flux_image_output_shape(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()

        height_width_pairs = [(32, 32), (72, 57)]
        for height, width in height_width_pairs:
            expected_height = height - height % (pipe.vae_scale_factor * 2)
            expected_width = width - width % (pipe.vae_scale_factor * 2)

            inputs.update({"height": height, "width": width})
            image = pipe(**inputs).images[0]
            _, output_height, output_width = image.shape
            assert (output_height, output_width) == (expected_height, expected_width), (
                f"Output shape {image.shape} does not match expected shape {(expected_height, expected_width)}"
            )


class TestFluxControlPipelineMemory(FluxControlPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Flux Control pipeline."""


class TestFluxControlPipelineLoRA(FluxControlPipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the Flux Control pipeline.

    On top of the shared suite, this covers the Control-LoRA specifics: `x_embedder` shape expansion (and the
    zero-padding of regular LoRAs loaded into an expanded denoiser) plus the norm layers that Control LoRAs ship
    alongside the LoRA weights.
    """

    def run_pipe_without_control(self, pipe):
        """`run_pipe` for a plain `FluxPipeline`, which doesn't accept a `control_image`."""
        inputs = self.get_dummy_inputs()
        inputs.pop("control_image")
        torch.manual_seed(0)
        return pipe(**inputs)[0]

    def test_with_norm_in_state_dict(self, base_pipe_output):
        pipe = self.get_pipeline().to(torch_device)

        logger = logging.get_logger("diffusers.loaders.lora_pipeline")
        logger.setLevel(logging.INFO)

        for norm_layer in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]:
            norm_state_dict = {}
            for name, module in pipe.transformer.named_modules():
                if norm_layer not in name or not hasattr(module, "weight") or module.weight is None:
                    continue
                norm_state_dict[f"transformer.{name}.weight"] = torch.randn(
                    module.weight.shape, device=module.weight.device, dtype=module.weight.dtype
                )

                with CaptureLogger(logger) as cap_logger:
                    pipe.load_lora_weights(norm_state_dict)
                lora_load_output = self.run_pipe(pipe)

                assert (
                    "The provided state dict contains normalization layers in addition to LoRA layers"
                    in cap_logger.out
                )
                assert len(pipe.transformer._transformer_norm_layers) > 0

                pipe.unload_lora_weights()
                lora_unload_output = self.run_pipe(pipe)

            assert pipe.transformer._transformer_norm_layers is None
            assert_tensors_close(
                lora_unload_output,
                base_pipe_output,
                atol=1e-5,
                rtol=1e-5,
                msg="Unloading the norm layers should restore the original output.",
            )
            assert not torch.allclose(base_pipe_output, lora_load_output, atol=1e-6, rtol=1e-6), (
                f"{norm_layer} is tested"
            )

        with CaptureLogger(logger) as cap_logger:
            for key in list(norm_state_dict.keys()):
                norm_state_dict[key.replace("norm", "norm_k_something_random")] = norm_state_dict.pop(key)
            pipe.load_lora_weights(norm_state_dict)

        assert "Unsupported keys found in state dict when trying to load normalization layers" in cap_logger.out

    def test_lora_parameter_expanded_shapes(self, base_pipe_output):
        components = self.get_dummy_components()
        pipe = self.get_pipeline(**components).to(torch_device)

        logger = logging.get_logger("diffusers.loaders.lora_pipeline")
        logger.setLevel(logging.DEBUG)

        # Change the transformer config to mimic a real use case.
        num_channels_without_control = 4
        transformer = FluxTransformer2DModel.from_config(
            components["transformer"].config, in_channels=num_channels_without_control
        ).to(torch_device)
        transformer.eval()
        assert transformer.config.in_channels == num_channels_without_control, (
            f"Expected {num_channels_without_control} channels in the modified transformer but has "
            f"{transformer.config.in_channels=}"
        )

        original_transformer_state_dict = pipe.transformer.state_dict()
        x_embedder_weight = original_transformer_state_dict.pop("x_embedder.weight")
        incompatible_keys = transformer.load_state_dict(original_transformer_state_dict, strict=False)
        assert "x_embedder.weight" in incompatible_keys.missing_keys, (
            "Could not find x_embedder.weight in the missing keys."
        )
        transformer.x_embedder.weight.data.copy_(x_embedder_weight[..., :num_channels_without_control])
        pipe.transformer = transformer

        out_features, in_features = pipe.transformer.x_embedder.weight.shape
        rank = 4

        dummy_lora_A = torch.nn.Linear(2 * in_features, rank, bias=False)
        dummy_lora_B = torch.nn.Linear(rank, out_features, bias=False)
        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": dummy_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": dummy_lora_B.weight,
        }
        with CaptureLogger(logger) as cap_logger:
            pipe.load_lora_weights(lora_state_dict, "adapter-1")

        assert check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in denoiser"

        lora_out = self.run_pipe(pipe)

        assert not torch.allclose(base_pipe_output, lora_out, atol=1e-4, rtol=1e-4)
        assert pipe.transformer.x_embedder.weight.data.shape[1] == 2 * in_features
        assert pipe.transformer.config.in_channels == 2 * in_features
        assert cap_logger.out.startswith("Expanding the nn.Linear input/output features for module")

        # Testing opposite direction where the LoRA params are zero-padded.
        pipe = self.get_pipeline().to(torch_device)
        dummy_lora_A = torch.nn.Linear(1, rank, bias=False)
        dummy_lora_B = torch.nn.Linear(rank, out_features, bias=False)
        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": dummy_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": dummy_lora_B.weight,
        }
        with CaptureLogger(logger) as cap_logger:
            pipe.load_lora_weights(lora_state_dict, "adapter-1")

        assert check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in denoiser"

        lora_out = self.run_pipe(pipe)

        assert not torch.allclose(base_pipe_output, lora_out, atol=1e-4, rtol=1e-4)
        assert pipe.transformer.x_embedder.weight.data.shape[1] == 2 * in_features
        assert pipe.transformer.config.in_channels == 2 * in_features
        assert "The following LoRA modules were zero padded to match the state dict of" in cap_logger.out

    def test_normal_lora_with_expanded_lora_raises_error(self):
        # Test the following situation. Load a regular LoRA (such as the ones trained on Flux.1-Dev). And then
        # load shape expanded LoRA (such as Control LoRA).
        components = self.get_dummy_components()

        # Change the transformer config to mimic a real use case.
        num_channels_without_control = 4
        components["transformer"] = FluxTransformer2DModel.from_config(
            components["transformer"].config, in_channels=num_channels_without_control
        ).to(torch_device)

        pipe = self.get_pipeline(**components).to(torch_device)

        logger = logging.get_logger("diffusers.loaders.lora_pipeline")
        logger.setLevel(logging.DEBUG)

        out_features, in_features = pipe.transformer.x_embedder.weight.shape
        rank = 4

        shape_expander_lora_A = torch.nn.Linear(2 * in_features, rank, bias=False)
        shape_expander_lora_B = torch.nn.Linear(rank, out_features, bias=False)
        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": shape_expander_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": shape_expander_lora_B.weight,
        }
        with CaptureLogger(logger) as cap_logger:
            pipe.load_lora_weights(lora_state_dict, "adapter-1")

        assert check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in denoiser"
        assert pipe.get_active_adapters() == ["adapter-1"]
        assert pipe.transformer.x_embedder.weight.data.shape[1] == 2 * in_features
        assert pipe.transformer.config.in_channels == 2 * in_features
        assert cap_logger.out.startswith("Expanding the nn.Linear input/output features for module")

        lora_output = self.run_pipe(pipe)

        normal_lora_A = torch.nn.Linear(in_features, rank, bias=False)
        normal_lora_B = torch.nn.Linear(rank, out_features, bias=False)
        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": normal_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": normal_lora_B.weight,
        }

        with CaptureLogger(logger) as cap_logger:
            pipe.load_lora_weights(lora_state_dict, "adapter-2")

        assert check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in denoiser"
        assert "The following LoRA modules were zero padded to match the state dict of" in cap_logger.out
        assert pipe.get_active_adapters() == ["adapter-2"]

        lora_output_2 = self.run_pipe(pipe)
        assert not torch.allclose(lora_output, lora_output_2, atol=1e-3, rtol=1e-3)

        # Test the opposite case where the first lora has the correct input features and the second lora has expanded
        # input features. This should raise a runtime error on input shapes being incompatible.
        components = self.get_dummy_components()
        # Change the transformer config to mimic a real use case.
        components["transformer"] = FluxTransformer2DModel.from_config(
            components["transformer"].config, in_channels=num_channels_without_control
        ).to(torch_device)

        pipe = self.get_pipeline(**components).to(torch_device)

        out_features, in_features = pipe.transformer.x_embedder.weight.shape

        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": normal_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": normal_lora_B.weight,
        }
        pipe.load_lora_weights(lora_state_dict, "adapter-1")

        assert check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in denoiser"
        assert pipe.transformer.x_embedder.weight.data.shape[1] == in_features
        assert pipe.transformer.config.in_channels == in_features

        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": shape_expander_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": shape_expander_lora_B.weight,
        }

        # We should check for input shapes being incompatible here. But because above mentioned issue is
        # not a supported use case, and because of the PEFT renaming, we will currently have a shape
        # mismatch error.
        with pytest.raises(RuntimeError, match="size mismatch for x_embedder.lora_A.adapter-2.weight"):
            pipe.load_lora_weights(lora_state_dict, "adapter-2")

    def test_fuse_expanded_lora_with_regular_lora(self):
        # This test checks if it works when a lora with expanded shapes (like control loras) but
        # another lora with correct shapes is loaded. The opposite direction isn't supported and is
        # tested with it.
        components = self.get_dummy_components()

        # Change the transformer config to mimic a real use case.
        num_channels_without_control = 4
        components["transformer"] = FluxTransformer2DModel.from_config(
            components["transformer"].config, in_channels=num_channels_without_control
        ).to(torch_device)

        pipe = self.get_pipeline(**components).to(torch_device)

        out_features, in_features = pipe.transformer.x_embedder.weight.shape
        rank = 4

        shape_expander_lora_A = torch.nn.Linear(2 * in_features, rank, bias=False)
        shape_expander_lora_B = torch.nn.Linear(rank, out_features, bias=False)
        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": shape_expander_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": shape_expander_lora_B.weight,
        }
        pipe.load_lora_weights(lora_state_dict, "adapter-1")
        assert check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in denoiser"

        lora_output = self.run_pipe(pipe)

        normal_lora_A = torch.nn.Linear(in_features, rank, bias=False)
        normal_lora_B = torch.nn.Linear(rank, out_features, bias=False)
        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": normal_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": normal_lora_B.weight,
        }

        pipe.load_lora_weights(lora_state_dict, "adapter-2")
        assert check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in denoiser"

        lora_output_2 = self.run_pipe(pipe)

        pipe.set_adapters(["adapter-1", "adapter-2"], [1.0, 1.0])
        lora_output_3 = self.run_pipe(pipe)

        assert not torch.allclose(lora_output, lora_output_2, atol=1e-3, rtol=1e-3)
        assert not torch.allclose(lora_output, lora_output_3, atol=1e-3, rtol=1e-3)
        assert not torch.allclose(lora_output_2, lora_output_3, atol=1e-3, rtol=1e-3)

        pipe.fuse_lora(lora_scale=1.0, adapter_names=["adapter-1", "adapter-2"])
        lora_output_4 = self.run_pipe(pipe)
        assert_tensors_close(
            lora_output_4,
            lora_output_3,
            atol=1e-3,
            rtol=1e-3,
            msg="Fusing the adapters shouldn't change the output.",
        )

    def test_load_regular_lora(self, base_pipe_output):
        # This test checks if a regular lora (think of one trained on Flux.1 Dev for example) can be loaded
        # into the transformer with more input channels than Flux.1 Dev, for example. Some examples of those
        # transformers include Flux Fill, Flux Control, etc.
        pipe = self.get_pipeline().to(torch_device)

        out_features, in_features = pipe.transformer.x_embedder.weight.shape
        rank = 4
        in_features = in_features // 2  # to mimic the Flux.1-Dev LoRA.
        normal_lora_A = torch.nn.Linear(in_features, rank, bias=False)
        normal_lora_B = torch.nn.Linear(rank, out_features, bias=False)
        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": normal_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": normal_lora_B.weight,
        }

        logger = logging.get_logger("diffusers.loaders.lora_pipeline")
        logger.setLevel(logging.INFO)
        with CaptureLogger(logger) as cap_logger:
            pipe.load_lora_weights(lora_state_dict, "adapter-1")
        assert check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in denoiser"

        lora_output = self.run_pipe(pipe)

        assert "The following LoRA modules were zero padded to match the state dict of" in cap_logger.out
        assert pipe.transformer.x_embedder.weight.data.shape[1] == in_features * 2
        assert not torch.allclose(base_pipe_output, lora_output, atol=1e-3, rtol=1e-3)

    def test_lora_unload_with_parameter_expanded_shapes(self):
        components = self.get_dummy_components()

        logger = logging.get_logger("diffusers.loaders.lora_pipeline")
        logger.setLevel(logging.DEBUG)

        # Change the transformer config to mimic a real use case.
        num_channels_without_control = 4
        transformer = FluxTransformer2DModel.from_config(
            components["transformer"].config, in_channels=num_channels_without_control
        ).to(torch_device)
        assert transformer.config.in_channels == num_channels_without_control, (
            f"Expected {num_channels_without_control} channels in the modified transformer but has "
            f"{transformer.config.in_channels=}"
        )

        # This should be initialized with a Flux pipeline variant that doesn't accept `control_image`.
        components["transformer"] = transformer
        pipe = FluxPipeline(**components).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        original_out = self.run_pipe_without_control(pipe)

        control_pipe = self.get_pipeline(**components)
        out_features, in_features = control_pipe.transformer.x_embedder.weight.shape
        rank = 4

        dummy_lora_A = torch.nn.Linear(2 * in_features, rank, bias=False)
        dummy_lora_B = torch.nn.Linear(rank, out_features, bias=False)
        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": dummy_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": dummy_lora_B.weight,
        }
        with CaptureLogger(logger) as cap_logger:
            control_pipe.load_lora_weights(lora_state_dict, "adapter-1")
            assert check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in denoiser"

        lora_out = self.run_pipe(control_pipe)

        assert not torch.allclose(original_out, lora_out, atol=1e-4, rtol=1e-4)
        assert pipe.transformer.x_embedder.weight.data.shape[1] == 2 * in_features
        assert pipe.transformer.config.in_channels == 2 * in_features
        assert cap_logger.out.startswith("Expanding the nn.Linear input/output features for module")

        control_pipe.unload_lora_weights(reset_to_overwritten_params=True)
        assert control_pipe.transformer.config.in_channels == num_channels_without_control, (
            f"Expected {num_channels_without_control} channels in the modified transformer but has "
            f"{control_pipe.transformer.config.in_channels=}"
        )
        loaded_pipe = FluxPipeline.from_pipe(control_pipe)
        assert loaded_pipe.transformer.config.in_channels == num_channels_without_control, (
            f"Expected {num_channels_without_control} channels in the modified transformer but has "
            f"{loaded_pipe.transformer.config.in_channels=}"
        )
        unloaded_lora_out = self.run_pipe_without_control(loaded_pipe)

        assert not torch.allclose(unloaded_lora_out, lora_out, atol=1e-4, rtol=1e-4)
        assert_tensors_close(
            unloaded_lora_out,
            original_out,
            atol=1e-4,
            rtol=1e-4,
            msg="Unloading the LoRA should restore the original output.",
        )
        assert pipe.transformer.x_embedder.weight.data.shape[1] == in_features
        assert pipe.transformer.config.in_channels == in_features

    def test_lora_unload_with_parameter_expanded_shapes_and_no_reset(self):
        components = self.get_dummy_components()

        logger = logging.get_logger("diffusers.loaders.lora_pipeline")
        logger.setLevel(logging.DEBUG)

        # Change the transformer config to mimic a real use case.
        num_channels_without_control = 4
        transformer = FluxTransformer2DModel.from_config(
            components["transformer"].config, in_channels=num_channels_without_control
        ).to(torch_device)
        assert transformer.config.in_channels == num_channels_without_control, (
            f"Expected {num_channels_without_control} channels in the modified transformer but has "
            f"{transformer.config.in_channels=}"
        )

        # This should be initialized with a Flux pipeline variant that doesn't accept `control_image`.
        components["transformer"] = transformer
        pipe = FluxPipeline(**components).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        original_out = self.run_pipe_without_control(pipe)

        control_pipe = self.get_pipeline(**components)
        out_features, in_features = control_pipe.transformer.x_embedder.weight.shape
        rank = 4

        dummy_lora_A = torch.nn.Linear(2 * in_features, rank, bias=False)
        dummy_lora_B = torch.nn.Linear(rank, out_features, bias=False)
        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": dummy_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": dummy_lora_B.weight,
        }
        with CaptureLogger(logger) as cap_logger:
            control_pipe.load_lora_weights(lora_state_dict, "adapter-1")
            assert check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in denoiser"

        lora_out = self.run_pipe(control_pipe)

        assert not torch.allclose(original_out, lora_out, atol=1e-4, rtol=1e-4)
        assert pipe.transformer.x_embedder.weight.data.shape[1] == 2 * in_features
        assert pipe.transformer.config.in_channels == 2 * in_features
        assert cap_logger.out.startswith("Expanding the nn.Linear input/output features for module")

        control_pipe.unload_lora_weights(reset_to_overwritten_params=False)
        assert control_pipe.transformer.config.in_channels == 2 * num_channels_without_control, (
            f"Expected {2 * num_channels_without_control} channels in the modified transformer but has "
            f"{control_pipe.transformer.config.in_channels=}"
        )
        no_lora_out = self.run_pipe(control_pipe)

        assert not torch.allclose(no_lora_out, lora_out, atol=1e-4, rtol=1e-4)
        assert pipe.transformer.x_embedder.weight.data.shape[1] == in_features * 2
        assert pipe.transformer.config.in_channels == in_features * 2


@nightly
@require_torch_accelerator
@require_peft_backend
@require_big_accelerator
class TestFluxControlLoRAIntegration:
    num_inference_steps = 10
    seed = 0
    prompt = "A robot made of exotic candies and chocolates of different kinds."
    repo_id = "black-forest-labs/FLUX.1-dev"

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    @pytest.fixture
    def pipeline(self):
        return FluxControlPipeline.from_pretrained(self.repo_id, torch_dtype=torch.bfloat16).to(torch_device)

    def get_control_image(self, lora_ckpt_id):
        condition = "canny" if "Canny" in lora_ckpt_id else "depth"
        return load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/"
            f"flux-control-lora/{condition}_condition_image.png"
        )

    @pytest.mark.parametrize(
        "lora_ckpt_id", ["black-forest-labs/FLUX.1-Canny-dev-lora", "black-forest-labs/FLUX.1-Depth-dev-lora"]
    )
    def test_lora(self, pipeline, lora_ckpt_id):
        pipeline.load_lora_weights(lora_ckpt_id)
        pipeline.fuse_lora()
        pipeline.unload_lora_weights()

        image = pipeline(
            prompt=self.prompt,
            control_image=self.get_control_image(lora_ckpt_id),
            height=1024,
            width=1024,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=30.0 if "Canny" in lora_ckpt_id else 10.0,
            output_type="np",
            generator=torch.manual_seed(self.seed),
        ).images

        out_slice = image[0, -3:, -3:, -1].flatten()
        if "Canny" in lora_ckpt_id:
            expected_slice = np.array([0.8438, 0.8438, 0.8438, 0.8438, 0.8438, 0.8398, 0.8438, 0.8438, 0.8516])
        else:
            expected_slice = np.array([0.8203, 0.8320, 0.8359, 0.8203, 0.8281, 0.8281, 0.8203, 0.8242, 0.8359])

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), out_slice)

        assert max_diff < 1e-3

    @pytest.mark.parametrize(
        "lora_ckpt_id", ["black-forest-labs/FLUX.1-Canny-dev-lora", "black-forest-labs/FLUX.1-Depth-dev-lora"]
    )
    def test_lora_with_turbo(self, pipeline, lora_ckpt_id):
        pipeline.load_lora_weights(lora_ckpt_id)
        pipeline.load_lora_weights("ByteDance/Hyper-SD", weight_name="Hyper-FLUX.1-dev-8steps-lora.safetensors")
        pipeline.fuse_lora()
        pipeline.unload_lora_weights()

        image = pipeline(
            prompt=self.prompt,
            control_image=self.get_control_image(lora_ckpt_id),
            height=1024,
            width=1024,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=30.0 if "Canny" in lora_ckpt_id else 10.0,
            output_type="np",
            generator=torch.manual_seed(self.seed),
        ).images

        out_slice = image[0, -3:, -3:, -1].flatten()
        if "Canny" in lora_ckpt_id:
            expected_slice = np.array([0.6562, 0.7266, 0.7578, 0.6367, 0.6758, 0.7031, 0.6172, 0.6602, 0.6484])
        else:
            expected_slice = np.array([0.6680, 0.7344, 0.7656, 0.6484, 0.6875, 0.7109, 0.6328, 0.6719, 0.6562])

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), out_slice)

        assert max_diff < 1e-3
