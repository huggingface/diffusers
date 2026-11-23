import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
)

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3PAGPipeline,
    StableDiffusion3Pipeline,
)

from ...testing_utils import torch_device
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
)
from .testing_utils import PAGPipelineTesterMixin


class StableDiffusion3PAGPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusion3PAGPipeline
    required_input_params_in_call_signature = frozenset(
        [
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (3, 32, 32)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = SD3Transformer2DModel(
            sample_size=32,
            patch_size=1,
            in_channels=4,
            num_layers=2,
            attention_head_dim=8,
            num_attention_heads=4,
            caption_projection_dim=32,
            joint_attention_dim=32,
            pooled_projection_dim=64,
            out_channels=4,
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
        text_encoder = CLIPTextModelWithProjection(clip_text_encoder_config)

        torch.manual_seed(0)
        text_encoder_2 = CLIPTextModelWithProjection(clip_text_encoder_config)

        torch.manual_seed(0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder_3 = T5EncoderModel(config)

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_3 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

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
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "text_encoder_3": text_encoder_3,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "tokenizer_3": tokenizer_3,
            "transformer": transformer,
            "vae": vae,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "pag_scale": 0.0,
        }


class TestStableDiffusion3PAGPipeline(StableDiffusion3PAGPipelineTesterConfig, PAGPipelineTesterMixin):
    base_pipeline_class = StableDiffusion3Pipeline
    # The dummy inputs disable PAG (`pag_scale=0.0`); only the "PAG off reproduces the base pipeline" leg is
    # asserted here, as it was before the migration.
    check_pag_changes_output = False

    def test_stable_diffusion_3_different_prompts(self):
        pipe = self.get_pipeline().to(torch_device)

        output_same_prompt = self.run_pipe(pipe)[0]
        output_different_prompts = self.run_pipe(
            pipe, prompt_2="a different prompt", prompt_3="another different prompt"
        )[0]

        # Outputs should be different here
        assert (output_same_prompt - output_different_prompts).abs().max() > 1e-2

    def test_stable_diffusion_3_different_negative_prompts(self):
        pipe = self.get_pipeline().to(torch_device)

        output_same_prompt = self.run_pipe(pipe)[0]
        output_different_prompts = self.run_pipe(pipe, negative_prompt_2="deformed", negative_prompt_3="blurry")[0]

        # Outputs should be different here
        assert (output_same_prompt - output_different_prompts).abs().max() > 1e-2

    def test_pag_applied_layers(self):
        pipe = self.get_pipeline()

        all_self_attn_layers = [k for k in pipe.transformer.attn_processors.keys() if "attn" in k]
        original_attn_procs = pipe.transformer.attn_processors
        pag_layers = ["blocks.0", "blocks.1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_layers)

        # blocks.0
        block_0_self_attn = ["transformer_blocks.0.attn.processor"]
        pipe.transformer.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["blocks.0"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(block_0_self_attn)

        pipe.transformer.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["blocks.0.attn"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(block_0_self_attn)

        pipe.transformer.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["blocks.(0|1)"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert (len(pipe.pag_attn_processors)) == 2

        pipe.transformer.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["blocks.0", r"blocks\.1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert len(pipe.pag_attn_processors) == 2


class TestStableDiffusion3PAGPipelineMemory(StableDiffusion3PAGPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SD3 PAG pipeline."""
