import unittest

import numpy as np
import shutil
import torch
from pathlib import Path
from torch.onnx import export

from diffusers import StableDiffusion3Pipeline, OnnxStableDiffusion3Pipeline, OnnxRuntimeModel
from diffusers.utils.testing_utils import (
    torch_device,
    is_onnx_available
)

from ..test_pipelines_onnx_common import OnnxPipelineTesterMixin

if is_onnx_available():
    import onnxruntime as ort

def onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export(
        model,
        model_args,
        f=output_path,
        input_names=ordered_input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

def build_tiny_onnx_model():
    torch.manual_seed(0)
    pipeline = StableDiffusion3Pipeline.from_pretrained("yujiepan/stable-diffusion-3-tiny-random").to(torch_device)
    dtype = torch.float32

    # TEXT ENCODER
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size
    text_input = pipeline.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    onnx_export(
        pipeline.text_encoder,
        model_args=(
            text_input.input_ids.to(device=torch_device, dtype=torch.int32),
            None,
            None,
            None,
            True,
        ),
        output_path=Path("sd3/text_encoder/model.onnx"),
        ordered_input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output", "hidden_states"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
        },
    )
    del pipeline.text_encoder

    num_tokens = pipeline.text_encoder_2.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder_2.config.hidden_size
    text_input = pipeline.tokenizer_2(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    onnx_export(
        pipeline.text_encoder_2,
        # casting to torch.int32 until the CLIP fix is released: https://github.com/huggingface/transformers/pull/18515/files
        model_args=(
            text_input.input_ids.to(device=torch_device, dtype=torch.int32),
            None,
            None,
            None,
            True,
        ),
        output_path=Path("sd3/text_encoder_2/model.onnx"),
        ordered_input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output", "hidden_states"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
        },
    )
    del pipeline.text_encoder_2

    text_input = pipeline.tokenizer_3(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer_3.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    onnx_export(
        pipeline.text_encoder_3,
        model_args=(text_input.input_ids.to(device=torch_device, dtype=torch.int32)),
        output_path=Path("sd3/text_encoder_3/model.onnx"),
        ordered_input_names=["input_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
        },
    )
    del pipeline.text_encoder_3

    # TRANSFORMER
    in_channels = pipeline.transformer.config.in_channels
    sample_size = pipeline.transformer.config.sample_size
    joint_attention_dim = pipeline.transformer.config.joint_attention_dim
    pooled_projection_dim = pipeline.transformer.config.pooled_projection_dim
    transformer_path = Path("sd3/transformer/model.onnx")
    onnx_export(
        pipeline.transformer,
        model_args=(
            torch.randn(2, in_channels, sample_size, sample_size).to(device=torch_device, dtype=dtype),
            torch.randn(2, num_tokens, joint_attention_dim).to(device=torch_device, dtype=dtype),
            torch.randn(2, pooled_projection_dim).to(device=torch_device, dtype=dtype),
            torch.randn(2).to(device=torch_device, dtype=dtype),
        ),
        output_path=transformer_path,
        ordered_input_names=["hidden_states", "encoder_hidden_states", "pooled_projections", "timestep"],
        output_names=["out_sample"],  # has to be different from "sample" for correct tracing
        dynamic_axes={
            "hidden_states": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            "encoder_hidden_states": {0: "batch", 1: "sequence", 2: "embed_dims"},
            "pooled_projections": {0: "batch", 1: "projection_dim"},
            "timestep": {0: "batch"},
        },
    )

    # VAE ENCODER
    vae_encoder = pipeline.vae
    vae_in_channels = vae_encoder.config.in_channels
    vae_sample_size = vae_encoder.config.sample_size
    vae_encoder.forward = lambda sample, return_dict: vae_encoder.encode(sample, return_dict)[0].sample()
    onnx_export(
        vae_encoder,
        model_args=(
            torch.randn(1, vae_in_channels, vae_sample_size, vae_sample_size).to(device=torch_device, dtype=dtype),
            False,
        ),
        output_path=Path("sd3/vae_encoder/model.onnx"),
        ordered_input_names=["sample", "return_dict"],
        output_names=["latent_sample"],
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        },
    )

    # VAE DECODER
    vae_decoder = pipeline.vae
    vae_latent_channels = vae_decoder.config.latent_channels
    vae_out_channels = vae_decoder.config.out_channels
    vae_decoder.forward = vae_encoder.decode
    onnx_export(
        vae_decoder,
        model_args=(
            torch.randn(1, vae_latent_channels, sample_size, sample_size).to(device=torch_device, dtype=dtype),
            False,
        ),
        output_path=Path("sd3/vae_decoder/model.onnx"),
        ordered_input_names=["latent_sample", "return_dict"],
        output_names=["sample"],
        dynamic_axes={
            "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        },
    )
    del pipeline.vae

    onnx_pipeline = OnnxStableDiffusion3Pipeline(
        vae_encoder=OnnxRuntimeModel.from_pretrained("sd3/vae_encoder"),
        vae_decoder=OnnxRuntimeModel.from_pretrained("sd3/vae_decoder"),
        text_encoder=OnnxRuntimeModel.from_pretrained("sd3/text_encoder"),
        tokenizer=pipeline.tokenizer,
        text_encoder_2=OnnxRuntimeModel.from_pretrained("sd3/text_encoder_2"),
        tokenizer_2=pipeline.tokenizer_2,
        text_encoder_3=OnnxRuntimeModel.from_pretrained("sd3/text_encoder_3"),
        tokenizer_3=pipeline.tokenizer_3,
        transformer=OnnxRuntimeModel.from_pretrained("sd3/transformer"),
        scheduler=pipeline.scheduler,
    )

    onnx_pipeline.save_pretrained("sd3")

class OnnxStableDiffusion3PipelineFastTests(unittest.TestCase, OnnxPipelineTesterMixin):
    pipeline_class = OnnxStableDiffusion3Pipeline
    params = frozenset(
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
    batch_params = frozenset(["prompt", "negative_prompt"])
    checkpoint = "sd3"

    def get_dummy_inputs(self, device, seed=0):
        generator = np.random.RandomState(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
        }
        return inputs

    @classmethod
    def setUpClass(self):
        build_tiny_onnx_model()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("sd3", ignore_errors=True)

    def test_onnx_stable_diffusion_3_different_prompts(self):
        pipe = self.pipeline_class.from_pretrained(self.checkpoint)

        inputs = self.get_dummy_inputs(torch_device)
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt_2"] = "a different prompt"
        inputs["prompt_3"] = "another different prompt"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = np.abs(output_same_prompt - output_different_prompts).max()

        # Outputs should be different here
        assert max_diff > 1e-2

    def test_onnx_stable_diffusion_3_different_negative_prompts(self):
        pipe = self.pipeline_class.from_pretrained(self.checkpoint)

        inputs = self.get_dummy_inputs(torch_device)
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["negative_prompt_2"] = "deformed"
        inputs["negative_prompt_3"] = "blurry"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = np.abs(output_same_prompt - output_different_prompts).max()

        # Outputs should be different here
        assert max_diff > 1e-2

    def test_onnx_stable_diffusion_3_prompt_embeds(self):
        pipe = self.pipeline_class.from_pretrained(self.checkpoint)
        inputs = self.get_dummy_inputs(torch_device)

        output_with_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        prompt = inputs.pop("prompt")

        do_classifier_free_guidance = inputs["guidance_scale"] > 1
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt,
            prompt_2=None,
            prompt_3=None,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        output_with_embeds = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            **inputs,
        ).images[0]

        max_diff = np.abs(output_with_prompt - output_with_embeds).max()
        assert max_diff < 1e-4
