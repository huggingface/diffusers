# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import importlib.util
import json
import os
import socket
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
    DINOv3ViTConfig,
    DINOv3ViTModel,
    PreTrainedTokenizerFast,
    UMT5Config,
    UMT5EncoderModel,
)

from diffusers import (
    AutoencoderKLWan,
    ContextParallelConfig,
    EdenLiveFlowMatchEulerDiscreteScheduler,
    EdenLiveLightTAEDecoder,
    EdenLivePipeline,
    EdenLiveProxyProjection,
    EdenLiveTransformer3DModel,
    UniPCMultistepScheduler,
)
from diffusers.models.transformers import transformer_edenlive
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BasePipelineTesterConfig


@pytest.fixture(autouse=True, scope="class")
def full_precision_matmul():
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def make_tokenizer():
    words = "a character walks through snow blurry racing car on sunny road fire ocean one too many".split()
    vocabulary = {word: index for index, word in enumerate(["<pad>", "</s>", "<unk>", *words])}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>", model_max_length=512
    )


def file_digests(directory):
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.fixture(params=["5b-bidir", "14b-bidir", "5b-causal", "14b-causal"])
def original_checkpoint(tmp_path, request, monkeypatch):
    import httpx
    import requests

    from diffusers.pipelines.edenlive import loading_utils

    def unexpected_request(*args, **kwargs):
        pytest.fail("A local original checkpoint must not request any external Hub configuration.")

    monkeypatch.setattr(httpx.Client, "send", unexpected_request)
    monkeypatch.setattr(requests.Session, "send", unexpected_request)

    def tiny_decoder(**_kwargs):
        # Production LightTAE dimensions stay fixed by the release profile. Only
        # its construction is reduced for this weight-loading regression fixture.
        return EdenLiveLightTAEDecoder(latent_channels=4, patch_size=1, block_out_channels=(4, 4, 4, 4))

    monkeypatch.setattr(loading_utils, "EdenLiveLightTAEDecoder", tiny_decoder)
    profile = request.param
    size, mode = profile.split("-")
    fixture = EdenLivePipelineTesterConfig()
    fixture.profile = size
    components = fixture.get_dummy_components()
    source = tmp_path / "original"
    source.mkdir()
    (source / "model_index.json").write_text(
        json.dumps({"_class_name": "EdenLivePipeline", "_edenlive_original": {"_path_overrides": {}}})
    )
    transformer_dir = source / (f"dit-{size}-bidir" if mode == "bidir" else f"dit-{size}")
    transformer_dir.mkdir()
    components["transformer"].save_config(transformer_dir)
    save_file(components["transformer"].state_dict(), transformer_dir / "diffusion_pytorch_model.safetensors")
    vae_dir = source / f"vae-{size}"
    vae_dir.mkdir()
    components["vae"].save_config(vae_dir)
    # Include original Wan parameter names and the original .pth container.
    vae_state = {}
    replacements = {
        "encoder.conv_in.": "encoder.conv1.",
        "decoder.conv_out.": "decoder.head.2.",
        "quant_conv.": "conv1.",
        "post_quant_conv.": "conv2.",
    }
    for key, value in components["vae"].state_dict().items():
        for prefix, original_prefix in replacements.items():
            if key.startswith(prefix):
                key = original_prefix + key[len(prefix) :]
                break
        vae_state[key] = value
    torch.save(vae_state, vae_dir / ("Wan2.2_VAE.pth" if size == "5b" else "Wan2.1_VAE.pth"))
    components["text_encoder"].save_pretrained(source / "umt5/text_encoder")
    components["tokenizer"].save_pretrained(source / "umt5/tokenizer")
    # This fast fixture consumes tokenizer.json. Mirror the additional vocabulary
    # and special-token files listed by the original release's download plan.
    (source / "umt5/tokenizer/spiece.model").write_bytes(b"unused-fast-tokenizer-fixture")
    (source / "umt5/tokenizer/special_tokens_map.json").write_text(
        json.dumps({"pad_token": "<pad>", "eos_token": "</s>", "unk_token": "<unk>"})
    )
    proxy_dir = source / "vision-encoder"
    components["proxy_encoder"].config.save_pretrained(proxy_dir / "backbone")
    proxy_state = {
        f"module.encoder.backbone.{key}": value for key, value in components["proxy_encoder"].state_dict().items()
    }
    proxy_state.update(
        {f"module.encoder.{key}": value for key, value in components["proxy_projection"].state_dict().items()}
    )
    proxy_state.update(
        {
            "module.encoder.global_projection.weight": torch.zeros(2, 32),
            "module.encoder.global_projection.bias": torch.zeros(2),
        }
    )
    save_file(proxy_state, proxy_dir / "projection.safetensors")
    if size == "14b":
        components["image_encoder"].save_pretrained(source / "clip/image_encoder")
        components["image_processor"].save_pretrained(source / "clip/image_processor")
    decoder = EdenLiveLightTAEDecoder(latent_channels=4, patch_size=1, block_out_channels=(4, 4, 4, 4))
    decoder_dir = source / f"decoder-{size}"
    decoder_dir.mkdir()
    decoder_state = {
        key.replace("decoder.", "decoder.blocks.", 1): value for key, value in decoder.state_dict().items()
    }
    temporal_key = "decoder.blocks.7.conv.weight"
    decoder_state[temporal_key] = torch.cat(
        [torch.zeros_like(decoder_state[temporal_key]), decoder_state[temporal_key]]
    )
    decoder_state["encoder.unused.weight"] = torch.ones(1)
    torch.save(decoder_state, decoder_dir / ("lighttaew2_2.pth" if size == "5b" else "lighttaew2_1.pth"))
    return SimpleNamespace(
        source=source,
        profile=profile,
        pipeline_class=EdenLivePipeline,
        components=components,
        decoder=decoder,
        hashes=file_digests(source),
        transformer_dir=transformer_dir,
    )


@pytest.fixture
def original_hub(original_checkpoint, monkeypatch):
    from diffusers.pipelines.edenlive import loading_utils as pipeline_utils

    fixture = original_checkpoint
    calls = SimpleNamespace(metadata=[], snapshots=[])
    revision = "0123456789abcdef0123456789abcdef01234567"
    repo_id = "test-org/edenlive-original-fixture"

    def metadata_download(repository, filename, **kwargs):
        assert repository == repo_id and filename == "model_index.json"
        assert kwargs["revision"] == revision
        calls.metadata.append(kwargs)
        return str(fixture.source / filename)

    def snapshot_download(repository, **kwargs):
        assert repository == repo_id
        calls.snapshots.append(kwargs)
        return str(fixture.source)

    monkeypatch.setattr(pipeline_utils, "_resolve_revision", lambda *args, **kwargs: revision)
    monkeypatch.setattr(pipeline_utils, "hf_hub_download", metadata_download)
    monkeypatch.setattr(pipeline_utils, "snapshot_download", snapshot_download)
    return SimpleNamespace(fixture=fixture, calls=calls, revision=revision, repo_id=repo_id, module=pipeline_utils)


class EdenLivePipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = EdenLivePipeline
    profile = "5b"
    causal = False
    required_input_params_in_call_signature = frozenset(
        ["prompt", "negative_prompt", "image", "video", "height", "width", "prompt_embeds", "negative_prompt_embeds"]
    )
    optional_input_params = frozenset(
        ["num_frames", "num_inference_steps", "guidance_scale", "generator", "latents", "output_type", "return_dict"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (9, 3, 16, 16)

    def get_dummy_components(self):
        torch.manual_seed(0)
        vae = AutoencoderKLWan(
            base_dim=4,
            z_dim=4,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
        )
        torch.manual_seed(0)
        text_encoder = UMT5EncoderModel(
            UMT5Config(vocab_size=1103, d_model=32, d_ff=32, d_kv=8, num_layers=1, num_heads=4, dropout_rate=0.0)
        ).eval()
        tokenizer = make_tokenizer()
        torch.manual_seed(0)
        proxy_encoder = DINOv3ViTModel(
            DINOv3ViTConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                patch_size=16,
                image_size=32,
                num_register_tokens=4,
            )
        ).eval()
        torch.manual_seed(0)
        proxy_projection = EdenLiveProxyProjection(in_channels=32, out_channels=8)
        torch.manual_seed(0)
        transformer = EdenLiveTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=4,
            attention_head_dim=12,
            in_channels=4 if self.profile == "5b" else 12,
            out_channels=4,
            proxy_channels=8,
            condition_rmsnorm=self.profile == "14b",
            text_dim=32,
            freq_dim=16,
            ffn_dim=64,
            num_layers=1,
            image_dim=32 if self.profile == "14b" else None,
            added_kv_proj_dim=48 if self.profile == "14b" else None,
            rope_max_seq_len=64,
        )
        if self.profile == "14b":
            torch.manual_seed(0)
            image_encoder = CLIPVisionModel(
                CLIPVisionConfig(
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    image_size=16,
                    patch_size=8,
                    projection_dim=32,
                )
            ).eval()
            image_processor = CLIPImageProcessor(size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16})
            scheduler = EdenLiveFlowMatchEulerDiscreteScheduler(shift=3)
        else:
            image_encoder, image_processor = None, None
            scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5)
        return {
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "transformer": transformer,
            "scheduler": scheduler,
            "proxy_encoder": proxy_encoder,
            "proxy_projection": proxy_projection,
            "image_encoder": image_encoder,
            "image_processor": image_processor,
        }

    def get_pipeline(self, **components):
        causal = self.causal
        pipe = self.pipeline_class(
            **(components or self.get_dummy_components()),
            profile=f"{self.profile}-{'causal' if causal else 'bidir'}",
            proxy_image_size=(32, 32),
            proxy_resize_mode=("bicubic_center_crop" if causal else "bicubic")
            if self.profile == "14b"
            else "bilinear",
            proxy_unconditional="zero" if self.profile == "14b" and causal else "same",
        )
        for component in pipe.components.values():
            if hasattr(component, "eval"):
                component.eval()
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def test_inference_and_standard_save_load(self, tmp_path):
        pipe = self.get_pipeline().to(torch_device)
        expected = pipe(**self.get_dummy_inputs()).frames
        assert expected.shape == (1, *self.output_shape)
        assert torch.isfinite(expected).all()
        pipe.save_pretrained(tmp_path)
        for loader in (self.pipeline_class,):
            restored = loader.from_pretrained(tmp_path, local_files_only=True).to(torch_device)
            restored.set_progress_bar_config(disable=True)
            actual = restored(**self.get_dummy_inputs()).frames
            assert type(restored) is self.pipeline_class
            assert restored.config.profile == pipe.config.profile
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def get_dummy_inputs(self):
        generator = np.random.default_rng(0)
        return {
            "prompt": "a character walks through snow",
            "negative_prompt": "blurry",
            "image": Image.fromarray(generator.integers(0, 256, (16, 16, 3), dtype=np.uint8)),
            "video": generator.random((9, 16, 16, 3), dtype=np.float32),
            "height": 16,
            "width": 16,
            "num_frames": 9,
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "generator": self.get_generator(0),
            "max_sequence_length": 16,
            "output_type": "pt",
        }


class TestEdenLivePipeline(EdenLivePipelineTesterConfig):
    @pytest.mark.parametrize("num_frames", [0, 4, 8, 13])
    def test_invalid_frame_count(self, num_frames):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = num_frames
        with pytest.raises(ValueError, match="frames|frame"):
            pipe(**inputs)

    def test_automatic_frame_prefix(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = None
        inputs["video"] = np.concatenate([inputs["video"], inputs["video"][:2]])
        output = pipe(**inputs).frames
        assert output.shape == (1, *self.output_shape)

    @pytest.mark.parametrize("output_type", ["np", "pt", "pil", "latent"])
    def test_output_types(self, output_type):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["output_type"] = output_type
        output = pipe(**inputs).frames
        if output_type == "np":
            assert output.shape == (1, 9, 16, 16, 3)
        elif output_type == "pt":
            assert output.shape == (1, *self.output_shape)
        elif output_type == "latent":
            assert output.shape == (1, 4, 3, 2, 2)
        else:
            assert len(output) == 1 and len(output[0]) == 9
            assert isinstance(output[0][0], Image.Image)

    @pytest.mark.parametrize("condition", ["image", "video", "prompt"])
    def test_conditions_change_generation(self, condition):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        expected = pipe(**inputs).frames
        inputs = self.get_dummy_inputs()
        if condition == "prompt":
            inputs[condition] = "a racing car on a sunny road"
        elif condition == "image":
            inputs[condition] = Image.new("RGB", (16, 16), "white")
        else:
            inputs[condition] = 1 - inputs[condition]
        actual = pipe(**inputs).frames
        assert (actual - expected).abs().max() > 1e-6

    def test_timeline_boundary(self):
        from diffusers.pipelines.edenlive.pipeline_utils import encode_timeline

        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["text_timeline"] = {
            "num_frames": 9,
            "fps": 4,
            "segments": [
                {"start_sec": 0, "end_sec": 1, "text": "snow"},
                {"start_sec": 1, "end_sec": 2, "text": "fire"},
            ],
        }
        output = pipe(**inputs).frames
        assert output.shape == (1, *self.output_shape)
        _, _, layout = encode_timeline(
            pipe, inputs["text_timeline"], inputs["negative_prompt"], 9, (1, 4, 3, 2, 2), 16, True
        )
        assert layout["lengths"] == ((2, 1),)


class EdenLive14BRequiredComponentsTesterMixin:
    def test_save_load_optional_components(self, tmp_path):
        pipe = self.get_pipeline()
        pipe.register_modules(image_encoder=None, image_processor=None)
        pipe.save_pretrained(tmp_path)
        restored = self.pipeline_class.from_pretrained(tmp_path).to(torch_device)
        assert restored.image_encoder is None and restored.image_processor is None
        with pytest.raises(ValueError, match="14B profile requires"):
            restored(**self.get_dummy_inputs())


class TestEdenLive14BPipeline(EdenLivePipelineTesterConfig, EdenLive14BRequiredComponentsTesterMixin):
    profile = "14b"

    def test_timeline_is_rejected(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["text_timeline"] = {}
        with pytest.raises(ValueError, match="5B"):
            pipe(**inputs)


class EdenLiveChunkCausalPipelineTesterConfig(EdenLivePipelineTesterConfig):
    causal = True

    def get_dummy_components(self):
        components = super().get_dummy_components()
        components["scheduler"] = EdenLiveFlowMatchEulerDiscreteScheduler(
            stochastic_sampling=True, default_timesteps=[1000, 500]
        )
        return components


class TestEdenLiveChunkCausalPipeline(EdenLiveChunkCausalPipelineTesterConfig):
    def test_short_tail_and_no_state_leak(self):
        pipe = self.get_pipeline().to(torch_device)

        def inputs():
            result = self.get_dummy_inputs()
            result["video"] = np.concatenate([result["video"], result["video"][:4]])
            result["num_frames"] = 13
            result["chunk_prompts"] = ["snow", "fire"]
            return result

        expected = pipe(**inputs()).frames
        actual = pipe(**inputs()).frames
        assert actual.shape == (1, 13, 3, 16, 16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not any("cache" in name for name in pipe.transformer.state_dict())

    def test_chunk_prompt_changes_tail(self):
        pipe = self.get_pipeline().to(torch_device)

        def inputs(tail):
            result = self.get_dummy_inputs()
            result["video"] = np.concatenate([result["video"], result["video"][:4]])
            result.update(num_frames=13, chunk_prompts=["snow", tail], output_type="latent")
            return result

        first = pipe(**inputs("fire")).frames
        second = pipe(**inputs("ocean")).frames
        torch.testing.assert_close(first[:, :, :3], second[:, :, :3], atol=0, rtol=0)
        assert (first[:, :, 3:] - second[:, :, 3:]).abs().max() > 1e-6

    def test_invalid_chunk_prompts(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["chunk_prompts"] = ["one", "too many"]
        with pytest.raises(ValueError, match="chunk_prompts"):
            pipe(**inputs)

    def test_chunk_prompts_reject_batched_prompt(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs.update(prompt=["snow", "fire"], chunk_prompts=["ocean"])
        with pytest.raises(ValueError, match="batch size one"):
            pipe(**inputs)


class TestEdenLive14BChunkCausalPipeline(
    EdenLiveChunkCausalPipelineTesterConfig, EdenLive14BRequiredComponentsTesterMixin
):
    profile = "14b"

    def test_zero_proxy_changes_unconditional_prediction(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["output_type"] = "latent"
        zero_proxy = pipe(**inputs).frames
        pipe.register_to_config(proxy_unconditional="same")
        inputs = self.get_dummy_inputs()
        inputs["output_type"] = "latent"
        same_proxy = pipe(**inputs).frames
        assert (zero_proxy - same_proxy).abs().max() > 1e-6


class TestEdenLiveUnifiedInterface:
    @pytest.mark.parametrize(
        "profile,steps,guidance",
        [("5b-bidir", 48, 3), ("14b-bidir", 30, 3), ("5b-causal", 4, 1), ("14b-causal", 4, 1)],
    )
    def test_profile_defaults(self, profile, steps, guidance):
        fixture = (
            EdenLiveChunkCausalPipelineTesterConfig() if profile.endswith("causal") else EdenLivePipelineTesterConfig()
        )
        fixture.profile = profile.split("-")[0]
        components = fixture.get_dummy_components()
        if fixture.causal:
            components["scheduler"].register_to_config(default_timesteps=[1000, 750, 500, 250])
        pipe = fixture.get_pipeline(**components).to(torch_device)
        inputs = fixture.get_dummy_inputs()
        inputs.pop("num_inference_steps")
        inputs.pop("guidance_scale")
        inputs.update(num_frames=1, output_type="latent")
        indices = []

        def callback(pipeline, index, timestep, tensors):
            indices.append(index)
            return {}

        output = pipe(**inputs, callback_on_step_end=callback).frames
        assert type(pipe) is EdenLivePipeline and pipe.config.profile == profile
        assert pipe.guidance_scale == guidance
        assert pipe.num_timesteps == steps
        assert indices == list(range(steps))
        assert output.shape == (1, 4, 1, 2, 2)

    @pytest.mark.parametrize(
        "profile,argument",
        [
            ("5b-bidir", "chunk_prompts"),
            ("14b-bidir", "chunk_prompts"),
            ("14b-bidir", "text_timeline"),
            ("5b-causal", "text_timeline"),
            ("14b-causal", "text_timeline"),
        ],
    )
    def test_incompatible_temporal_prompts_fail_before_inference(self, profile, argument):
        fixture = EdenLivePipelineTesterConfig()
        fixture.profile, mode = profile.split("-")
        fixture.causal = mode == "causal"
        pipe = fixture.get_pipeline()
        with pytest.raises(ValueError, match=argument):
            pipe(**{argument: []})

    @pytest.mark.parametrize("size", ["5b", "14b"])
    def test_causal_callback_exception_cleans_history(self, size):
        fixture = EdenLiveChunkCausalPipelineTesterConfig()
        fixture.profile = size
        pipe = fixture.get_pipeline().to(torch_device)
        inputs = fixture.get_dummy_inputs()
        inputs.update(num_frames=13, output_type="latent")
        inputs["video"] = np.concatenate([inputs["video"], inputs["video"][:4]])
        states = []
        steps = []

        def record_state(module, args, kwargs):
            state = kwargs["causal_state"]
            if all(state is not item for item in states):
                states.append(state)

        def fail_on_tail(pipeline, index, timestep, tensors):
            steps.append((index, tensors["latents"].shape[2]))
            if index == 2:
                raise RuntimeError("callback failed on tail")
            return {}

        handle = pipe.transformer.register_forward_pre_hook(record_state, with_kwargs=True)
        try:
            with pytest.raises(RuntimeError, match="callback failed on tail"):
                pipe(**inputs, callback_on_step_end=fail_on_tail)
        finally:
            handle.remove()
        assert steps == [(0, 3), (1, 3), (2, 1)]
        assert len(states) == 2 and all(state.next_frame == 0 for state in states)
        assert not any("cache" in key for key in pipe.transformer.state_dict())

    @pytest.mark.parametrize("profile", ["5b-bidir", "14b-bidir", "5b-causal", "14b-causal"])
    def test_standard_profile_selector_conflict(self, tmp_path, profile):
        fixture = EdenLivePipelineTesterConfig()
        fixture.profile, mode = profile.split("-")
        fixture.causal = mode == "causal"
        pipe = fixture.get_pipeline()
        pipe.save_pretrained(tmp_path)
        restored = EdenLivePipeline.from_pretrained(tmp_path, profile=profile, local_files_only=True)
        assert restored.config.profile == profile
        wrong_profile = f"{fixture.profile}-{'bidir' if fixture.causal else 'causal'}"
        with pytest.raises(ValueError, match="conflicts with saved profile"):
            EdenLivePipeline.from_pretrained(tmp_path, profile=wrong_profile, local_files_only=True)


class RecordingDecoder(torch.nn.Module):
    scale_factor_temporal = 4
    scale_factor_spatial = 8

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.config = SimpleNamespace(latent_channels=4)
        self.inputs = []

    def decode(self, latents, return_dict=False):
        self.inputs.append(latents.detach().clone())
        batch, _, frames, height, width = latents.shape
        return (latents.new_zeros((batch, 3, (frames - 1) * 4 + 1, height * 8, width * 8)),)


class TestEdenLiveNativeNumerics:
    @pytest.mark.parametrize("profile", ["5b", "14b"])
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dino_input_uses_fp32_profile_preprocessing(self, monkeypatch, profile, input_dtype):
        from diffusers.pipelines.edenlive.pipeline_utils import prepare_conditions

        fixture = EdenLivePipelineTesterConfig()
        fixture.profile = profile
        pipe = fixture.get_pipeline().to(torch_device)
        # A bilinear override must not give 14B the 5B normalization round trip.
        pipe.register_to_config(proxy_resize_mode="bilinear")
        rgb = torch.arange(256).reshape(1, 1, 16, 16).expand(1, 3, -1, -1)
        frame = (rgb.float() / 255).to(device=torch_device, dtype=input_dtype)
        seen = []
        preprocess = pipe.proxy_video_processor.preprocess_proxy_frames

        def capture(value, *args):
            seen.append(value.clone())
            return preprocess(value, *args)

        monkeypatch.setattr(pipe.proxy_video_processor, "preprocess_proxy_frames", capture)
        prepare_conditions(
            pipe,
            fixture.get_dummy_inputs()["image"],
            frame.unsqueeze(2),
            16,
            16,
            1,
            1,
            torch.device(torch_device),
            torch.float32,
        )
        assert seen[0].dtype == torch.float32
        if profile == "5b":
            # Fixed IEEE-754 values for the 1/255 input after the FP32 round trip.
            expected = {
                torch.float32: 0.003921568393707275,
                torch.float16: 0.0039215087890625,
                torch.bfloat16: 0.003936767578125,
            }[input_dtype]
            assert seen[0][0, 0, 0, 1].item() == expected
        else:
            torch.testing.assert_close(seen[0], frame.float(), rtol=0, atol=0)

    @pytest.mark.parametrize("guidance", [1.0, 3.0])
    def test_chunk_prompts_encode_unique_texts_individually(self, guidance):
        fixture = EdenLiveChunkCausalPipelineTesterConfig()
        pipe = fixture.get_pipeline().to(torch_device)
        encoded, used = [], []

        def record_text(module, args, kwargs):
            encoded.append((args[0].clone(), kwargs["attention_mask"].clone()))

        def record_dit(module, args, kwargs):
            if kwargs.get("cache_update"):
                used.append(args[2].clone())

        pipe.text_encoder.register_forward_pre_hook(record_text, with_kwargs=True)
        pipe.transformer.register_forward_pre_hook(record_dit, with_kwargs=True)
        inputs = fixture.get_dummy_inputs()
        inputs.update(
            video=np.tile(inputs["video"], (4, 1, 1, 1))[:29],
            num_frames=29,
            chunk_prompts=["snow", "fire", "snow"],
            guidance_scale=guidance,
            output_type="latent",
        )
        pipe(**inputs)
        expected_texts = ["snow", "fire"] + ([inputs["negative_prompt"]] if guidance > 1 else [])
        assert len(encoded) == len(expected_texts)
        for (ids, mask), text in zip(encoded, expected_texts):
            tokens = pipe.tokenizer(
                [text],
                padding="max_length",
                max_length=inputs["max_sequence_length"],
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            torch.testing.assert_close(ids.cpu(), tokens.input_ids)
            torch.testing.assert_close(mask.cpu(), tokens.attention_mask)
        conditional = used[::2] if guidance > 1 else used
        assert len(conditional) == 3
        torch.testing.assert_close(conditional[0], conditional[2], rtol=0, atol=0)
        assert not torch.equal(conditional[0], conditional[1])

    @pytest.mark.parametrize("profile", ["5b", "14b"])
    def test_clean_commit_timestep_shape_and_cfg(self, profile):
        fixture = EdenLiveChunkCausalPipelineTesterConfig()
        fixture.profile = profile
        pipe = fixture.get_pipeline().to(torch_device)
        calls = []

        def capture(module, args, kwargs):
            calls.append((args[1].clone(), kwargs.get("cache_update", False)))

        pipe.transformer.register_forward_pre_hook(capture, with_kwargs=True)
        inputs = fixture.get_dummy_inputs()
        inputs.update(
            video=np.concatenate([inputs["video"], inputs["video"][:4]]),
            num_frames=13,
            output_type="latent",
            guidance_scale=3.0,
        )
        pipe(**inputs)
        commits = [time for time, commit in calls if commit]
        assert len(commits) == 4
        assert all(time.dtype == torch.float32 and not time.any() for time in commits)
        assert commits[0].shape == ((1,) if profile == "14b" else (1, 3))
        noisy = calls[0][0]
        assert noisy.shape == (1, 3) and noisy[0, 0] == 0 and (noisy[0, 1:] > 0).all()

    @pytest.mark.parametrize("decoder_kind,expected", [("tiny", 4.0625), ("custom", 4.03125), ("vae", 4.03125)])
    def test_bf16_decoder_denormalization(self, monkeypatch, decoder_kind, expected):
        from diffusers.pipelines.edenlive.pipeline_utils import decode_latents

        fixture = EdenLivePipelineTesterConfig()
        decoder = (
            EdenLiveLightTAEDecoder(latent_channels=4, block_out_channels=(4, 4, 4, 4))
            if decoder_kind == "tiny"
            else RecordingDecoder()
            if decoder_kind == "custom"
            else None
        )
        pipe = fixture.get_pipeline(**fixture.get_dummy_components(), decoder=decoder)
        pipe.vae.to(dtype=torch.bfloat16)
        pipe.vae.register_to_config(latents_mean=[0.0] * 4, latents_std=[3.0] * 4)
        seen = []

        def decode(value, **kwargs):
            seen.append(value.clone())
            return (value.new_zeros(1, 3, 1, 16, 16),)

        monkeypatch.setattr(decoder if decoder is not None else pipe.vae, "decode", decode)
        z = torch.full((1, 4, 1, 2, 2), 1.3515625, dtype=torch.bfloat16)
        assert decode_latents(pipe, z, "latent") is z and not seen
        decode_latents(pipe, z, "pt")
        assert len(seen) == 1 and (seen[0] == expected).all()

    def test_example_fsdp_casts_only_time_and_preserves_cache_identity(self, monkeypatch):
        from torch.distributed import device_mesh, fsdp

        path = Path(__file__).resolve().parents[3] / "examples/edenlive/inference_edenlive.py"
        spec = importlib.util.spec_from_file_location("edenlive_example_test", path)
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        monkeypatch.setattr(device_mesh, "init_device_mesh", lambda *args, **kwargs: {"ulysses": object()})
        policies = []
        monkeypatch.setattr(fsdp, "fully_shard", lambda module, **kwargs: policies.append(kwargs["mp_policy"]))
        monkeypatch.setattr(example, "ContextParallelConfig", lambda **kwargs: kwargs)

        class Recorder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Identity()])

            def enable_parallelism(self, **kwargs):
                pass

            def forward(self, hidden_states, timestep, *, causal_state):
                causal_state.next_frame += 3
                return hidden_states, timestep, causal_state

        model = Recorder()
        example.configure_parallelism(model, torch.device("cpu"), 2, True)
        assert len(policies) == 2 and all(not p.cast_forward_inputs for p in policies)
        state = SimpleNamespace(next_frame=0, layers=[object()])
        hidden = torch.ones(1)
        for positional, time in [(True, torch.tensor([999])), (False, torch.tensor([[0.0, 375.0]]))]:
            args = (hidden, time) if positional else (hidden,)
            kwargs = {} if positional else {"timestep": time}
            actual_hidden, actual_time, actual_state = model(*args, **kwargs, causal_state=state)
            assert actual_hidden is hidden and actual_state is state
            assert actual_time.dtype == torch.bfloat16
            assert actual_time.tolist() == ([1000.0] if positional else [[0.0, 376.0]])
        assert state.next_frame == 6


class TestEdenLiveDecoder:
    @pytest.mark.parametrize("patch_size", [1, 2])
    @pytest.mark.parametrize("frames", [1, 3, 4, 5])
    @torch.no_grad()
    def test_tiny_frames_causality_and_call_isolation(self, patch_size, frames):
        torch.manual_seed(0)
        decoder = EdenLiveLightTAEDecoder(
            latent_channels=4, patch_size=patch_size, block_out_channels=(4, 4, 4, 4)
        ).to(torch_device)
        latents = randn_tensor((1, 4, frames, 2, 2), generator=torch.Generator().manual_seed(9), device=torch_device)
        first = decoder.decode(latents).sample
        # A separate, different decode must not contaminate the next call's temporal memory.
        decoder.decode(latents + 1)
        repeated = decoder.decode(latents, return_dict=False)[0]
        torch.testing.assert_close(first, repeated, atol=0, rtol=0)
        assert first.shape == (1, 3, (frames - 1) * 4 + 1, 16 * patch_size, 16 * patch_size)
        assert torch.isfinite(first).all() and first.min() >= -1 and first.max() <= 1
        if frames > 1:
            changed = latents.clone()
            changed[:, :, -1] += 1
            prefix = decoder.decode(changed).sample[:, :, :-4]
            torch.testing.assert_close(prefix, first[:, :, :-4], atol=0, rtol=0)
        assert not any("cache" in key for key in decoder.state_dict())

    @pytest.mark.parametrize("causal", [False, True])
    def test_custom_decoder_receives_raw_latents_once_and_latent_output_bypasses_it(self, causal):
        from diffusers.pipelines.edenlive.pipeline_utils import decode_latents

        fixture = EdenLivePipelineTesterConfig()
        fixture.causal = causal
        decoder = RecordingDecoder()
        pipe = fixture.get_pipeline(**fixture.get_dummy_components(), decoder=decoder)
        pipe.vae.register_to_config(latents_mean=[0.25] * 4, latents_std=[2.0] * 4)
        latents = torch.full((1, 4, 4, 2, 2), 0.5)
        assert decode_latents(pipe, latents, "latent") is latents
        assert decoder.inputs == []
        output = decode_latents(pipe, latents, "pt")
        assert pipe.decoder is decoder and decoder.weight.dtype == torch.float64
        assert decoder.inputs[0].dtype == torch.float64
        torch.testing.assert_close(decoder.inputs[0], torch.full_like(decoder.inputs[0], 1.25))
        assert output.shape == (1, 13, 3, 16, 16)

    @pytest.mark.parametrize("causal", [False, True])
    @pytest.mark.parametrize("saved_decoder", ["tiny", "none", "legacy"])
    @pytest.mark.parametrize("size", ["5b", "14b"])
    @pytest.mark.parametrize("legacy_class", [False, True])
    def test_standard_snapshot_preserves_decoder_choice(self, tmp_path, causal, saved_decoder, size, legacy_class):
        fixture = EdenLivePipelineTesterConfig()
        fixture.profile = size
        fixture.causal = causal
        decoder = (
            EdenLiveLightTAEDecoder(latent_channels=4, block_out_channels=(4, 4, 4, 4))
            if saved_decoder == "tiny"
            else None
        )
        pipe = fixture.get_pipeline(**fixture.get_dummy_components(), decoder=decoder)
        pipe.save_pretrained(tmp_path)
        index_path = tmp_path / "model_index.json"
        index = json.loads(index_path.read_text())
        assert "_edenlive_original" not in index
        if legacy_class:
            index["_class_name"] = "EdenLiveChunkCausalPipeline" if causal else "EdenLivePipeline"
            index["profile"] = size
        if saved_decoder == "legacy":
            index.pop("decoder", None)
            index["_class_name"] = "EdenLiveChunkCausalPipeline" if causal else "EdenLivePipeline"
            index["profile"] = size if legacy_class else pipe.config.profile
        index_path.write_text(json.dumps(index))
        hashes = file_digests(tmp_path)
        restored = EdenLivePipeline.from_pretrained(tmp_path, local_files_only=True)
        assert type(restored) is EdenLivePipeline
        assert restored.config.profile == f"{size}-{'causal' if causal else 'bidir'}"
        assert (restored.config.chunk_size, restored.config.max_context_frames, restored.config.sink_frames) == (
            3,
            18,
            6,
        )
        assert file_digests(tmp_path) == hashes
        if legacy_class:
            restored.save_pretrained(tmp_path / "new-snapshot")
            updated = json.loads((tmp_path / "new-snapshot/model_index.json").read_text())
            assert updated["_class_name"] == "EdenLivePipeline"
            assert updated["profile"] == restored.config.profile
        if saved_decoder == "tiny":
            assert isinstance(restored.decoder, EdenLiveLightTAEDecoder)
            assert restored.decoder.config.latent_channels == 4
            assert tuple(restored.decoder.config.block_out_channels) == (4, 4, 4, 4)
            for key, value in decoder.state_dict().items():
                torch.testing.assert_close(restored.decoder.state_dict()[key], value, atol=0, rtol=0)
        else:
            assert restored.decoder is None

    @pytest.mark.parametrize("causal", [False, True])
    @pytest.mark.parametrize("saved_tiny", [False, True])
    def test_standard_snapshot_accepts_plain_decoder_override(self, tmp_path, causal, saved_tiny):
        fixture = EdenLivePipelineTesterConfig()
        fixture.causal = causal
        decoder = EdenLiveLightTAEDecoder(latent_channels=4, block_out_channels=(4, 4, 4, 4)) if saved_tiny else None
        pipe = fixture.get_pipeline(**fixture.get_dummy_components(), decoder=decoder)
        pipe.save_pretrained(tmp_path)
        if saved_tiny:
            for path in (tmp_path / "decoder").glob("*.safetensors"):
                path.unlink()
        override = RecordingDecoder()
        restored = EdenLivePipeline.from_pretrained(
            tmp_path, decoder=override, dtype=torch.bfloat16, local_files_only=True
        )
        assert restored.decoder is override and override.weight.dtype == torch.float64


class TestEdenLiveOriginalLoading:
    @pytest.mark.parametrize("original_checkpoint", ["5b-bidir"], indirect=True)
    def test_inactive_loading_options_are_accepted(self, original_checkpoint):
        pipe = EdenLivePipeline.from_pretrained(
            original_checkpoint.source,
            profile=original_checkpoint.profile,
            local_files_only=True,
            device_map=None,
            variant=None,
            use_onnx=False,
            offload_state_dict=False,
        )
        assert pipe.config.profile == original_checkpoint.profile

    def test_original_profiles_components_and_default_dtype(self, original_checkpoint):
        fixture = original_checkpoint
        loader = EdenLivePipeline
        pipe = loader.from_pretrained(fixture.source, profile=fixture.profile, local_files_only=True)
        assert type(pipe) is fixture.pipeline_class
        assert pipe.config.profile == fixture.profile
        assert isinstance(pipe.decoder, EdenLiveLightTAEDecoder)
        expected_components = {**fixture.components, "decoder": fixture.decoder}
        for name, expected in expected_components.items():
            if not isinstance(expected, torch.nn.Module):
                continue
            actual = getattr(pipe, name)
            dtype = torch.float32 if name in ("proxy_encoder", "proxy_projection") else torch.bfloat16
            assert actual.dtype == dtype
            assert set(actual.state_dict()) == set(expected.state_dict())
            for key, tensor in expected.state_dict().items():
                restored = actual.state_dict()[key]
                torch.testing.assert_close(restored, tensor.to(restored.dtype), atol=0, rtol=0)
        if fixture.profile == "5b-bidir":
            assert isinstance(pipe.scheduler, UniPCMultistepScheduler)
            assert pipe.scheduler.config.flow_shift == 5
        elif fixture.profile.endswith("causal"):
            points = [1000, 937.5, 833.3333333333, 625] if fixture.profile.startswith("5b") else [1000, 900, 750, 500]
            assert pipe.scheduler.config.stochastic_sampling
            assert list(pipe.scheduler.config.default_timesteps) == points
            assert (pipe.config.chunk_size, pipe.config.max_context_frames, pipe.config.sink_frames) == (3, 18, 6)
        else:
            assert pipe.scheduler.config.shift == 3 and not pipe.scheduler.config.stochastic_sampling
        assert pipe.config.proxy_unconditional == ("zero" if fixture.profile == "14b-causal" else "same")
        expected_resize = (
            "bilinear"
            if fixture.profile.startswith("5b")
            else ("bicubic_center_crop" if fixture.profile.endswith("causal") else "bicubic")
        )
        assert pipe.config.proxy_resize_mode == expected_resize
        assert tuple(pipe.config.proxy_image_size) == (480, 832)
        assert file_digests(fixture.source) == fixture.hashes

    def test_original_save_load_produces_self_contained_standard_snapshot(self, original_checkpoint, tmp_path):
        fixture = original_checkpoint
        pipe = EdenLivePipeline.from_pretrained(
            fixture.source, profile=fixture.profile, dtype=torch.float32, local_files_only=True
        )
        snapshot = tmp_path / "standard"
        pipe.save_pretrained(snapshot)
        index = json.loads((snapshot / "model_index.json").read_text())
        assert "_edenlive_original" not in index
        assert index["decoder"] == ["diffusers", "EdenLiveLightTAEDecoder"]
        restored = EdenLivePipeline.from_pretrained(snapshot, local_files_only=True)
        assert type(restored) is fixture.pipeline_class
        assert isinstance(restored.decoder, EdenLiveLightTAEDecoder)
        for name in ("transformer", "vae", "decoder", "proxy_encoder", "proxy_projection"):
            expected, actual = getattr(pipe, name).state_dict(), getattr(restored, name).state_dict()
            for key, value in expected.items():
                torch.testing.assert_close(actual[key], value, atol=0, rtol=0)
        assert file_digests(fixture.source) == fixture.hashes

    @pytest.mark.parametrize("dtype_kind", ["scalar", "mapping"])
    def test_original_dtype_override_and_component_identity(self, original_checkpoint, dtype_kind):
        fixture = original_checkpoint
        decoder = RecordingDecoder()
        text_encoder = fixture.components["text_encoder"].to(dtype=torch.float64)
        dtype = torch.float32 if dtype_kind == "scalar" else {"default": torch.float64, "proxy_encoder": torch.float32}
        pipe = EdenLivePipeline.from_pretrained(
            fixture.source,
            profile=fixture.profile,
            decoder=decoder,
            text_encoder=text_encoder,
            dtype=dtype,
            local_files_only=True,
        )
        assert pipe.decoder is decoder and decoder.weight.dtype == torch.float64
        assert pipe.text_encoder is text_encoder and text_encoder.dtype == torch.float64
        assert pipe.transformer.dtype == (torch.float32 if dtype_kind == "scalar" else torch.float64)
        assert pipe.proxy_encoder.dtype == torch.float32
        assert pipe.proxy_projection.dtype == (torch.float32 if dtype_kind == "scalar" else torch.float64)
        assert file_digests(fixture.source) == fixture.hashes

    def test_original_decoder_none_skips_missing_tiny_weights(self, original_checkpoint):
        fixture = original_checkpoint
        size = fixture.profile.split("-")[0]
        for path in (fixture.source / f"decoder-{size}").iterdir():
            path.unlink()
        with pytest.raises((OSError, ValueError), match="decoder|lighttae|No such file|does not appear"):
            EdenLivePipeline.from_pretrained(fixture.source, profile=fixture.profile, local_files_only=True)
        pipe = EdenLivePipeline.from_pretrained(
            fixture.source, profile=fixture.profile, decoder=None, local_files_only=True
        )
        assert pipe.decoder is None
        assert isinstance(pipe.vae, AutoencoderKLWan)

    def test_original_custom_weight_names_and_shared_proxy(self, original_checkpoint):
        fixture = original_checkpoint
        transformer_path = fixture.transformer_dir / "diffusion_pytorch_model.safetensors"
        custom_transformer = transformer_path.with_name("trained-transformer.safetensors")
        transformer_path.rename(custom_transformer)
        proxy_path = fixture.source / "vision-encoder/projection.safetensors"
        custom_proxy = proxy_path.with_name("trained-backbone-and-projector.safetensors")
        proxy_path.rename(custom_proxy)
        size = fixture.profile.split("-")[0]
        decoder_path = next((fixture.source / f"decoder-{size}").glob("*.pth"))
        custom_decoder = decoder_path.with_name("trained-decoder.pth")
        decoder_path.rename(custom_decoder)
        index_path = fixture.source / "model_index.json"
        index = json.loads(index_path.read_text())
        index["_edenlive_original"]["_path_overrides"] = {
            "transformer": {fixture.profile: {"weights": str(custom_transformer.relative_to(fixture.source))}},
            "proxy_encoder": {"weights": str(custom_proxy.relative_to(fixture.source))},
            "proxy_projection": {"weights": str(custom_proxy.relative_to(fixture.source))},
            "decoder": {size: {"weights": str(custom_decoder.relative_to(fixture.source))}},
        }
        index_path.write_text(json.dumps(index))
        hashes = file_digests(fixture.source)
        pipe = EdenLivePipeline.from_pretrained(fixture.source, profile=fixture.profile, local_files_only=True)
        torch.testing.assert_close(
            pipe.proxy_projection.patch_projection.weight,
            fixture.components["proxy_projection"].patch_projection.weight,
            atol=0,
            rtol=0,
        )
        assert isinstance(pipe.decoder, EdenLiveLightTAEDecoder)
        assert file_digests(fixture.source) == hashes

    def test_original_profile_validation(self, original_checkpoint):
        fixture = original_checkpoint
        with pytest.raises(ValueError, match="profile"):
            EdenLivePipeline.from_pretrained(fixture.source, local_files_only=True)

    def test_original_download_selected_profile_files(self, original_hub, tmp_path):
        hub = original_hub
        pipe = EdenLivePipeline.from_pretrained(
            hub.repo_id,
            profile=hub.fixture.profile,
            revision="release-test",
            cache_dir=tmp_path / "cache",
            token="fixture-token",
            local_files_only=False,
            force_download=True,
        )
        assert type(pipe) is EdenLivePipeline and pipe.config.profile == hub.fixture.profile
        assert len(hub.calls.metadata) == len(hub.calls.snapshots) == 1
        options = hub.calls.snapshots[0]
        assert set(options["allow_patterns"]) == set(hub.fixture.hashes)
        assert options["revision"] == hub.revision
        assert options["cache_dir"] == tmp_path / "cache"
        assert options["token"] == "fixture-token" and options["force_download"]
        assert options["local_files_only"] is False
        assert options["user_agent"]["pipeline_class"] == hub.fixture.pipeline_class.__name__

    @pytest.mark.parametrize("original_checkpoint", ["5b-causal"], indirect=True)
    @pytest.mark.parametrize("complete_cache", [False, True])
    def test_original_download_offline_cache(self, original_hub, monkeypatch, tmp_path, complete_cache):
        hub = original_hub

        def unexpected_model_info(*args, **kwargs):
            pytest.fail("Offline loading must not query the remote repository tree.")

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "model_info", unexpected_model_info)
        monkeypatch.setattr(huggingface_hub, "get_cached_repo_tree", unexpected_model_info)
        if not complete_cache:
            metadata_dir = tmp_path / "metadata-only-cache"
            metadata_dir.mkdir()
            index_path = metadata_dir / "model_index.json"
            index_path.write_text((hub.fixture.source / "model_index.json").read_text())

            def metadata_download(repository, filename, **kwargs):
                assert repository == hub.repo_id and filename == "model_index.json"
                assert kwargs["local_files_only"] is True
                return str(index_path)

            monkeypatch.setattr(hub.module, "hf_hub_download", metadata_download)
        pipe = EdenLivePipeline.from_pretrained(
            hub.repo_id, profile=hub.fixture.profile, revision="release-test", local_files_only=True
        )
        assert type(pipe) is EdenLivePipeline and pipe.config.profile == hub.fixture.profile
        if complete_cache:
            assert hub.calls.snapshots == []
            assert hub.calls.metadata[0]["local_files_only"] is True
        else:
            assert len(hub.calls.snapshots) == 1
            assert hub.calls.snapshots[0]["local_files_only"] is True
            assert set(hub.calls.snapshots[0]["allow_patterns"]) == set(hub.fixture.hashes)

    @pytest.mark.parametrize("original_checkpoint", ["14b-causal"], indirect=True)
    @pytest.mark.parametrize("options", [{"variant": "fp16"}, {"device_map": "balanced"}, {"use_safetensors": True}])
    def test_original_download_rejects_unsupported_before_weights(self, original_hub, options):
        hub = original_hub
        with pytest.raises(ValueError, match="does not support|includes PyTorch"):
            EdenLivePipeline.from_pretrained(
                hub.repo_id, profile=hub.fixture.profile, local_files_only=False, force_download=True, **options
            )
        assert len(hub.calls.metadata) == 1
        assert hub.calls.snapshots == []

    @pytest.mark.parametrize("original_checkpoint", ["5b-bidir"], indirect=True)
    def test_original_rejects_unknown_proxy_checkpoint_key(self, original_checkpoint):
        fixture = original_checkpoint
        path = fixture.source / "vision-encoder/projection.safetensors"
        state = {key: value.clone() for key, value in load_file(path).items()}
        state["module.encoder.unrecognized.weight"] = torch.ones(1)
        save_file(state, path)
        with pytest.raises(ValueError, match="Unrecognized proxy checkpoint key"):
            EdenLivePipeline.from_pretrained(fixture.source, profile=fixture.profile, local_files_only=True)


class EdenLiveTransformerTesterConfig:
    model_class = EdenLiveTransformer3DModel
    main_input_name = "hidden_states"
    input_shape = (4, 3, 8, 8)
    output_shape = (4, 3, 8, 8)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "patch_size": (1, 2, 2),
            "num_attention_heads": 4,
            "attention_head_dim": 12,
            "in_channels": 4,
            "out_channels": 4,
            "text_dim": 16,
            "freq_dim": 16,
            "ffn_dim": 32,
            "num_layers": 2,
            "proxy_channels": 8,
            "rope_max_seq_len": 64,
        }

    def get_dummy_inputs(self, batch_size=1, height=8, width=8):
        generator = self.generator
        return {
            "hidden_states": randn_tensor((batch_size, 4, 3, height, width), generator=generator, device=torch_device),
            "proxy_features": randn_tensor((batch_size, 8, 3, 4, 4), generator=generator, device=torch_device),
            "encoder_hidden_states": randn_tensor((batch_size, 12, 16), generator=generator, device=torch_device),
            "timestep": torch.tensor([500.0], device=torch_device).expand(batch_size),
        }


class TestEdenLiveTransformerModel(EdenLiveTransformerTesterConfig):
    @torch.no_grad()
    def test_causal_state_commits_only_clean_chunks_and_preserves_sink(self):
        model = self.model_class(**self.get_init_dict()).eval().to(torch_device)
        inputs = self.get_dummy_inputs()
        state = model.create_causal_state(
            batch_size=1,
            height=8,
            width=8,
            chunk_size=3,
            max_context_frames=18,
            sink_frames=6,
            device=torch_device,
            dtype=torch.float32,
        )
        assert state.next_frame == 0
        model(**inputs, causal_state=state)
        assert state.next_frame == 0
        assert all(layer.num_tokens == 0 for layer in state.layers)
        sink_keys = None
        for index in range(14):
            inputs["timestep"].zero_()
            model(**inputs, causal_state=state, cache_update=True)
            assert state.next_frame == 3 * (index + 1)
            assert all(layer.num_tokens <= 18 * state.tokens_per_frame for layer in state.layers)
            if index == 1:
                sink_keys = [layer.key[:, : 6 * state.tokens_per_frame].clone() for layer in state.layers]
            elif index > 1:
                for expected, layer in zip(sink_keys, state.layers):
                    torch.testing.assert_close(layer.key[:, : 6 * state.tokens_per_frame], expected, atol=0, rtol=0)
        state.reset()
        assert state.next_frame == 0 and all(layer.num_tokens == 0 for layer in state.layers)
        assert not any("cache" in key for key in model.state_dict())

    @torch.no_grad()
    def test_text_timeline_uses_complete_segment_bank(self):
        model = self.model_class(**self.get_init_dict()).eval().to(torch_device)
        attn = model.blocks[0].attn2
        hidden = randn_tensor((1, 12, 48), generator=self.generator, device=torch_device)
        context = randn_tensor((1, 10, 48), generator=self.generator, device=torch_device)
        layout = {"lengths": ((3, 9),), "text_length": 5}
        expected = torch.cat((attn(hidden[:, :3], context[:, :5]), attn(hidden[:, 3:], context[:, 5:])), dim=1)
        actual = attn(hidden, context, text_timeline=layout)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        local = attn(hidden[:, 6:], context, text_timeline=layout, query_start=6)
        torch.testing.assert_close(local, expected[:, 6:], atol=1e-6, rtol=1e-6)


class TestEdenLiveCausalCacheOrder:
    def test_local_chunk_order_survives_eviction_and_short_tail(self, monkeypatch):
        # Capture the real processor's dispatch boundary. Each rank keeps its
        # own chunk slices; Ulysses performs the rank-major exchange downstream.
        processor = transformer_edenlive.EdenLiveCausalAttnProcessor()
        identity = torch.nn.Identity()
        attn = SimpleNamespace(
            fused_projections=False,
            to_q=identity,
            to_k=identity,
            to_v=identity,
            norm_q=identity,
            norm_k=identity,
            heads=1,
            to_out=[identity, identity],
        )
        cache = transformer_edenlive._EdenLiveLayerCache(torch.zeros(1, 6, 1, 2), torch.zeros(1, 6, 1, 2))
        seen = []

        def dispatch(query, key, value, **kwargs):
            seen.append((key.clone(), value.clone()))
            return query

        monkeypatch.setattr(transformer_edenlive, "dispatch_attention_fn", dispatch)

        def call(ids, commit):
            hidden = torch.tensor(ids, dtype=torch.float32).view(1, -1, 1).expand(-1, -1, 2)
            rope = (torch.ones(1, 1, 1, 2), torch.zeros(1, 1, 1, 2))
            processor(
                attn,
                hidden,
                rotary_emb=rope,
                key_rotary_emb=rope,
                layer_cache=cache,
                cache_update=commit,
                cache_window_tokens=6,
                cache_sink_tokens=2,
            )
            assert torch.equal(seen[-1][0], seen[-1][1])
            return seen[-1][0][0, :, 0, 0].tolist()

        assert call([0, 1], True) == [0, 1]
        assert call([4, 5], True) == [0, 1, 4, 5]
        assert call([8, 9], True) == [0, 1, 4, 5, 8, 9]
        assert call([12, 13], True) == [0, 1, 8, 9, 12, 13]
        stored = cache.key.clone()
        assert call([16], False) == [0, 1, 9, 12, 13, 16]
        assert cache.num_tokens == 6 and torch.equal(cache.key, stored)
        assert call([16], True) == [0, 1, 9, 12, 13, 16]


def _parallel_outputs(model, inputs):
    with torch.no_grad():
        output = model(**inputs, return_dict=False)[0]
        state = model.create_causal_state(
            batch_size=1,
            height=8,
            width=8,
            chunk_size=3,
            max_context_frames=18,
            sink_frames=6,
            device=inputs["hidden_states"].device,
            dtype=inputs["hidden_states"].dtype,
        )
        clean_inputs = {**inputs, "timestep": torch.zeros_like(inputs["timestep"])}
        sink_keys = None
        for index in range(8):
            model(**clean_inputs, causal_state=state, cache_update=True)
            assert state.next_frame == 3 * (index + 1)
            assert all(layer.num_tokens <= 18 * state.tokens_per_frame // state.cp_size for layer in state.layers)
            sink_tokens = 6 * state.tokens_per_frame // state.cp_size
            if index == 1:
                sink_keys = [layer.key[:, :sink_tokens].clone() for layer in state.layers]
            elif index > 1:
                for expected, layer in zip(sink_keys, state.layers):
                    torch.testing.assert_close(layer.key[:, :sink_tokens], expected, rtol=0, atol=0)
        tail_inputs = {
            **inputs,
            "hidden_states": inputs["hidden_states"][:, :, :1],
            "proxy_features": inputs["proxy_features"][:, :, :1],
            "timestep": inputs["timestep"][:, :16] if inputs["timestep"].ndim == 2 else inputs["timestep"],
        }
        tail = model(**tail_inputs, causal_state=state, return_dict=False)[0]
        assert state.next_frame == 24
        tail_clean = {**tail_inputs, "timestep": torch.zeros_like(tail_inputs["timestep"])}
        model(**tail_clean, causal_state=state, cache_update=True)
        assert state.next_frame == 25
        return [output.float().cpu(), tail.float().cpu()]


def _context_parallel_worker(rank, world_size, port, init_dict, state_dict, inputs, backend, use_fsdp, output_path):
    # Spawned workers must establish the same settings as the single-device reference.
    enable_full_determinism()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dtype = torch.bfloat16 if backend == "_native_flash" else torch.float32
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=180))
        model = EdenLiveTransformer3DModel(**init_dict).eval()
        model.load_state_dict(state_dict)
        model.to(device=device, dtype=dtype)
        model.set_attention_backend(backend)
        inputs = {
            name: tensor.to(device=device, dtype=torch.float32 if name == "timestep" else dtype)
            for name, tensor in inputs.items()
        }

        def check_timestep(_module, _args, kwargs):
            assert kwargs["timestep"].dtype == torch.float32
            assert torch.isin(kwargs["timestep"], torch.tensor([0.0, 375.0], device=device)).all()

        model.register_forward_pre_hook(check_timestep, with_kwargs=True)
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cuda", (1, world_size), mesh_dim_names=("ring", "ulysses"))
        model.enable_parallelism(config=ContextParallelConfig(ring_degree=1, ulysses_degree=world_size, mesh=mesh))
        if use_fsdp:
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

            policy = MixedPrecisionPolicy(
                param_dtype=dtype, reduce_dtype=torch.float32, output_dtype=dtype, cast_forward_inputs=False
            )
            for block in model.blocks:
                fully_shard(block, mesh=mesh["ulysses"], mp_policy=policy, reshard_after_forward=True)
            fully_shard(model, mesh=mesh["ulysses"], mp_policy=policy, reshard_after_forward=True)
        outputs = _parallel_outputs(model, inputs)
        if rank == 0:
            torch.save(outputs, output_path)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestEdenLiveContextParallel(EdenLiveTransformerTesterConfig):
    @pytest.mark.context_parallel
    @pytest.mark.attention
    @pytest.mark.parametrize("world_size", [2, 4])
    @pytest.mark.parametrize(
        "backend,use_fsdp", [("native", False), ("_native_flash", False), ("_native_flash", True)]
    )
    @pytest.mark.parametrize("num_layers,token_time", [(2, False), (1, True)])
    def test_context_parallel_inference(self, tmp_path, world_size, backend, use_fsdp, num_layers, token_time):
        if not dist.is_available() or not dist.is_nccl_available() or torch.cuda.device_count() < world_size:
            pytest.skip(f"This case requires {world_size} CUDA devices with NCCL.")
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            enable_full_determinism()
            config = {**self.get_init_dict(), "attention_head_dim": 16, "num_layers": num_layers}
            model = EdenLiveTransformer3DModel(**config).eval()
            state = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}
            inputs = {name: tensor.cpu() for name, tensor in self.get_dummy_inputs().items()}
            # This value would be rounded to 376 by an indiscriminate BF16 input cast.
            inputs["timestep"] = torch.tensor([375.0], dtype=torch.float32)
            if token_time:
                inputs["timestep"] = torch.full((1, 48), 375.0)
                inputs["timestep"][:, :16] = 0
            dtype = torch.bfloat16 if backend == "_native_flash" else torch.float32
            model.to(device="cuda:0", dtype=dtype).set_attention_backend(backend)
            reference_inputs = {
                name: tensor.to(device="cuda:0", dtype=torch.float32 if name == "timestep" else dtype)
                for name, tensor in inputs.items()
            }
            expected = _parallel_outputs(model, reference_inputs)
            del model, reference_inputs
            torch.cuda.empty_cache()
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            output_path = str(tmp_path / "context_parallel.pt")
            mp.spawn(
                _context_parallel_worker,
                args=(world_size, port, config, state, inputs, backend, use_fsdp, output_path),
                nprocs=world_size,
                join=True,
            )
            actual = torch.load(output_path, map_location="cpu", weights_only=True)
            for reference, output in zip(expected, actual):
                assert torch.isfinite(output).all()
                # Keep the existing small-model CP backend tolerance.
                torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


class EdenLiveFlowMatchEulerDiscreteSchedulerTests(unittest.TestCase):
    def test_shifted_schedule(self):
        scheduler = EdenLiveFlowMatchEulerDiscreteScheduler(shift=3.0)
        scheduler.set_timesteps(30)
        self.assertEqual(scheduler.sigmas.dtype, torch.float32)
        self.assertEqual(len(scheduler.timesteps), 30)
        torch.testing.assert_close(
            scheduler.sigmas[[0, 1, 15, 29, 30]],
            torch.tensor([0.9999989867210388, 0.9886479377746582, 0.7503749132156372, 0.09629422426223755, 0.0]),
            rtol=0,
            atol=0,
        )

    def test_explicit_timesteps_are_not_shifted(self):
        for points in ([1000, 937.5, 833.3333333333, 625], [1000, 900, 750, 500]):
            scheduler = EdenLiveFlowMatchEulerDiscreteScheduler(shift=9.0, default_timesteps=points)
            scheduler.set_timesteps(4)
            torch.testing.assert_close(scheduler.sigmas[:-1], torch.tensor(points) / 1000, rtol=0, atol=0)
            scheduler.set_timesteps(sigmas=[1.0, 0.5, 0.0])
            torch.testing.assert_close(scheduler.sigmas, torch.tensor([1.0, 0.5, 0.0]), rtol=0, atol=0)
            self.assertEqual(len(scheduler.timesteps), 2)

    def test_stochastic_dtype_and_variance_noise(self):
        for dtype in (torch.float32, torch.bfloat16):
            scheduler = EdenLiveFlowMatchEulerDiscreteScheduler(stochastic_sampling=True)
            scheduler.set_timesteps(timesteps=[1000, 900])
            generator = torch.Generator().manual_seed(9)
            sample = torch.randn((2, 3, 4), generator=generator).to(dtype)
            prediction = torch.randn(sample.shape, generator=generator).to(dtype)
            noise_generator = torch.Generator().manual_seed(12)
            noise = torch.randn(sample.shape, generator=noise_generator, dtype=dtype)
            explicit = scheduler.step(prediction, scheduler.timesteps[0], sample, variance_noise=noise).prev_sample
            scheduler.set_timesteps(timesteps=[1000, 900])
            generated = scheduler.step(
                prediction, scheduler.timesteps[0], sample, generator=torch.Generator().manual_seed(12)
            ).prev_sample
            self.assertEqual(explicit.dtype, dtype)
            torch.testing.assert_close(explicit, generated, rtol=0, atol=0)

    def test_final_step_does_not_advance_generator(self):
        scheduler = EdenLiveFlowMatchEulerDiscreteScheduler(stochastic_sampling=True)
        scheduler.set_timesteps(timesteps=[500])
        generator = torch.Generator().manual_seed(7)
        state = generator.get_state().clone()
        sample = torch.ones((1, 2, 3), dtype=torch.bfloat16)
        result = scheduler.step(sample, scheduler.timesteps[0], sample, generator=generator).prev_sample
        self.assertTrue(torch.equal(state, generator.get_state()))
        self.assertTrue(torch.equal(result, sample * 0.5))

    def test_chunk_reset_and_serialization(self):
        scheduler = EdenLiveFlowMatchEulerDiscreteScheduler(
            stochastic_sampling=True, default_timesteps=[1000, 900, 750, 500]
        )
        sample = torch.ones((1, 2, 3))
        scheduler.set_timesteps(4)
        result = scheduler.step(sample, scheduler.timesteps[0], sample, variance_noise=sample).prev_sample
        self.assertEqual(scheduler.step_index, 1)
        scheduler.set_timesteps(4)
        self.assertIsNone(scheduler.step_index)
        repeated = scheduler.step(sample, scheduler.timesteps[0], sample, variance_noise=sample).prev_sample
        torch.testing.assert_close(result, repeated, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            scheduler.save_pretrained(directory)
            restored = EdenLiveFlowMatchEulerDiscreteScheduler.from_pretrained(directory)
        restored.set_timesteps(4)
        torch.testing.assert_close(restored.sigmas, scheduler.sigmas, rtol=0, atol=0)
        self.assertIsNone(restored.step_index)
        self.assertTrue(restored.config.stochastic_sampling)

    def test_invalid_schedule(self):
        scheduler = EdenLiveFlowMatchEulerDiscreteScheduler(default_timesteps=[1000, 500])
        for kwargs in (
            {"timesteps": [1000], "sigmas": [1]},
            {"timesteps": [500, 1000]},
            {"timesteps": [1000, 1000, 500]},
            {"timesteps": [float("nan")]},
            {"sigmas": []},
            {"sigmas": [1.1, 0.5]},
            {"num_inference_steps": 4},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                scheduler.set_timesteps(**kwargs)
