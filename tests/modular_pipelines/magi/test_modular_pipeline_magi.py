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

import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, T5Config, T5EncoderModel

from diffusers import (
    AutoencoderKLMagi,
    MagiClassifierFreeGuidance,
    MagiEulerScheduler,
    MagiModularPipeline,
    MagiTextConditioningModel,
    MagiTextToVideoBlocks,
    MagiTransformer3DModel,
    ModularPipeline,
)

from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
)
from .testing_utils import MagiGuiderTesterMixin


@pytest.fixture(scope="module")
def tiny_magi_path(tmp_path_factory):
    torch.manual_seed(0)
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2, "cat": 3, "runs": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]")
    pipe = MagiTextToVideoBlocks().init_pipeline()
    conditioning = MagiTextConditioningModel(caption_channels=16, caption_max_length=8, null_token_length=4)
    with torch.no_grad():
        conditioning.null_embedding.weight.normal_()
        conditioning.special_embedding.weight.normal_()
    torch.manual_seed(0)
    text_encoder = T5EncoderModel(T5Config(vocab_size=5, d_model=16, d_ff=32, d_kv=8, num_heads=2, num_layers=1))
    torch.manual_seed(0)
    transformer = MagiTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        attention_head_dim=32,
        ffn_dim=96,
        condition_dim=16,
        caption_channels=16,
        caption_max_length=8,
        frequency_embedding_size=16,
    )
    torch.manual_seed(0)
    vae = AutoencoderKLMagi(
        latent_channels=4,
        embed_dim=32,
        num_layers=1,
        num_attention_heads=4,
        mlp_ratio=2,
        patch_size=2,
        patch_length=4,
        sample_size=8,
        sample_frames=8,
    )
    pipe.update_components(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        text_conditioning=conditioning,
        transformer=transformer,
        vae=vae,
        scheduler=MagiEulerScheduler(),
        guider=MagiClassifierFreeGuidance(),
    )
    pipe.load_components()
    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module):
            component.eval()
    path = str(tmp_path_factory.mktemp("tiny-magi"))
    pipe.save_pretrained(path, overwrite_modular_index=True)
    return path


class MagiPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = MagiModularPipeline
    pipeline_blocks_class = MagiTextToVideoBlocks
    params = frozenset(["prompt", "height", "width", "num_frames"])
    batch_params = frozenset(["prompt"])
    output_name = "videos"

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def model_path(cls, tiny_magi_path):
        cls.pretrained_model_name_or_path = tiny_magi_path

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a cat runs",
            "height": 8,
            "width": 8,
            "num_frames": 16,
            "chunk_width": 2,
            "window_size": 2,
            "num_inference_steps": 4,
            "max_sequence_length": 8,
            "clean_caption": False,
            "output_type": "pt",
            "generator": self.get_generator(seed),
        }


class TestMagiPipelineFast(MagiPipelineTesterConfig, ModularPipelineTesterMixin):
    def test_convert_sharded_t5(self, tmp_path):
        from scripts.convert_magi_to_diffusers import load_t5

        torch.manual_seed(0)
        model = T5EncoderModel(T5Config(vocab_size=5, d_model=16, d_ff=32, d_kv=8, num_heads=2, num_layers=1))
        model.config.save_pretrained(tmp_path)
        tensors = list(model.state_dict().items())
        mapping = {}
        for index in range(2):
            name = f"pytorch_model-{index + 1:05d}-of-00002.bin"
            shard = dict(tensors[index::2])
            torch.save(shard, tmp_path / name)
            mapping.update(dict.fromkeys(shard, name))
        (tmp_path / "pytorch_model.bin.index.json").write_text(json.dumps({"weight_map": mapping}))
        restored = load_t5(tmp_path)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, restored.state_dict()[name], atol=0, rtol=0)
        assert not list(tmp_path.glob("*.safetensors"))

    def test_repeat_and_roundtrip(self, tmp_path):
        pipe = self.get_pipeline()
        expected = self.run_pipe(pipe)
        torch.testing.assert_close(self.run_pipe(pipe), expected, rtol=0, atol=0)
        pipe.save_pretrained(str(tmp_path), overwrite_modular_index=True)
        restored = ModularPipeline.from_pretrained(str(tmp_path))
        restored.load_components()
        torch.testing.assert_close(self.run_pipe(restored), expected, rtol=0, atol=0)
        assert expected.shape == (1, 16, 3, 8, 8)

    def test_chunk_conditioning_and_decode(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        output = pipe(
            **inputs,
            output=[
                "videos",
                "latents",
                "prompt_embeds",
                "prompt_attention_mask",
                "negative_prompt_embeds",
                "negative_prompt_attention_mask",
                "text_embeds",
            ],
        )
        text = output["prompt_embeds"]
        torch.testing.assert_close(text[0, :, 0], pipe.text_conditioning.special_embedding.weight[[2, 1]])
        torch.testing.assert_close(text[0, :, 1], pipe.text_conditioning.special_embedding.weight[0].expand(2, -1))
        torch.testing.assert_close(text[:, :, 2:], output["text_embeds"][:, None, :6].expand(-1, 2, -1, -1))
        torch.testing.assert_close(output["negative_prompt_embeds"][0], pipe.text_conditioning.null_embedding.weight)
        assert output["negative_prompt_attention_mask"].sum().item() == 4
        chunks = []
        with torch.no_grad():
            for chunk in output["latents"].split(2, dim=2):
                chunks.append(pipe.vae.decode(chunk / 0.18215).sample)
        expected = pipe.video_processor.postprocess_video(torch.cat(chunks, dim=2), output_type="pt")
        torch.testing.assert_close(output["videos"], expected, rtol=0, atol=0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @torch.no_grad()
    def test_low_precision_decode_matches_autocast(self, dtype):
        from diffusers.modular_pipelines.magi.decoders import MagiVaeDecoderStep

        pipe = MagiVaeDecoderStep().init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=dtype)
        pipe.to("cuda")
        torch.manual_seed(0)
        latents = torch.randn(1, 4, 4, 8, 8, device="cuda")
        actual = pipe(latents=latents, chunk_width=2, output_type="pt", output="videos")
        chunks = []
        with torch.autocast("cuda", dtype=dtype):
            for chunk in latents.split(2, dim=2):
                chunks.append(pipe.vae.decode((chunk / 0.18215).to(dtype), num_frames=8).sample)
        expected = pipe.video_processor.postprocess_video(torch.cat(chunks, dim=2).float(), output_type="pt")
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_single_latent_frame_is_video(self):
        pipe = self.get_pipeline()
        video = self.run_pipe(pipe, num_frames=4, chunk_width=1)
        assert video.shape[1] == 4

    def test_round_up_frames(self):
        pipe = self.get_pipeline()
        video = self.run_pipe(pipe, num_frames=12)
        assert video.shape[1] == 16

    def test_caption_cleaning(self):
        pytest.importorskip("ftfy")
        pytest.importorskip("bs4")
        pipe = self.get_pipeline()
        first = self.run_pipe(pipe, prompt="<p>A CAT runs</p> https://example.com @someone", clean_caption=True)
        second = self.run_pipe(pipe, prompt="a cat runs", clean_caption=False)
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_duration_saturates(self):
        pipe = self.get_pipeline()
        prepare = pipe.blocks.sub_blocks["prepare_latents"].init_pipeline(self.pretrained_model_name_or_path)
        prepare.load_components()
        features = torch.randn(1, 8, 16)
        result = prepare(
            text_embeds=features,
            text_attention_mask=torch.ones(1, 8, dtype=torch.bool),
            num_frames=80,
            height=8,
            width=8,
            chunk_width=2,
            output=["prompt_embeds", "latents"],
        )
        torch.testing.assert_close(
            result["prompt_embeds"][0, :, 0],
            prepare.text_conditioning.special_embedding.weight[[8, 8, 8, 7, 6, 5, 4, 3, 2, 1]],
        )
        assert result["latents"].shape[2] == 20

    def test_output_formats_and_latent(self):
        pipe = self.get_pipeline()
        latent = self.run_pipe(pipe, output_type="latent")
        assert latent.shape == (1, 4, 4, 4, 4)
        pt = self.run_pipe(pipe)
        array = self.run_pipe(pipe, output_type="np")
        np.testing.assert_allclose(array, pt.permute(0, 1, 3, 4, 2).numpy(), rtol=0, atol=0)
        pil = self.run_pipe(pipe, output_type="pil")
        assert len(pil) == 1 and len(pil[0]) == 16
        assert pil[0][0].size == (8, 8)

    @pytest.mark.parametrize(
        "changed",
        [
            {"prompt": []},
            {"height": 7},
            {"num_frames": 7},
            {"num_images_per_prompt": 0},
            {"latents": torch.zeros(1)},
            {"prefix_latents": torch.zeros(1)},
            {"max_sequence_length": 7},
            {"output_type": "invalid"},
        ],
    )
    def test_invalid_input(self, changed):
        with pytest.raises(ValueError):
            self.run_pipe(self.get_pipeline(), **changed)

    def test_pipeline_converter(self, tiny_magi_path, tmp_path):
        from scripts.convert_magi_to_diffusers import convert_pipeline

        pipe = MagiModularPipeline.from_pretrained(tiny_magi_path)
        pipe.load_components()
        transformer_config = dict(pipe.transformer.config)
        transformer_config["caption_max_length"] = 64
        transformer = MagiTransformer3DModel.from_config(transformer_config)
        transformer.save_pretrained(str(tmp_path / "transformer"))
        pipe.text_encoder.save_pretrained(str(tmp_path / "t5"))
        pipe.tokenizer.save_pretrained(str(tmp_path / "t5"))
        other = np.random.default_rng(0).normal(size=(100, 16)).astype(np.float32)
        np.savez(tmp_path / "special.npz", other_tokens=other)
        converted = convert_pipeline(transformer, pipe.vae, str(tmp_path / "t5"), tmp_path / "special.npz")
        expected = torch.from_numpy(other[[1, *range(7, 15)]].astype(np.float16)).float()
        torch.testing.assert_close(converted.text_conditioning.special_embedding.weight, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            converted.text_conditioning.null_embedding.weight,
            transformer.condition_embedder.y_embedder.null_caption_embedding,
            atol=0,
            rtol=0,
        )
        output = converted(
            prompt="a cat",
            height=8,
            width=8,
            num_frames=8,
            chunk_width=2,
            num_inference_steps=4,
            window_size=2,
            max_sequence_length=64,
            clean_caption=False,
            output_type="pt",
            output="videos",
        )
        assert output.shape == (1, 8, 3, 8, 8)
        assert output.isfinite().all()


class TestMagiPipelineLoading(MagiPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestMagiPipelineMemory(MagiPipelineTesterConfig, ModularMemoryTesterMixin):
    pass


class TestMagiPipelineGuider(MagiPipelineTesterConfig, MagiGuiderTesterMixin):
    pass
