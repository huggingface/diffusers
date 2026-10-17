# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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


from types import SimpleNamespace

import pytest
import torch

from diffusers import BlockRefinementScheduler, EntropyBoundScheduler
from diffusers.modular_pipelines import DiffusionGemmaBlocks, DiffusionGemmaModularPipeline, ModularPipeline

from ...testing_utils import torch_device
from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


class DiffusionGemmaModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = DiffusionGemmaModularPipeline
    pipeline_blocks_class = DiffusionGemmaBlocks
    pretrained_model_name_or_path = "akshan-main/tiny-diffusion-gemma-modular-pipe"

    params = frozenset(["prompt", "messages", "gen_length"])
    batch_params = frozenset(["prompt"])
    optional_params = frozenset(["num_inference_steps", "temperature", "eos_token_id"])
    output_name = "sequences"

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "Why is the sky blue?",
            "generator": self.get_generator(seed),
            "gen_length": 32,
            "num_inference_steps": 2,
        }


class TestDiffusionGemmaModularPipelineFast(DiffusionGemmaModularPipelineTesterConfig, ModularPipelineTesterMixin):
    @pytest.mark.skip(
        reason="The canvas noise is drawn from a single generator for the whole batch (same as the standard "
        "pipeline), so the per-prompt generator lists this test passes are not supported"
    )
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(
        reason="The canvas noise is drawn from a single generator for the whole batch (same as the standard "
        "pipeline), so the per-prompt generator lists this test passes are not supported"
    )
    def test_inference_batch_single_identical(self):
        pass

    adaptive_stopping_vocab_size = 8

    def _run_adaptive_stopping(self, pipe, prompt):
        pipe.model.config.get_text_config(decoder=True).vocab_size = self.adaptive_stopping_vocab_size
        return pipe(
            prompt=prompt,
            gen_length=32,
            num_inference_steps=5,
            confidence_threshold=0.005,
            eos_early_stop=False,
            generator=self.get_generator(),
            output="sequences",
        )

    def test_adaptive_stopping_freezes_finished_rows(self):
        pipe = self.get_pipeline().to(torch_device)
        forward_calls = 0

        def forward(decoder_input_ids, **kwargs):
            nonlocal forward_calls
            batch_size, canvas_length = decoder_input_ids.shape
            token_ids = ([1, 3], [1, 4], [2, 5], [2, 5], [2, 6])[forward_calls]
            tokens = torch.tensor(token_ids, device=decoder_input_ids.device)[:, None].expand_as(decoder_input_ids)
            logits = torch.full(
                (batch_size, canvas_length, self.adaptive_stopping_vocab_size),
                -100.0,
                device=decoder_input_ids.device,
            )
            logits.scatter_(-1, tokens[..., None], 100.0)
            forward_calls += 1
            return SimpleNamespace(logits=logits)

        pipe.model.forward = forward
        pipe.update_components(scheduler=BlockRefinementScheduler())
        sequences = self._run_adaptive_stopping(
            pipe, ["Short prompt.", "A somewhat longer prompt for the second batch row."]
        )

        # The first row is stable from the second step and the second row from the fourth, so the loop runs four
        # of the five steps and each row keeps the prediction it was frozen on.
        assert forward_calls == 4
        assert bool((sequences[0] == 1).all())
        assert bool((sequences[1] == 5).all())

    def test_adaptive_stopping_uses_scheduler_logits(self):
        pipe = self.get_pipeline().to(torch_device)
        forward_calls = 0

        def forward(decoder_input_ids, **kwargs):
            nonlocal forward_calls
            forward_calls += 1
            batch_size, canvas_length = decoder_input_ids.shape
            logits = torch.zeros(
                batch_size, canvas_length, self.adaptive_stopping_vocab_size, device=decoder_input_ids.device
            )
            logits[..., 0] = 2.0
            return SimpleNamespace(logits=logits)

        pipe.model.forward = forward
        pipe.update_components(scheduler=EntropyBoundScheduler(t_max=0.1, t_min=0.1))
        sequences = self._run_adaptive_stopping(pipe, "Name a color.")

        assert forward_calls == 2
        assert bool((sequences == 0).all())

    def test_text_output(self):
        pipe = self.get_pipeline().to("cpu")

        inputs = self.get_dummy_inputs()
        state = pipe(**inputs)
        sequences = state.get("sequences")
        texts = state.get("texts")

        assert sequences.dtype == torch.long
        assert sequences.shape == (1, 32)
        vocab_size = pipe.model.config.get_text_config(decoder=True).vocab_size
        assert bool((sequences >= 0).all()) and bool((sequences < vocab_size).all())
        assert isinstance(texts, list) and len(texts) == 1 and isinstance(texts[0], str)


class TestDiffusionGemmaModularPipelineLoading(DiffusionGemmaModularPipelineTesterConfig, ModularLoadingTesterMixin):
    def test_save_from_pretrained(self, tmp_path, base_pipe_output):
        # The base test compares an image slice; text output is compared token for token.
        base_pipe = self.get_pipeline().to(torch_device)
        base_pipe.save_pretrained(str(tmp_path))

        pipe = ModularPipeline.from_pretrained(tmp_path)
        pipe.load_components(dtype=torch.float32)
        pipe.to(torch_device)

        sequences = pipe(**self.get_dummy_inputs(), output=self.output_name)
        assert torch.equal(sequences, base_pipe_output)


class TestDiffusionGemmaModularPipelineWorkflow(DiffusionGemmaModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestDiffusionGemmaModularPipelineMemory(DiffusionGemmaModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass
