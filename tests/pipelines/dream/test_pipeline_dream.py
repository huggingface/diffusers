import gc
import unittest

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from diffusers import (
    DreamMaskedDiffusionScheduler,
    DreamTextPipeline,
    DreamTokenizer,
    DreamTransformer1DModel,
)
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    nightly,
    require_big_accelerator,
    slow,
    torch_device,
)

from ..test_pipelines_common import PipelineTesterMixin


class DreamTextPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = DreamTextPipeline
    params = frozenset(["prompt", "prompt_embeds"])  # TODO: add as necessary
    batch_params = frozenset(["prompt"])

    # there is no xformers processor for Dream
    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True

    def get_dummy_components(
        self,
        num_layers: int = 1,
        attention_head_dim: int = 16,
        num_attention_heads: int = 4,
        num_attention_kv_heads: int = 2,
        ff_intermediate_dim: int = 256,
        vocab_size: int = 152064,
        pad_token_id: int = 151643,
        mask_token_id: int = 151666,
        start_token_id: int = 151643,
        logit_sampling_alg: str = "entropy",
    ):
        torch.manual_seed(0)
        transformer = DreamTransformer1DModel(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            num_attention_kv_heads=num_attention_kv_heads,
            ff_intermediate_dim=ff_intermediate_dim,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
        )

        # For now, use the full Dream tokenizer, although this will cause the model's embedding weights to be
        # relatively large
        model_repo = "Dream-org/Dream-v0-Instruct-7B"
        tokenizer = DreamTokenizer.from_pretrained(model_repo)

        scheduler = DreamMaskedDiffusionScheduler(
            logit_sampling_alg=logit_sampling_alg,
            mask_token_id=mask_token_id,
            start_token_id=start_token_id,
        )

        components = {
            "scheduler": scheduler,
            "tokenizer": tokenizer,
            "transformer": transformer,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=torch_device).manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "max_sequence_length": 48,
            "output_type": "latent",
        }
        return inputs

    def test_dream_text_output_shape(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        text_latents = pipe(**inputs).texts[0]

        seq_len = text_latents.shape[0]
        expected_seq_len = inputs["max_sequence_length"]
        self.assertEqual(
            seq_len, expected_seq_len, f"Output seq len {seq_len} does not match expected seq_len {expected_seq_len}"
        )

    def test_dream_text_output_shape_origin(self):
        components = self.get_dummy_components(logit_sampling_alg="origin")
        pipe = self.pipeline_class(**components).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        text_latents = pipe(**inputs).texts[0]

        seq_len = text_latents.shape[0]
        expected_seq_len = inputs["max_sequence_length"]
        self.assertEqual(
            seq_len, expected_seq_len, f"Output seq len {seq_len} does not match expected seq_len {expected_seq_len}"
        )


@nightly
@require_big_accelerator
class DreamTextPipelineSlowTests(unittest.TestCase):
    pipeline_class = DreamTextPipeline
    repo_id = "Dream-org/Dream-v0-Instruct-7B"

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, seed=0):
        pass

    def test_dream_inference(self):
        pass
