import unittest

import torch

from diffusers import DFlashTokenDiffusionScheduler
from diffusers.pipelines.dflash.pipeline_dflash import DFlashPipeline


class _DummyModelOutput:
    def __init__(self, logits, hidden_states=None):
        self.logits = logits
        self.hidden_states = hidden_states


class _DummyConfig:
    def __init__(self, block_size, num_target_layers, num_hidden_layers):
        self.block_size = block_size
        self.num_target_layers = num_target_layers
        self.num_hidden_layers = num_hidden_layers


class _DummyTargetModel(torch.nn.Module):
    """Minimal target (causal LM) model that returns logits and hidden_states."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        output_hidden_states=False,
        logits_to_keep=None,
        **kwargs,
    ):
        bsz, seq_len = input_ids.shape
        h = self.embed(input_ids)
        # Create hidden_states list: one entry per layer + 1 for the embedding layer
        hidden_states = [h] * (self.num_layers + 1) if output_hidden_states else None
        logits = self.lm_head(h)
        # Make token 0 the most likely so acceptance is deterministic
        logits[:, :, 0] = 10.0
        return _DummyModelOutput(logits=logits, hidden_states=hidden_states)

    def parameters(self):
        return super().parameters()


class _DummyDraftModel(torch.nn.Module):
    """Minimal draft model that returns hidden states of the expected shape."""

    def __init__(self, hidden_dim: int, num_target_layers: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.config = _DummyConfig(
            block_size=block_size,
            num_target_layers=num_target_layers,
            num_hidden_layers=1,
        )
        # The draft model receives concatenated hidden states from num_target_layers target layers,
        # each of dim hidden_dim, and produces a hidden state of dim hidden_dim.
        self.proj = torch.nn.Linear(hidden_dim * num_target_layers, hidden_dim, bias=False)
        self.register_buffer("_device_anchor", torch.empty(0))

    @property
    def device(self):
        return self._device_anchor.device

    def forward(
        self,
        target_hidden,
        noise_embedding,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        is_causal=False,
        **kwargs,
    ):
        # Return a tensor with shape (batch, seq_len, hidden_dim)
        bsz = noise_embedding.shape[0]
        seq_len = position_ids.shape[1] if position_ids is not None else noise_embedding.shape[1]
        h = torch.zeros(bsz, seq_len, self.proj.out_features, device=noise_embedding.device)
        return h


def _make_pipeline(tokenizer=None, vocab_size=32, hidden_dim=16, num_target_layers=4, block_size=4):
    target = _DummyTargetModel(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=num_target_layers)
    draft = _DummyDraftModel(hidden_dim=hidden_dim, num_target_layers=1, block_size=block_size)
    # Set target_layer_ids directly so we skip the config-based computation.
    draft.target_layer_ids = [1]
    scheduler = DFlashTokenDiffusionScheduler()
    return DFlashPipeline(draft_model=draft, target_model=target, tokenizer=tokenizer, scheduler=scheduler)


class DFlashPipelineTest(unittest.TestCase):
    # ------------------------------------------------------------------
    # Pipeline runs
    # ------------------------------------------------------------------
    def test_pipeline_runs_with_input_ids(self):
        pipe = _make_pipeline()
        input_ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)

        out = pipe(
            input_ids=input_ids,
            max_new_tokens=8,
            temperature=0.0,
            mask_token_id=31,
            stop_token_ids=None,
            output_type="seq",
        )

        self.assertIsNotNone(out.sequences)
        self.assertEqual(out.sequences.ndim, 2)
        self.assertEqual(out.sequences.shape[0], 1)
        # Generated tokens should not be longer than max_new_tokens
        self.assertLessEqual(out.sequences.shape[1], 8)

    # ------------------------------------------------------------------
    # output_type="seq"
    # ------------------------------------------------------------------
    def test_output_type_seq(self):
        pipe = _make_pipeline()
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        out = pipe(
            input_ids=input_ids,
            max_new_tokens=8,
            temperature=0.0,
            mask_token_id=31,
            output_type="seq",
        )

        self.assertIsNotNone(out.sequences)
        self.assertIsNone(out.texts)

    # ------------------------------------------------------------------
    # output_type="text" with mock tokenizer
    # ------------------------------------------------------------------
    def test_output_type_text_with_tokenizer(self):
        tok = type(
            "Tok",
            (),
            {
                "eos_token_id": None,
                "mask_token_id": 31,
                "batch_decode": lambda self, seqs, **kw: [f"decoded_{len(s)}" for s in seqs],
            },
        )()
        pipe = _make_pipeline(tokenizer=tok)

        out = pipe(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            max_new_tokens=8,
            temperature=0.0,
            output_type="text",
        )

        self.assertIsNotNone(out.sequences)
        self.assertIsNotNone(out.texts)
        self.assertEqual(len(out.texts), 1)
        self.assertTrue(out.texts[0].startswith("decoded_"))

    def test_output_type_text_without_tokenizer(self):
        """output_type='text' without a tokenizer should return texts=None."""
        pipe = _make_pipeline(tokenizer=None)

        out = pipe(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            max_new_tokens=8,
            temperature=0.0,
            mask_token_id=31,
            output_type="text",
        )

        self.assertIsNotNone(out.sequences)
        self.assertIsNone(out.texts)

    # ------------------------------------------------------------------
    # output_type invalid
    # ------------------------------------------------------------------
    def test_output_type_invalid_raises(self):
        pipe = _make_pipeline()

        with self.assertRaises(ValueError):
            pipe(
                input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
                max_new_tokens=8,
                mask_token_id=31,
                output_type="invalid",
            )

    # ------------------------------------------------------------------
    # return_dict=False
    # ------------------------------------------------------------------
    def test_pipeline_return_tuple(self):
        pipe = _make_pipeline()
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        result = pipe(
            input_ids=input_ids,
            max_new_tokens=8,
            temperature=0.0,
            mask_token_id=31,
            output_type="seq",
            return_dict=False,
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        sequences, texts = result
        self.assertIsNotNone(sequences)
        self.assertIsNone(texts)

    # ------------------------------------------------------------------
    # check_inputs validation
    # ------------------------------------------------------------------
    def test_check_inputs_no_inputs_raises(self):
        pipe = _make_pipeline()
        with self.assertRaises(ValueError):
            pipe.check_inputs(
                prompt=None,
                messages=None,
                input_ids=None,
                max_new_tokens=16,
                output_type="seq",
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
            )

    def test_check_inputs_both_prompt_and_messages_raises(self):
        pipe = _make_pipeline()
        with self.assertRaises(ValueError):
            pipe.check_inputs(
                prompt="hello",
                messages=[{"role": "user", "content": "hi"}],
                input_ids=None,
                max_new_tokens=16,
                output_type="seq",
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
            )

    def test_check_inputs_invalid_input_ids_ndim_raises(self):
        pipe = _make_pipeline()
        bad_ids = torch.zeros(2, 3, 4, dtype=torch.long)
        with self.assertRaises(ValueError):
            pipe.check_inputs(
                prompt=None,
                messages=None,
                input_ids=bad_ids,
                max_new_tokens=16,
                output_type="seq",
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
            )

    def test_check_inputs_invalid_input_ids_dtype_raises(self):
        pipe = _make_pipeline()
        bad_ids = torch.zeros(1, 4, dtype=torch.float32)
        with self.assertRaises(ValueError):
            pipe.check_inputs(
                prompt=None,
                messages=None,
                input_ids=bad_ids,
                max_new_tokens=16,
                output_type="seq",
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
            )

    def test_check_inputs_invalid_max_new_tokens_raises(self):
        pipe = _make_pipeline()
        with self.assertRaises(ValueError):
            pipe.check_inputs(
                prompt=None,
                messages=None,
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                max_new_tokens=0,
                output_type="seq",
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
            )

    def test_check_inputs_invalid_output_type_raises(self):
        pipe = _make_pipeline()
        with self.assertRaises(ValueError):
            pipe.check_inputs(
                prompt=None,
                messages=None,
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                max_new_tokens=16,
                output_type="bad",
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
            )

    def test_check_inputs_prompt_without_tokenizer_raises(self):
        pipe = _make_pipeline(tokenizer=None)
        with self.assertRaises(ValueError):
            pipe.check_inputs(
                prompt="hello",
                messages=None,
                input_ids=None,
                max_new_tokens=16,
                output_type="seq",
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
            )

    def test_check_inputs_messages_without_tokenizer_raises(self):
        pipe = _make_pipeline(tokenizer=None)
        with self.assertRaises(ValueError):
            pipe.check_inputs(
                prompt=None,
                messages=[{"role": "user", "content": "hi"}],
                input_ids=None,
                max_new_tokens=16,
                output_type="seq",
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
            )

    def test_check_inputs_valid_input_ids_passes(self):
        pipe = _make_pipeline()
        # Should not raise.
        pipe.check_inputs(
            prompt=None,
            messages=None,
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            max_new_tokens=16,
            output_type="seq",
            callback_on_step_end=None,
            callback_on_step_end_tensor_inputs=None,
        )

    # ------------------------------------------------------------------
    # _prepare_input_ids
    # ------------------------------------------------------------------
    def test_prepare_input_ids_from_tensor(self):
        pipe = _make_pipeline()
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        result = pipe._prepare_input_ids(
            prompt=None,
            messages=None,
            input_ids=ids,
            use_chat_template=False,
            add_generation_prompt=False,
            chat_template_kwargs=None,
        )
        self.assertTrue(torch.equal(result, ids))

    def test_prepare_input_ids_from_1d_tensor(self):
        pipe = _make_pipeline()
        ids = torch.tensor([1, 2, 3], dtype=torch.long)
        result = pipe._prepare_input_ids(
            prompt=None,
            messages=None,
            input_ids=ids,
            use_chat_template=False,
            add_generation_prompt=False,
            chat_template_kwargs=None,
        )
        self.assertEqual(result.shape, (1, 3))

    # ------------------------------------------------------------------
    # prepare_latents
    # ------------------------------------------------------------------
    def test_prepare_latents(self):
        pipe = _make_pipeline()
        mask_token_id = 99
        latents = pipe.prepare_latents(
            max_length=10, block_size=4, mask_token_id=mask_token_id, device=torch.device("cpu")
        )
        self.assertEqual(latents.shape, (1, 14))  # 10 + 4
        self.assertTrue((latents == mask_token_id).all().item())
        self.assertEqual(latents.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
