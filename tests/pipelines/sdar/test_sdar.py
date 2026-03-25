import unittest

import torch

from diffusers import SDARPipeline, SDARTokenDiffusionScheduler


class _DummyModelOutput:
    def __init__(self, logits):
        self.logits = logits


class _DummyCausalLM(torch.nn.Module):
    """Minimal causal LM that returns deterministic logits given input_ids."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.register_buffer("_device_anchor", torch.empty(0))

    @property
    def dtype(self):
        return torch.float32

    @property
    def device(self):
        return self._device_anchor.device

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros((batch_size, seq_len, self.vocab_size), device=input_ids.device, dtype=torch.float32)
        # Make confidence vary with token position so top-k commits are deterministic.
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.float32).view(1, seq_len, 1)
        token_ids = (torch.arange(seq_len, device=input_ids.device) % (self.vocab_size - 2)).view(1, seq_len, 1)
        logits.scatter_(2, token_ids.expand(batch_size, -1, -1), 1.0 + positions.expand(batch_size, -1, -1) * 0.1)
        return _DummyModelOutput(logits=logits)


def _make_pipeline(tokenizer=None):
    model = _DummyCausalLM(vocab_size=32)
    scheduler = SDARTokenDiffusionScheduler()
    return SDARPipeline(model=model, scheduler=scheduler, tokenizer=tokenizer)


class SDARPipelineTest(unittest.TestCase):
    # ------------------------------------------------------------------
    # Basic pipeline run
    # ------------------------------------------------------------------
    def test_pipeline_runs_with_input_ids(self):
        pipe = _make_pipeline().to("cpu")

        input_ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
        out = pipe(
            input_ids=input_ids,
            use_chat_template=False,
            max_new_tokens=16,
            block_length=4,
            num_inference_steps=4,
            temperature=0.0,
            mask_token_id=31,
            output_type="seq",
        )

        self.assertIsNotNone(out.sequences)
        self.assertEqual(out.sequences.ndim, 2)
        # Generated tokens only (prompt stripped)
        self.assertGreater(out.sequences.shape[1], 0)

    # ------------------------------------------------------------------
    # output_type="seq" → texts is None
    # ------------------------------------------------------------------
    def test_output_type_seq(self):
        pipe = _make_pipeline().to("cpu")

        out = pipe(
            input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            use_chat_template=False,
            max_new_tokens=16,
            block_length=4,
            num_inference_steps=4,
            temperature=0.0,
            mask_token_id=31,
            output_type="seq",
        )

        self.assertIsNotNone(out.sequences)
        self.assertIsNone(out.texts)

    # ------------------------------------------------------------------
    # output_type="text" with dummy tokenizer
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
        pipe = _make_pipeline(tokenizer=tok).to("cpu")

        out = pipe(
            input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            use_chat_template=False,
            max_new_tokens=16,
            block_length=4,
            num_inference_steps=4,
            temperature=0.0,
            output_type="text",
        )

        self.assertIsNotNone(out.sequences)
        self.assertIsNotNone(out.texts)
        self.assertEqual(len(out.texts), 1)
        self.assertTrue(out.texts[0].startswith("decoded_"))

    # ------------------------------------------------------------------
    # output_type="text" without tokenizer → texts is None
    # ------------------------------------------------------------------
    def test_output_type_text_without_tokenizer(self):
        pipe = _make_pipeline(tokenizer=None).to("cpu")

        out = pipe(
            input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            use_chat_template=False,
            max_new_tokens=16,
            block_length=4,
            num_inference_steps=4,
            temperature=0.0,
            mask_token_id=31,
            output_type="text",
        )

        self.assertIsNotNone(out.sequences)
        self.assertIsNone(out.texts)

    # ------------------------------------------------------------------
    # Invalid output_type raises ValueError
    # ------------------------------------------------------------------
    def test_output_type_invalid_raises(self):
        pipe = _make_pipeline().to("cpu")

        with self.assertRaises(ValueError):
            pipe(
                input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
                use_chat_template=False,
                max_new_tokens=16,
                block_length=4,
                num_inference_steps=4,
                mask_token_id=31,
                output_type="invalid",
            )

    # ------------------------------------------------------------------
    # check_inputs validation
    # ------------------------------------------------------------------
    def test_check_inputs_no_source_raises(self):
        pipe = _make_pipeline()
        with self.assertRaises(ValueError):
            pipe.check_inputs(
                prompt=None,
                messages=None,
                input_ids=None,
                block_length=4,
                num_inference_steps=4,
                mask_token_id=31,
                output_type="seq",
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
            )

    def test_check_inputs_prompt_and_messages_raises(self):
        pipe = _make_pipeline()
        with self.assertRaises(ValueError):
            pipe.check_inputs(
                prompt="hello",
                messages=[{"role": "user", "content": "hi"}],
                input_ids=None,
                block_length=4,
                num_inference_steps=4,
                mask_token_id=31,
                output_type="seq",
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
            )

    def test_check_inputs_bad_input_ids_ndim_raises(self):
        pipe = _make_pipeline()
        bad_ids = torch.zeros(2, 3, 4, dtype=torch.long)
        with self.assertRaises(ValueError):
            pipe.check_inputs(
                prompt=None,
                messages=None,
                input_ids=bad_ids,
                block_length=4,
                num_inference_steps=4,
                mask_token_id=31,
                output_type="seq",
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
            )

    def test_check_inputs_bad_block_length_raises(self):
        pipe = _make_pipeline()
        with self.assertRaises(ValueError):
            pipe.check_inputs(
                prompt=None,
                messages=None,
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                block_length=0,
                num_inference_steps=4,
                mask_token_id=31,
                output_type="seq",
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
            )

    def test_check_inputs_no_mask_token_raises(self):
        pipe = _make_pipeline()
        with self.assertRaises(ValueError):
            pipe.check_inputs(
                prompt=None,
                messages=None,
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                block_length=4,
                num_inference_steps=4,
                mask_token_id=None,
                output_type="seq",
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
            )

    # ------------------------------------------------------------------
    # return_dict=False returns tuple
    # ------------------------------------------------------------------
    def test_return_dict_false(self):
        pipe = _make_pipeline().to("cpu")

        result = pipe(
            input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            use_chat_template=False,
            max_new_tokens=16,
            block_length=4,
            num_inference_steps=4,
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


if __name__ == "__main__":
    unittest.main()
