import unittest

import torch

from diffusers import BlockRefinementScheduler, LLaDA2Pipeline


class _DummyModelOutput:
    def __init__(self, logits):
        self.logits = logits


class _DummyCausalLM(torch.nn.Module):
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
    scheduler = BlockRefinementScheduler()
    return LLaDA2Pipeline(model=model, scheduler=scheduler, tokenizer=tokenizer)


class LLaDA2PipelineTest(unittest.TestCase):
    def test_pipeline_runs(self):
        pipe = _make_pipeline().to("cpu")

        input_ids = torch.tensor([[5, 6, 7, 8], [1, 2, 3, 4]], dtype=torch.long)
        out = pipe(
            input_ids=input_ids,
            use_chat_template=False,
            gen_length=24,
            block_length=8,
            num_inference_steps=8,
            temperature=0.0,
            threshold=2.0,  # force top-k commits
            minimal_topk=1,
            eos_early_stop=False,
            mask_token_id=31,
            eos_token_id=None,
            output_type="seq",
        )

        self.assertEqual(out.sequences.shape, (2, 24))
        self.assertFalse((out.sequences == 31).any().item())

    def test_pipeline_return_tuple(self):
        pipe = _make_pipeline().to("cpu")

        input_ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
        sequences, texts = pipe(
            input_ids=input_ids,
            use_chat_template=False,
            gen_length=16,
            block_length=8,
            num_inference_steps=4,
            temperature=0.0,
            threshold=2.0,
            minimal_topk=1,
            eos_early_stop=False,
            mask_token_id=31,
            output_type="seq",
            return_dict=False,
        )

        self.assertEqual(sequences.shape, (1, 16))
        self.assertIsNone(texts)

    def test_output_type_seq(self):
        """output_type='seq' should return sequences but no texts."""
        pipe = _make_pipeline().to("cpu")

        out = pipe(
            input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            use_chat_template=False,
            gen_length=16,
            block_length=8,
            num_inference_steps=4,
            temperature=0.0,
            threshold=2.0,
            minimal_topk=1,
            eos_early_stop=False,
            mask_token_id=31,
            output_type="seq",
        )

        self.assertIsNotNone(out.sequences)
        self.assertEqual(out.sequences.shape, (1, 16))
        self.assertIsNone(out.texts)

    def test_output_type_text_without_tokenizer(self):
        """output_type='text' without a tokenizer should return texts=None."""
        pipe = _make_pipeline(tokenizer=None).to("cpu")

        out = pipe(
            input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            use_chat_template=False,
            gen_length=16,
            block_length=8,
            num_inference_steps=4,
            temperature=0.0,
            threshold=2.0,
            minimal_topk=1,
            eos_early_stop=False,
            mask_token_id=31,
            output_type="text",
        )

        self.assertIsNotNone(out.sequences)
        self.assertIsNone(out.texts)

    def test_output_type_text_with_tokenizer(self):
        """output_type='text' with a tokenizer should return decoded texts."""
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
            gen_length=16,
            block_length=8,
            num_inference_steps=4,
            temperature=0.0,
            threshold=2.0,
            minimal_topk=1,
            eos_early_stop=False,
            output_type="text",
        )

        self.assertIsNotNone(out.sequences)
        self.assertIsNotNone(out.texts)
        self.assertEqual(len(out.texts), 1)
        self.assertTrue(out.texts[0].startswith("decoded_"))

    def test_output_type_invalid_raises(self):
        """Invalid output_type should raise ValueError."""
        pipe = _make_pipeline().to("cpu")

        with self.assertRaises(ValueError):
            pipe(
                input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
                use_chat_template=False,
                gen_length=16,
                block_length=8,
                num_inference_steps=4,
                mask_token_id=31,
                output_type="invalid",
            )

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

    def test_prepare_input_ids_no_tokenizer_raises(self):
        pipe = _make_pipeline(tokenizer=None)
        with self.assertRaises(ValueError):
            pipe._prepare_input_ids(
                prompt="hello",
                messages=None,
                input_ids=None,
                use_chat_template=False,
                add_generation_prompt=False,
                chat_template_kwargs=None,
            )

    def test_prepare_input_ids_both_prompt_and_messages_raises(self):
        pipe = _make_pipeline()
        # Manually set tokenizer to a simple object so _prepare_input_ids doesn't short-circuit
        pipe.tokenizer = type("Tok", (), {"eos_token_id": None, "mask_token_id": None})()
        with self.assertRaises(ValueError):
            pipe._prepare_input_ids(
                prompt="hello",
                messages=[{"role": "user", "content": "hi"}],
                input_ids=None,
                use_chat_template=False,
                add_generation_prompt=False,
                chat_template_kwargs=None,
            )

    def test_prepare_input_ids_neither_raises(self):
        pipe = _make_pipeline()
        pipe.tokenizer = type("Tok", (), {"eos_token_id": None, "mask_token_id": None})()
        with self.assertRaises(ValueError):
            pipe._prepare_input_ids(
                prompt=None,
                messages=None,
                input_ids=None,
                use_chat_template=False,
                add_generation_prompt=False,
                chat_template_kwargs=None,
            )


if __name__ == "__main__":
    unittest.main()
