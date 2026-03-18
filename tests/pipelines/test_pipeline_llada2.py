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
            steps=8,
            temperature=0.0,
            threshold=2.0,  # force top-k commits
            minimal_topk=1,
            eos_early_stop=False,
            mask_token_id=31,
            eos_token_id=None,
            return_text=False,
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
            steps=4,
            temperature=0.0,
            threshold=2.0,
            minimal_topk=1,
            eos_early_stop=False,
            mask_token_id=31,
            return_text=False,
            return_dict=False,
        )

        self.assertEqual(sequences.shape, (1, 16))
        self.assertIsNone(texts)

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


class TestTopPFiltering(unittest.TestCase):
    def test_top_p_filtering(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        filtered = LLaDA2Pipeline._top_p_filtering(logits, top_p=0.5)
        self.assertTrue((filtered > torch.finfo(filtered.dtype).min).any())
        self.assertTrue((filtered == torch.finfo(filtered.dtype).min).any())

    def test_top_p_filtering_none(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = LLaDA2Pipeline._top_p_filtering(logits, top_p=None)
        self.assertTrue(torch.equal(result, logits))

    def test_top_p_filtering_one(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = LLaDA2Pipeline._top_p_filtering(logits, top_p=1.0)
        self.assertTrue(torch.equal(result, logits))


class TestTopKFiltering(unittest.TestCase):
    def test_top_k_filtering(self):
        logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
        filtered = LLaDA2Pipeline._top_k_filtering(logits, top_k=2)
        self.assertAlmostEqual(filtered[0, 1].item(), 4.0)
        self.assertAlmostEqual(filtered[0, 3].item(), 3.0)
        self.assertEqual(filtered[0, 0].item(), torch.finfo(filtered.dtype).min)
        self.assertEqual(filtered[0, 2].item(), torch.finfo(filtered.dtype).min)

    def test_top_k_filtering_none(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = LLaDA2Pipeline._top_k_filtering(logits, top_k=None)
        self.assertTrue(torch.equal(result, logits))

    def test_top_k_filtering_zero(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = LLaDA2Pipeline._top_k_filtering(logits, top_k=0)
        self.assertTrue(torch.equal(result, logits))

    def test_top_k_filtering_large_k(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = LLaDA2Pipeline._top_k_filtering(logits, top_k=100)
        self.assertTrue(torch.equal(result, logits))


class TestSampleWithTemperature(unittest.TestCase):
    def test_greedy_sampling(self):
        logits = torch.tensor([[1.0, 5.0, 2.0]])
        tokens, probs = LLaDA2Pipeline._sample_with_temperature_topk_topp(
            logits,
            temperature=0.0,
            top_k=None,
            top_p=None,
            generator=None,
            use_multinomial=False,
        )
        self.assertEqual(tokens.item(), 1)
        self.assertEqual(tokens.shape, (1,))
        self.assertEqual(probs.shape, (1,))

    def test_multinomial_sampling(self):
        logits = torch.tensor([[0.0, 100.0, -100.0]])
        gen = torch.Generator().manual_seed(42)
        tokens, probs = LLaDA2Pipeline._sample_with_temperature_topk_topp(
            logits,
            temperature=1.0,
            top_k=None,
            top_p=None,
            generator=gen,
            use_multinomial=True,
        )
        self.assertEqual(tokens.item(), 1)

    def test_temperature_scaling(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        tokens, _ = LLaDA2Pipeline._sample_with_temperature_topk_topp(
            logits,
            temperature=0.01,
            top_k=None,
            top_p=None,
            generator=None,
            use_multinomial=False,
        )
        self.assertEqual(tokens.item(), 2)

    def test_negative_temperature_raises(self):
        logits = torch.tensor([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            LLaDA2Pipeline._sample_with_temperature_topk_topp(
                logits,
                temperature=-1.0,
                top_k=None,
                top_p=None,
                generator=None,
                use_multinomial=False,
            )


if __name__ == "__main__":
    unittest.main()
