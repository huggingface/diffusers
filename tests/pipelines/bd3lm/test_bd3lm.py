import unittest

import torch

from diffusers import BD3LMPipeline, BD3LMTokenDiffusionScheduler


class _DummyConfig:
    def __init__(self, block_size, vocab_size, mask_index):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.mask_index = mask_index


class _DummyBD3LMModel(torch.nn.Module):
    """Minimal model that satisfies BD3LMPipeline's interface."""

    def __init__(self, vocab_size=32, block_size=4):
        super().__init__()
        self.config = _DummyConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            mask_index=vocab_size - 1,
        )
        self.backbone = torch.nn.Linear(1, vocab_size, bias=False)
        self.register_buffer("_device_anchor", torch.empty(0))

    @property
    def dtype(self):
        return torch.float32

    @property
    def device(self):
        return self._device_anchor.device

    def reset_kv_cache(self, eval_batch_size):
        pass

    def forward(self, input_ids, timesteps, sample_mode=False, store_kv=False):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(
            (batch_size, seq_len, self.config.vocab_size),
            device=input_ids.device,
            dtype=torch.float32,
        )
        # Make logits vary by position so denoising is deterministic.
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.float32).view(1, seq_len, 1)
        token_ids = (torch.arange(seq_len, device=input_ids.device) % (self.config.vocab_size - 2)).view(1, seq_len, 1)
        logits.scatter_(
            2,
            token_ids.expand(batch_size, -1, -1),
            1.0 + positions.expand(batch_size, -1, -1) * 0.1,
        )
        return logits


def _make_pipeline(tokenizer=None, vocab_size=32, block_size=4):
    model = _DummyBD3LMModel(vocab_size=vocab_size, block_size=block_size)
    scheduler = BD3LMTokenDiffusionScheduler(
        block_size=block_size,
        mask_token_id=vocab_size - 1,
    )
    return BD3LMPipeline(model=model, scheduler=scheduler, tokenizer=tokenizer)


class BD3LMPipelineTest(unittest.TestCase):
    def test_pipeline_runs(self):
        """Basic end-to-end generation with input_ids."""
        pipe = _make_pipeline().to("cpu")

        input_ids = torch.tensor([[5, 6, 7, 8], [1, 2, 3, 4]], dtype=torch.long)
        out = pipe(
            input_ids=input_ids,
            gen_length=8,
            num_inference_steps=8,
            nucleus_p=1.0,
            output_type="seq",
        )

        self.assertEqual(out.sequences.shape[0], 2)
        self.assertGreater(out.sequences.shape[1], 0)
        self.assertLessEqual(out.sequences.shape[1], 8)

    def test_output_type_seq(self):
        """output_type='seq' returns sequences but no texts."""
        pipe = _make_pipeline().to("cpu")

        out = pipe(
            input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            gen_length=8,
            num_inference_steps=4,
            nucleus_p=1.0,
            output_type="seq",
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
            gen_length=8,
            num_inference_steps=4,
            nucleus_p=1.0,
            output_type="text",
        )

        self.assertIsNotNone(out.sequences)
        self.assertIsNotNone(out.texts)
        self.assertEqual(len(out.texts), 1)
        self.assertTrue(out.texts[0].startswith("decoded_"))

    def test_output_type_text_without_tokenizer(self):
        """output_type='text' without a tokenizer should return texts=None."""
        pipe = _make_pipeline(tokenizer=None).to("cpu")

        out = pipe(
            input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            gen_length=8,
            num_inference_steps=4,
            nucleus_p=1.0,
            output_type="text",
        )

        self.assertIsNotNone(out.sequences)
        self.assertIsNone(out.texts)

    def test_output_type_invalid_raises(self):
        """Invalid output_type should raise ValueError."""
        pipe = _make_pipeline().to("cpu")

        with self.assertRaises(ValueError):
            pipe(
                input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
                gen_length=8,
                num_inference_steps=4,
                nucleus_p=1.0,
                output_type="invalid",
            )

    def test_return_dict_false(self):
        """return_dict=False should return a plain tuple."""
        pipe = _make_pipeline().to("cpu")

        result = pipe(
            input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            gen_length=8,
            num_inference_steps=4,
            nucleus_p=1.0,
            output_type="seq",
            return_dict=False,
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        sequences, texts = result
        self.assertIsInstance(sequences, torch.Tensor)
        self.assertIsNone(texts)

    def test_check_inputs_bad_gen_length(self):
        """gen_length <= 0 should raise ValueError."""
        pipe = _make_pipeline().to("cpu")

        with self.assertRaises(ValueError):
            pipe(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                gen_length=0,
                num_inference_steps=4,
                nucleus_p=1.0,
                output_type="seq",
            )

    def test_check_inputs_bad_num_inference_steps(self):
        """num_inference_steps <= 0 should raise ValueError."""
        pipe = _make_pipeline().to("cpu")

        with self.assertRaises(ValueError):
            pipe(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                gen_length=8,
                num_inference_steps=0,
                nucleus_p=1.0,
                output_type="seq",
            )

    def test_check_inputs_bad_nucleus_p(self):
        """nucleus_p out of (0, 1] should raise ValueError."""
        pipe = _make_pipeline().to("cpu")

        with self.assertRaises(ValueError):
            pipe(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                gen_length=8,
                num_inference_steps=4,
                nucleus_p=0.0,
                output_type="seq",
            )

    def test_check_inputs_bad_output_type(self):
        """output_type not in {'seq', 'text'} should raise ValueError."""
        pipe = _make_pipeline().to("cpu")

        with self.assertRaises(ValueError):
            pipe(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                gen_length=8,
                num_inference_steps=4,
                nucleus_p=1.0,
                output_type="bad",
            )


if __name__ == "__main__":
    unittest.main()
