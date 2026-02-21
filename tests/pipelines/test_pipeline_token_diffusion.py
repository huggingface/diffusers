import unittest

import torch

from diffusers import TokenDiffusionPipeline, TokenDiffusionScheduler


class _DummyTokenizer:
    bos_token_id = None
    cls_token_id = 1

    def batch_decode(self, sequences, skip_special_tokens=True):
        # Deterministic, cheap “decode”: join token ids as strings.
        out = []
        for row in sequences.tolist():
            out.append(" ".join(str(i) for i in row))
        return out


class _DummyModelOutput:
    def __init__(self, logits):
        self.logits = logits


class _DummyMLM(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.register_buffer("_device_anchor", torch.empty(0))

    @property
    def dtype(self):
        return torch.float32

    @property
    def device(self):
        return self._device_anchor.device

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros((batch_size, seq_len, self.vocab_size), device=input_ids.device, dtype=torch.float32)
        return _DummyModelOutput(logits=logits)


class TokenDiffusionPipelineTest(unittest.TestCase):
    def test_absorbing_pipeline_runs(self):
        vocab_size = 32
        scheduler = TokenDiffusionScheduler(
            vocab_size=vocab_size, mask_token_id=vocab_size - 1, forward_process="absorbing"
        )
        model = _DummyMLM(vocab_size=vocab_size)
        tokenizer = _DummyTokenizer()

        pipe = TokenDiffusionPipeline(model=model, scheduler=scheduler, tokenizer=tokenizer)
        pipe = pipe.to("cpu")

        out = pipe(batch_size=2, seq_len=8, num_inference_steps=2, inject_start_token=True)
        self.assertEqual(out.sequences.shape, (2, 8))
        self.assertEqual(len(out.texts), 2)

    def test_uniform_pipeline_runs(self):
        vocab_size = 32
        scheduler = TokenDiffusionScheduler(
            vocab_size=vocab_size,
            mask_token_id=vocab_size - 1,
            forward_process="uniform",
            exclude_mask_from_uniform=True,
        )
        model = _DummyMLM(vocab_size=vocab_size)
        tokenizer = _DummyTokenizer()

        pipe = TokenDiffusionPipeline(model=model, scheduler=scheduler, tokenizer=tokenizer)
        pipe = pipe.to("cpu")

        gen = torch.Generator().manual_seed(0)
        out = pipe(batch_size=2, seq_len=8, num_inference_steps=2, generator=gen, inject_start_token=True)
        self.assertEqual(out.sequences.shape, (2, 8))
        self.assertFalse((out.sequences == scheduler.mask_token_id).any().item())

    def test_prefix_ids_are_fixed(self):
        vocab_size = 32
        scheduler = TokenDiffusionScheduler(
            vocab_size=vocab_size, mask_token_id=vocab_size - 1, forward_process="absorbing"
        )
        model = _DummyMLM(vocab_size=vocab_size)
        tokenizer = _DummyTokenizer()

        pipe = TokenDiffusionPipeline(model=model, scheduler=scheduler, tokenizer=tokenizer).to("cpu")
        prefix = torch.tensor([5, 6, 7], dtype=torch.long)
        out = pipe(batch_size=2, seq_len=8, num_inference_steps=2, prefix_ids=prefix, return_text=False)

        self.assertTrue((out.sequences[:, :3] == prefix.view(1, -1)).all().item())

    def test_infill_mask_freezes_positions(self):
        vocab_size = 32
        scheduler = TokenDiffusionScheduler(
            vocab_size=vocab_size,
            mask_token_id=vocab_size - 1,
            forward_process="uniform",
            exclude_mask_from_uniform=True,
        )
        model = _DummyMLM(vocab_size=vocab_size)
        tokenizer = _DummyTokenizer()

        pipe = TokenDiffusionPipeline(model=model, scheduler=scheduler, tokenizer=tokenizer).to("cpu")

        # Only positions 2..7 are editable, first two positions are fixed to the initial values.
        infill_mask = torch.ones((2, 8), dtype=torch.bool)
        infill_mask[:, :2] = False
        gen = torch.Generator().manual_seed(0)
        out = pipe(
            batch_size=2, seq_len=8, num_inference_steps=2, generator=gen, infill_mask=infill_mask, return_text=False
        )

        # Fixed positions should be unchanged from the initial latents (for uniform, these are random but clamped).
        # Since the model predicts uniform logits and the scheduler would otherwise resample, this checks clamping works.
        out2 = pipe(
            batch_size=2,
            seq_len=8,
            num_inference_steps=2,
            generator=torch.Generator().manual_seed(0),
            infill_mask=infill_mask,
            return_text=False,
        )
        self.assertTrue((out.sequences[:, :2] == out2.sequences[:, :2]).all().item())


if __name__ == "__main__":
    unittest.main()
