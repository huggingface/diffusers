import unittest

import torch

from diffusers import BlockRefinementPipeline


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


class _DummyCausalLM2DOnly(_DummyCausalLM):
    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("2D attention_mask required")
        return super().forward(input_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs)


class BlockRefinementPipelineTest(unittest.TestCase):
    def test_pipeline_runs(self):
        vocab_size = 32
        model = _DummyCausalLM(vocab_size=vocab_size)
        pipe = BlockRefinementPipeline(model=model, tokenizer=None).to("cpu")

        prompt_ids = torch.tensor([[5, 6, 7, 8], [1, 2, 3, 4]], dtype=torch.long)
        out = pipe(
            prompt_ids=prompt_ids,
            gen_length=24,
            block_length=8,
            steps=8,
            temperature=0.0,
            threshold=2.0,  # force top-k commits
            minimal_topk=1,
            eos_early_stop=False,
            mask_token_id=vocab_size - 1,
            eos_token_id=None,
            return_text=False,
        )

        self.assertEqual(out.sequences.shape, (2, 24))
        self.assertFalse((out.sequences == vocab_size - 1).any().item())

    def test_pipeline_falls_back_to_2d_attention_mask(self):
        vocab_size = 32
        model = _DummyCausalLM2DOnly(vocab_size=vocab_size)
        pipe = BlockRefinementPipeline(model=model, tokenizer=None).to("cpu")

        out = pipe(
            prompt_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            gen_length=16,
            block_length=8,
            steps=4,
            temperature=0.0,
            threshold=2.0,
            minimal_topk=1,
            eos_early_stop=False,
            mask_token_id=vocab_size - 1,
            eos_token_id=None,
            attention_mask_mode="auto",
            return_text=False,
        )

        self.assertEqual(out.sequences.shape, (1, 16))


if __name__ == "__main__":
    unittest.main()
