import unittest

import torch

from diffusers import BlockTokenDiffusionPipeline, BlockTokenDiffusionScheduler


class _DummyTokenizer:
    cls_token_id = 1
    bos_token_id = None

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [" ".join(map(str, row)) for row in sequences.tolist()]


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


class BlockTokenDiffusionPipelineTest(unittest.TestCase):
    def test_pipeline_runs_and_respects_prefix(self):
        vocab_size = 32
        scheduler = BlockTokenDiffusionScheduler(vocab_size=vocab_size, mask_token_id=vocab_size - 1)
        model = _DummyMLM(vocab_size=vocab_size)
        tokenizer = _DummyTokenizer()
        pipe = BlockTokenDiffusionPipeline(model=model, scheduler=scheduler, tokenizer=tokenizer).to("cpu")

        prefix = torch.tensor([5, 6, 7], dtype=torch.long)
        gen = torch.Generator().manual_seed(0)
        out = pipe(batch_size=2, seq_len=10, block_size=4, num_inference_steps=2, generator=gen, prefix_ids=prefix)

        self.assertEqual(out.sequences.shape, (2, 10))
        self.assertTrue((out.sequences[:, :3] == prefix.view(1, -1)).all().item())
        self.assertEqual(len(out.texts), 2)


if __name__ == "__main__":
    unittest.main()
