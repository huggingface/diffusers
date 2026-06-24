import unittest

import torch

from diffusers import IDLMBlockDiffusionScheduler, IDLMPipeline


class _DummyModelOutput:
    def __init__(self, logits):
        self.logits = logits


class _DummyConfig:
    def __init__(self, mask_token_id: int):
        self.mask_token_id = int(mask_token_id)


class _DummyCausalLM(torch.nn.Module):
    """Minimal causal LM that returns deterministic logits.

    The I-DLM pipeline reads logits under the Dream shift: logits[:, i, :] predicts input position i+1.
    We give each position a unique argmax (via a position-biased one-hot) so the scheduler's verify / sample
    paths exercise real code without depending on a real checkpoint.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.register_buffer("_device_anchor", torch.empty(0))
        self.config = _DummyConfig(mask_token_id=vocab_size - 1)

    @property
    def dtype(self):
        return torch.float32

    @property
    def device(self):
        return self._device_anchor.device

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros((batch_size, seq_len, self.vocab_size), device=input_ids.device, dtype=torch.float32)
        # Position-dependent argmax. Using position_ids if provided so that repeated forwards with the same
        # local seq_len but different absolute positions still differ.
        if position_ids is not None:
            pos = position_ids[0]
        else:
            pos = torch.arange(seq_len, device=input_ids.device)
        argmax_ids = (pos % (self.vocab_size - 2)).long()
        logits.scatter_(2, argmax_ids.view(1, seq_len, 1).expand(batch_size, -1, -1), 5.0)
        return _DummyModelOutput(logits=logits)


def _make_pipeline():
    model = _DummyCausalLM(vocab_size=32)
    scheduler = IDLMBlockDiffusionScheduler(gen_block_size=3, temperature=0.0, top_k=0, top_p=1.0)
    return IDLMPipeline(model=model, scheduler=scheduler)


class IDLMPipelineTest(unittest.TestCase):
    def test_scheduler_block_sizes(self):
        sch = IDLMBlockDiffusionScheduler(gen_block_size=4)
        self.assertEqual(sch.block_size, 7)
        self.assertEqual(sch.num_masks, 3)

    def test_pipeline_runs_with_input_ids(self):
        pipe = _make_pipeline().to("cpu")

        out = pipe(
            input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            use_chat_template=False,
            max_new_tokens=8,
            gen_block_size=3,
            temperature=0.0,
            mask_token_id=31,
            output_type="seq",
        )

        self.assertIsNotNone(out.sequences)
        self.assertEqual(out.sequences.ndim, 2)
        # Generation should produce at least one token even if it stops early.
        self.assertGreater(out.sequences.shape[1], 0)
        self.assertLessEqual(out.sequences.shape[1], 8)

    def test_output_type_seq_has_no_texts(self):
        pipe = _make_pipeline().to("cpu")
        out = pipe(
            input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            use_chat_template=False,
            max_new_tokens=6,
            gen_block_size=3,
            mask_token_id=31,
            output_type="seq",
        )
        self.assertIsNotNone(out.sequences)
        self.assertIsNone(out.texts)

    def test_scheduler_verify_rejects_impossible_spec(self):
        # p assigns zero mass to the spec token, q assigns full mass → always reject.
        sch = IDLMBlockDiffusionScheduler(gen_block_size=3, verify_alpha=1.0)
        vocab = 4
        # p: uniform over tokens {0,1} — prob 0 on token 2
        anchor_logits = torch.full((1, vocab), float("-inf"))
        anchor_logits[0, 0] = 0.0
        anchor_logits[0, 1] = 0.0
        # q: one-hot on token 2
        q = torch.zeros(1, vocab)
        q[0, 2] = 1.0
        accepted, resample = sch.verify_specs(anchor_logits, [2], q)
        self.assertEqual(accepted, 0)
        self.assertIsNotNone(resample)
        # The resample must come from max(0, p - q) → tokens {0, 1} only.
        self.assertIn(int(resample), {0, 1})

    def test_scheduler_verify_always_accepts_matched_spec(self):
        # p == q (one-hot on token 3) → ratio = 1 → always accept.
        sch = IDLMBlockDiffusionScheduler(gen_block_size=3)
        vocab = 6
        anchor_logits = torch.full((1, vocab), float("-inf"))
        anchor_logits[0, 3] = 0.0
        q = torch.zeros(1, vocab)
        q[0, 3] = 1.0
        accepted, resample = sch.verify_specs(anchor_logits, [3], q)
        self.assertEqual(accepted, 1)
        self.assertIsNone(resample)


class IDLMSchedulerTest(unittest.TestCase):
    def test_sample_greedy_returns_argmax(self):
        sch = IDLMBlockDiffusionScheduler(gen_block_size=3, temperature=0.0)
        logits = torch.zeros(3, 10)
        logits[0, 5] = 10.0
        logits[1, 1] = 10.0
        logits[2, 9] = 10.0
        tokens, probs = sch.sample(logits, temperature=0.0)
        self.assertEqual(tokens.tolist(), [5, 1, 9])
        # Greedy probs must be defined (used for verification later).
        self.assertEqual(probs.shape, (3,))

    def test_step_full_accept_produces_new_specs(self):
        sch = IDLMBlockDiffusionScheduler(gen_block_size=3, temperature=0.0)
        vocab = 16
        K = 2
        num_masks = sch.num_masks  # 2
        # Input layout: [pending, spec_0, spec_1, MASK, MASK] — 5 positions, logits[i] predicts i+1.
        L = 1 + K + num_masks  # 5
        logits = torch.full((1, L, vocab), float("-inf"))
        spec_tokens = [7, 8]
        # Each logit's argmax is distinct so we can verify which logit feeds which field.
        logits[0, 0, spec_tokens[0]] = 10.0  # predicts pos 1 → anchor p for spec_0 (matches spec)
        logits[0, 1, spec_tokens[1]] = 10.0  # predicts pos 2 → anchor p for spec_1 (matches spec)
        logits[0, 2, 3] = 10.0  # predicts pos 3 (first MASK) → next_pending for the next round
        logits[0, 3, 4] = 10.0  # predicts pos 4 (second MASK) → new_spec[0]
        logits[0, 4, 5] = 10.0  # predicts pos 5 (beyond input) → new_spec[1]

        # q matches p exactly for both specs → always accept.
        draft_probs = torch.zeros(K, vocab)
        draft_probs[0, spec_tokens[0]] = 1.0
        draft_probs[1, spec_tokens[1]] = 1.0

        step_out = sch.step(
            model_output=logits,
            timestep=0,
            pending=42,
            spec_tokens=spec_tokens,
            spec_draft_probs=draft_probs,
            temperature=0.0,
        )
        self.assertEqual(step_out.accepted_length, K)
        self.assertTrue(step_out.was_full_accept)
        self.assertEqual(step_out.committed_tokens.tolist(), [42, 7, 8])
        # next_pending ← logits[K] predicts first MASK slot.
        self.assertEqual(step_out.next_pending, 3)
        # new_specs ← logits[K+1 : K+1+num_masks] = logits[3:5].
        self.assertEqual(len(step_out.next_specs), num_masks)
        self.assertEqual(step_out.next_specs, [4, 5])


if __name__ == "__main__":
    unittest.main()
