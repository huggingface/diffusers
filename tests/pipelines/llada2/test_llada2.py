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
        result_ids, result_mask = pipe._prepare_input_ids(
            prompt=None,
            messages=None,
            input_ids=ids,
            use_chat_template=False,
            add_generation_prompt=False,
            chat_template_kwargs=None,
        )
        self.assertTrue(torch.equal(result_ids, ids))
        self.assertEqual(result_mask.shape, ids.shape)
        self.assertTrue((result_mask == 1).all().item())

    def test_prepare_input_ids_from_1d_tensor(self):
        pipe = _make_pipeline()
        ids = torch.tensor([1, 2, 3], dtype=torch.long)
        result_ids, result_mask = pipe._prepare_input_ids(
            prompt=None,
            messages=None,
            input_ids=ids,
            use_chat_template=False,
            add_generation_prompt=False,
            chat_template_kwargs=None,
        )
        self.assertEqual(result_ids.shape, (1, 3))
        self.assertEqual(result_mask.shape, (1, 3))

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


class LLaDA2RegressionTest(unittest.TestCase):
    """Pin the regressions identified in https://github.com/huggingface/diffusers/issues/13598."""

    def test_attention_mask_carried_through_for_pre_tokenized_input(self):
        """Issue #1: explicit `attention_mask` must reach the model and zero out padded prompt
        positions and the block-aligned tail past `prompt_length + gen_length`."""
        captured: list[torch.Tensor] = []

        class _MaskCapturingModel(_DummyCausalLM):
            def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
                captured.append(attention_mask.detach().cpu().clone() if attention_mask is not None else None)
                return super().forward(input_ids, attention_mask=attention_mask, position_ids=position_ids)

        model = _MaskCapturingModel(vocab_size=32)
        scheduler = BlockRefinementScheduler()
        pipe = LLaDA2Pipeline(model=model, scheduler=scheduler).to("cpu")

        input_ids = torch.tensor([[10, 11, 12, 0], [20, 0, 0, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.long)

        pipe(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_chat_template=False,
            gen_length=4,
            block_length=4,
            num_inference_steps=2,
            threshold=2.0,
            mask_token_id=31,
            eos_token_id=None,
            eos_early_stop=False,
            output_type="seq",
        )

        self.assertGreater(len(captured), 0)
        first_mask = captured[0]
        # Padded prompt positions stay zero in the runtime mask (Issue #1).
        self.assertEqual(first_mask[0, 3].item(), 0)
        self.assertEqual(first_mask[1, 1].item(), 0)
        self.assertEqual(first_mask[1, 2].item(), 0)
        self.assertEqual(first_mask[1, 3].item(), 0)
        # Real prompt positions stay one.
        self.assertEqual(first_mask[0, 0].item(), 1)
        self.assertEqual(first_mask[1, 0].item(), 1)

    def test_block_length_routes_into_scheduler_transfer_schedule(self):
        """Issue #2: the per-call `block_length` must drive the scheduler's `_transfer_schedule`."""
        commits: list[int] = []

        def cb(pipe, step, timestep, kwargs):
            commits.append(int(kwargs["transfer_index"].sum()))
            return {}

        pipe = _make_pipeline().to("cpu")
        pipe(
            input_ids=torch.empty((1, 0), dtype=torch.long),
            use_chat_template=False,
            gen_length=8,
            block_length=8,
            num_inference_steps=8,
            threshold=2.0,
            mask_token_id=31,
            eos_token_id=None,
            eos_early_stop=False,
            output_type="seq",
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=["transfer_index"],
        )
        # With block_length=num_inference_steps=8 the schedule commits exactly one token per step.
        self.assertEqual(commits[0], 1)
        self.assertEqual(commits[1], 1)
        self.assertEqual(commits[2], 1)

    def test_callback_tensor_inputs_advertised_keys_resolve(self):
        """Issue #3: every advertised callback key must be a bound local at callback time."""
        observed: list[str] = []

        def cb(pipe, step, timestep, kwargs):
            observed.extend(sorted(kwargs.keys()))
            return {}

        pipe = _make_pipeline().to("cpu")
        keys = list(pipe._callback_tensor_inputs)
        pipe(
            input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            use_chat_template=False,
            gen_length=8,
            block_length=8,
            num_inference_steps=4,
            threshold=2.0,
            mask_token_id=31,
            eos_token_id=None,
            eos_early_stop=False,
            output_type="seq",
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=keys,
        )
        self.assertEqual(set(observed), set(keys))

    def test_eos_at_first_generated_position_triggers_finished(self):
        """Issue #4: EOS exactly at index `prompt_length` must mark the row finished."""
        cur_x = torch.tensor([[10, 2, 99]])
        sampled_tokens = torch.tensor([[0, 2]])
        final_transfer = torch.tensor([[False, True]])
        finished = BlockRefinementScheduler.check_eos_finished(
            cur_x=cur_x,
            sampled_tokens=sampled_tokens,
            final_transfer=final_transfer,
            finished=torch.tensor([False]),
            eos_token_id=2,
            mask_token_id=99,
            prompt_length=1,
        )
        self.assertTrue(bool(finished[0].item()))

    def test_finished_rows_are_frozen_for_subsequent_blocks(self):
        """Issue #5: once a row emits EOS, later blocks must not overwrite its committed tokens."""

        class _EosThenJunkModel(_DummyCausalLM):
            """Row 0 commits EOS in the first block, then later blocks would emit token 7. Row 1 keeps emitting token 6."""

            def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
                batch_size, seq_len = input_ids.shape
                logits = torch.zeros((batch_size, seq_len, self.vocab_size), device=input_ids.device)
                # First block (seq_len <= 3): row 0 emits 5 then EOS=2; row 1 emits 6.
                if seq_len <= 3:
                    logits[0, :, 5] = 10
                    logits[0, 2, 2] = 20  # strong EOS at last block position
                    logits[1, :, 6] = 10
                else:
                    logits[0, :, 7] = 10  # would overwrite row 0's prior tokens if not frozen
                    logits[1, :, 6] = 10
                return _DummyModelOutput(logits=logits)

        model = _EosThenJunkModel(vocab_size=32)
        pipe = LLaDA2Pipeline(model=model, scheduler=BlockRefinementScheduler()).to("cpu")
        out = pipe(
            input_ids=torch.tensor([[10], [20]], dtype=torch.long),
            use_chat_template=False,
            gen_length=5,
            block_length=3,
            num_inference_steps=3,
            threshold=2.0,
            mask_token_id=31,
            eos_token_id=2,
            eos_early_stop=True,
            output_type="seq",
        )
        # Row 0's first generated tokens must not be overwritten by later-block sampling (token 7).
        self.assertNotIn(7, out.sequences[0].tolist()[:2])

    def test_progress_bar_disable_is_preserved_after_call(self):
        """Issue #6: calling the pipeline must not mutate `_progress_bar_config`."""
        pipe = _make_pipeline().to("cpu")
        pipe.set_progress_bar_config(disable=True)
        before = dict(pipe._progress_bar_config)
        pipe(
            input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            use_chat_template=False,
            gen_length=8,
            block_length=8,
            num_inference_steps=2,
            threshold=2.0,
            mask_token_id=31,
            eos_token_id=None,
            eos_early_stop=False,
            output_type="seq",
        )
        self.assertEqual(pipe._progress_bar_config, before)


if __name__ == "__main__":
    unittest.main()
