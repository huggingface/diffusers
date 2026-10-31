import pytest
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


class _PeakedModel(_DummyCausalLM):
    """Predicts one token everywhere with probability ~1, so any confidence threshold fires."""

    def __init__(self, vocab_size: int, token: int):
        super().__init__(vocab_size)
        self.token = int(token)
        self.calls = 0

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros((batch_size, seq_len, self.vocab_size), device=input_ids.device)
        logits[:, :, self.token] = 20.0
        self.calls += 1
        return _DummyModelOutput(logits=logits)


class _AlternatingModel(_DummyCausalLM):
    """Flips its confident prediction on every forward, so the editing phase never settles."""

    def __init__(self, vocab_size: int, tokens=(5, 6)):
        super().__init__(vocab_size)
        self.tokens = tokens
        self.calls = 0

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros((batch_size, seq_len, self.vocab_size), device=input_ids.device)
        logits[:, :, self.tokens[self.calls % len(self.tokens)]] = 20.0
        self.calls += 1
        return _DummyModelOutput(logits=logits)


def _make_pipeline(tokenizer=None, **scheduler_kwargs):
    model = _DummyCausalLM(vocab_size=32)
    # A confidence threshold above 1.0 can never be met, which forces the pure top-k commit quota.
    scheduler = BlockRefinementScheduler(**{"threshold": 2.0, **scheduler_kwargs})
    return LLaDA2Pipeline(model=model, scheduler=scheduler, tokenizer=tokenizer)


class TestLLaDA2Pipeline:
    def test_pipeline_runs(self):
        pipe = _make_pipeline().to("cpu")

        input_ids = torch.tensor([[5, 6, 7, 8], [1, 2, 3, 4]], dtype=torch.long)
        out = pipe(
            input_ids=input_ids,
            use_chat_template=False,
            gen_length=24,
            block_length=8,
            num_inference_steps=8,
            eos_early_stop=False,
            mask_token_id=31,
            eos_token_id=None,
            output_type="seq",
        )

        assert out.sequences.shape == (2, 24)
        assert not (out.sequences == 31).any().item()

    def test_pipeline_return_tuple(self):
        pipe = _make_pipeline().to("cpu")

        input_ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
        sequences, texts = pipe(
            input_ids=input_ids,
            use_chat_template=False,
            gen_length=16,
            block_length=8,
            num_inference_steps=4,
            eos_early_stop=False,
            mask_token_id=31,
            output_type="seq",
            return_dict=False,
        )

        assert sequences.shape == (1, 16)
        assert texts is None

    def test_output_type_seq(self):
        """output_type='seq' should return sequences but no texts."""
        pipe = _make_pipeline().to("cpu")

        out = pipe(
            input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            use_chat_template=False,
            gen_length=16,
            block_length=8,
            num_inference_steps=4,
            eos_early_stop=False,
            mask_token_id=31,
            output_type="seq",
        )

        assert out.sequences is not None
        assert out.sequences.shape == (1, 16)
        assert out.texts is None

    def test_output_type_text_without_tokenizer(self):
        """output_type='text' without a tokenizer should return texts=None."""
        pipe = _make_pipeline(tokenizer=None).to("cpu")

        out = pipe(
            input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            use_chat_template=False,
            gen_length=16,
            block_length=8,
            num_inference_steps=4,
            eos_early_stop=False,
            mask_token_id=31,
            output_type="text",
        )

        assert out.sequences is not None
        assert out.texts is None

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
            eos_early_stop=False,
            output_type="text",
        )

        assert out.sequences is not None
        assert out.texts is not None
        assert len(out.texts) == 1
        assert out.texts[0].startswith("decoded_")

    def test_output_type_invalid_raises(self):
        """Invalid output_type should raise ValueError."""
        pipe = _make_pipeline().to("cpu")

        with pytest.raises(ValueError):
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
        assert torch.equal(result_ids, ids)
        assert result_mask.shape == ids.shape
        assert (result_mask == 1).all().item()

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
        assert result_ids.shape == (1, 3)
        assert result_mask.shape == (1, 3)

    def test_prepare_input_ids_no_tokenizer_raises(self):
        pipe = _make_pipeline(tokenizer=None)
        with pytest.raises(ValueError):
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
        with pytest.raises(ValueError):
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
        with pytest.raises(ValueError):
            pipe._prepare_input_ids(
                prompt=None,
                messages=None,
                input_ids=None,
                use_chat_template=False,
                add_generation_prompt=False,
                chat_template_kwargs=None,
            )


class TestLLaDA2Regression:
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
        scheduler = BlockRefinementScheduler(threshold=2.0)
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
            mask_token_id=31,
            eos_token_id=None,
            eos_early_stop=False,
            output_type="seq",
        )

        assert len(captured) > 0
        first_mask = captured[0]
        # Padded prompt positions stay zero in the runtime mask (Issue #1).
        assert first_mask[0, 3].item() == 0
        assert first_mask[1, 1].item() == 0
        assert first_mask[1, 2].item() == 0
        assert first_mask[1, 3].item() == 0
        # Real prompt positions stay one.
        assert first_mask[0, 0].item() == 1
        assert first_mask[1, 0].item() == 1

    def test_block_length_routes_into_commit_quota(self):
        """Issue #2: the per-call `block_length` must drive the scheduler's per-step commit quota."""
        commits: list[int] = []

        def cb(pipe, step, timestep, kwargs):
            commits.append(int(kwargs["committed_mask"].sum()))
            return {}

        pipe = _make_pipeline().to("cpu")
        pipe(
            input_ids=torch.empty((1, 0), dtype=torch.long),
            use_chat_template=False,
            gen_length=8,
            block_length=8,
            num_inference_steps=8,
            mask_token_id=31,
            eos_token_id=None,
            eos_early_stop=False,
            output_type="seq",
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=["committed_mask"],
        )
        # With block_length=num_inference_steps=8 the schedule commits exactly one token per step.
        assert commits[0] == 1
        assert commits[1] == 1
        assert commits[2] == 1

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
            mask_token_id=31,
            eos_token_id=None,
            eos_early_stop=False,
            output_type="seq",
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=keys,
        )
        assert set(observed) == set(keys)

    def test_eos_at_first_generated_position_triggers_finished(self):
        """Issue #4: EOS exactly at index `prompt_length` must mark the row finished."""
        cur_x = torch.tensor([[10, 2, 99]])
        pred_original_sample = torch.tensor([[0, 2]])
        final_transfer = torch.tensor([[False, True]])
        finished = LLaDA2Pipeline._update_finished(
            cur_x=cur_x,
            pred_original_sample=pred_original_sample,
            final_transfer=final_transfer,
            finished=torch.tensor([False]),
            eos_token_id=2,
            mask_token_id=99,
            prompt_length=1,
        )
        assert bool(finished[0].item())

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
        pipe = LLaDA2Pipeline(model=model, scheduler=BlockRefinementScheduler(threshold=2.0)).to("cpu")
        out = pipe(
            input_ids=torch.tensor([[10], [20]], dtype=torch.long),
            use_chat_template=False,
            gen_length=5,
            block_length=3,
            num_inference_steps=3,
            mask_token_id=31,
            eos_token_id=2,
            eos_early_stop=True,
            output_type="seq",
        )
        # Row 0's first generated tokens must not be overwritten by later-block sampling (token 7).
        assert 7 not in out.sequences[0].tolist()[:2]

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
            mask_token_id=31,
            eos_token_id=None,
            eos_early_stop=False,
            output_type="seq",
        )
        assert pipe._progress_bar_config == before


class TestLLaDA2RefinementLoop:
    """The two-loop denoising structure: a `for` over `scheduler.timesteps`, then an explicit editing loop."""

    _COMMON = {
        "use_chat_template": False,
        "mask_token_id": 31,
        "eos_token_id": None,
        "eos_early_stop": False,
        "output_type": "seq",
    }

    def test_schedule_is_reset_for_every_block(self):
        """`step_index` auto-advances, so each block must start from a fresh schedule."""
        pipe = _make_pipeline().to("cpu")
        num_inference_steps = 8

        out = pipe(
            input_ids=torch.empty((1, 0), dtype=torch.long),
            gen_length=16,
            block_length=8,
            num_inference_steps=num_inference_steps,
            **self._COMMON,
        )

        # Two blocks of 8 positions at one commit per step: 8 steps each. The counter left on the scheduler
        # is therefore the *second* block's, so a schedule that leaked across blocks would read 16 here.
        assert pipe.scheduler.step_index == num_inference_steps
        assert len(pipe.scheduler.timesteps) == num_inference_steps
        assert not (out.sequences == 31).any().item()

    def test_pipeline_steps_its_own_scheduler(self):
        """
        The loop must step `pipe.scheduler` itself.

        Honouring the deprecated per-call knobs by running against a reconfigured *copy* is invisible to
        the output but breaks everything keyed on identity — callbacks, `step_index` inspection, and any
        wrapper a user has attached. The registered scheduler's own state is the evidence: it only carries
        this call's schedule if it is the object that ran.
        """
        pipe = _make_pipeline().to("cpu")
        num_inference_steps = 4
        assert len(pipe.scheduler.timesteps) != num_inference_steps  # the config default, untouched so far

        with pytest.warns(FutureWarning):
            pipe(
                input_ids=torch.empty((1, 0), dtype=torch.long),
                gen_length=8,
                block_length=8,
                num_inference_steps=num_inference_steps,
                threshold=2.0,
                **self._COMMON,
            )

        assert len(pipe.scheduler.timesteps) == num_inference_steps
        assert pipe.scheduler.step_index == num_inference_steps

    def test_editing_phase_runs_after_the_masks_are_gone(self):
        """Once every position is resolved there is no unmasking left to schedule."""
        model = _PeakedModel(vocab_size=32, token=9)
        pipe = LLaDA2Pipeline(
            model=model,
            scheduler=BlockRefinementScheduler(threshold=0.5, editing_threshold=0.5),
        ).to("cpu")

        out = pipe(
            input_ids=torch.empty((1, 0), dtype=torch.long),
            gen_length=8,
            block_length=8,
            num_inference_steps=8,
            **self._COMMON,
        )

        # One confident step fills the whole block; one editing sweep then finds nothing to change.
        assert model.calls == 2
        # `step_edit` consumes no schedule entry, which is what lets the phase outlast the schedule.
        assert pipe.scheduler.step_index == 1
        assert (out.sequences == 9).all().item()

    def test_editing_phase_is_capped_by_max_post_steps(self):
        """A model that never settles must still be bounded, and never consumes a schedule entry."""
        model = _AlternatingModel(vocab_size=32)
        pipe = LLaDA2Pipeline(
            model=model,
            scheduler=BlockRefinementScheduler(threshold=0.5, editing_threshold=0.5),
        ).to("cpu")
        max_post_steps = 3

        pipe(
            input_ids=torch.empty((1, 0), dtype=torch.long),
            gen_length=8,
            block_length=8,
            num_inference_steps=8,
            max_post_steps=max_post_steps,
            **self._COMMON,
        )

        # One mask-filling step, then `max_post_steps + 1` editing sweeps -- the iteration count the
        # released `while` loop reached, where the budget was checked after the sweep rather than before.
        assert model.calls == 1 + (max_post_steps + 1)
        assert pipe.scheduler.step_index == 1

    def test_editing_does_not_overwrite_prompt_positions(self):
        """`prompt_mask` left the scheduler: the pipeline restores frozen positions itself."""
        seen: list[torch.Tensor] = []

        class _Capturing(_PeakedModel):
            def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
                seen.append(input_ids.detach().clone())
                return super().forward(input_ids, attention_mask=attention_mask, position_ids=position_ids)

        prompt = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
        pipe = LLaDA2Pipeline(
            model=_Capturing(vocab_size=32, token=9),
            scheduler=BlockRefinementScheduler(threshold=0.5, editing_threshold=0.5),
        ).to("cpu")

        pipe(
            input_ids=prompt,
            gen_length=8,
            block_length=8,
            num_inference_steps=8,
            **self._COMMON,
        )

        # The prompt shares block 0 with generated positions, and the model would happily rewrite it to 9.
        assert len(seen) > 1
        for input_ids in seen:
            assert torch.equal(input_ids[:, :4], prompt)


class TestLLaDA2Deprecations:
    """The released per-call knobs keep working for one release, scoped to the call."""

    _COMMON = {
        "use_chat_template": False,
        "input_ids": torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
        "gen_length": 8,
        "block_length": 8,
        "num_inference_steps": 4,
        "mask_token_id": 31,
        "eos_token_id": None,
        "eos_early_stop": False,
        "output_type": "seq",
    }

    def test_sampling_arguments_are_honoured_and_restored(self):
        pipe = _make_pipeline(threshold=0.95).to("cpu")
        scheduler = pipe.scheduler

        with pytest.warns(FutureWarning, match="threshold"):
            pipe(threshold=2.0, **self._COMMON)

        # Honoured for this call only: the user's scheduler is the same object, with the same config.
        assert pipe.scheduler is scheduler
        assert pipe.scheduler.config.threshold == 0.95
        assert pipe.scheduler.config.mask_token_id is None

    def test_minimal_topk_warns(self):
        pipe = _make_pipeline().to("cpu")
        with pytest.warns(FutureWarning, match="minimal_topk"):
            pipe(minimal_topk=1, **self._COMMON)

    def test_callback_tensor_input_aliases_resolve(self):
        pipe = _make_pipeline().to("cpu")
        observed: list[dict] = []

        def cb(pipe, step, timestep, kwargs):
            observed.append(kwargs)
            return {}

        with pytest.warns(FutureWarning, match="transfer_index"):
            pipe(
                callback_on_step_end=cb,
                callback_on_step_end_tensor_inputs=["transfer_index", "committed_mask", "sampled_tokens"],
                **self._COMMON,
            )

        first = observed[0]
        assert torch.equal(first["transfer_index"], first["committed_mask"])
        assert first["sampled_tokens"].shape == first["committed_mask"].shape

    def test_mask_token_id_can_come_from_the_scheduler_config(self):
        pipe = _make_pipeline(mask_token_id=31).to("cpu")
        call_kwargs = {k: v for k, v in self._COMMON.items() if k != "mask_token_id"}
        out = pipe(**call_kwargs)
        assert not (out.sequences == 31).any().item()
