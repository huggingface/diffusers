# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import pytest
import torch

from diffusers import (
    RFD4ProteinaAutoencoder,
    RFD4ProteinaPipeline,
    RFD4ProteinaScheduler,
    RFD4ProteinaTransformer,
)
from diffusers.models.transformers.transformer_rfd4_proteina import (
    CONDITION_KEYS,
    RFD4ProteinaTransformerOutput,
    prepare_coord_group_seq,
)

from ...testing_utils import torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


SEQUENCE_KEY = CONDITION_KEYS["sequence"]


def prepared_inputs(device="cpu"):
    padding = torch.tensor([[True, True, False]], device=device)
    return {
        "padding_mask": padding,
        "is_in_context_conditioning": torch.tensor([[False, True, False]], device=device),
        "supports_local_latent": padding,
        "atom37_mask": torch.ones(1, 3, 37, dtype=torch.bool, device=device),
        "conditioning_features": {
            SEQUENCE_KEY: torch.zeros(1, 3, dtype=torch.long, device=device),
            "molecule_iid": torch.zeros(1, 3, dtype=torch.long, device=device),
            "chain_iid": torch.zeros(1, 3, dtype=torch.long, device=device),
        },
    }


def allowed_residue_types():
    """The 33-class protein and nucleic encoding: each token type may decode to its own classes, never the mask token."""
    allowed = [[False] * 33 for _ in range(5)]
    for index in range(1, 22):  # protein: twenty amino acids and the unknown residue
        allowed[0][index] = True
    for index in range(22, 27):  # RNA, including the unknown base
        allowed[1][index] = True
    for index in range(27, 32):  # DNA, including the unknown base
        allowed[2][index] = True
    allowed[3][32] = True  # atomized tokens decode to the atomized class only
    for index in range(1, 33):  # unknown token type: any class except the mask token
        allowed[4][index] = True
    return allowed


class RFD4ProteinaPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = RFD4ProteinaPipeline
    required_input_params_in_call_signature = frozenset(["padding_mask", "generator"])
    # `generator` is required: every design needs its own random stream.
    optional_input_params = frozenset(["return_dict"])
    batch_input_params = frozenset(
        ["padding_mask", "is_in_context_conditioning", "supports_local_latent", "atom37_mask", "conditioning_features"]
    )
    callback_cfg_params = frozenset()
    output_shape = (3, 37, 3)

    def get_dummy_components(self, depths=(1, 2), feats_seq=("xt_bb_ca", "xt_local_latents")):
        torch.manual_seed(0)
        transformer = RFD4ProteinaTransformer(
            nlayers=1,
            token_dim=8,
            pair_repr_dim=4,
            nheads=2,
            feats_seq=list(feats_seq),
            feats_cond_seq=[],
            feats_pair_repr=[],
            feats_pair_cond=[],
            dim_cond=8,
            latent_dim=4,
            num_sequence_classes=33,
            num_token_types=5,
        )
        torch.manual_seed(0)
        vae = RFD4ProteinaAutoencoder(
            atom37_mask=[[True] * 37 for _ in range(33)],
            allowed_residue_types=allowed_residue_types(),
            latent_dim=4,
            n_residue_types=33,
            num_token_types=5,
            nlayers=1,
            token_dim=8,
            nheads=2,
            dim_cond=8,
            pair_repr_dim=4,
            encoder_feats_seq=["chain_idx_seq"],
            decoder_feats_seq=["z_latent_seq"],
        )
        # `t_lim_ode_below=1` keeps every interval on the noise-free branch, so integration draws no random numbers.
        scheduler_kwargs = {"tail_steps": 0, "t_lim_ode": 0.98, "t_lim_ode_below": 1.0}
        return {
            "transformer": transformer,
            "vae": vae,
            "bb_ca_scheduler": RFD4ProteinaScheduler("log", "1/t", 16.0, 3.0, **scheduler_kwargs),
            "local_latents_scheduler": RFD4ProteinaScheduler("power", "tan", 4.0, 8.0, **scheduler_kwargs),
            "recycle_depths": list(depths),
        }

    def get_dummy_inputs(self):
        return {
            **prepared_inputs(torch_device),
            "generator": self.get_generator(0),
        }

    def batch_input(self, value, batch_size):
        # Inputs are `(batch, ...)` tensors, or a dictionary of them: batching repeats along the batch axis.
        if isinstance(value, dict):
            return {name: self.batch_input(tensor, batch_size) for name, tensor in value.items()}
        return value.repeat(batch_size, *([1] * (value.ndim - 1)))


class TestRFD4ProteinaPipeline(RFD4ProteinaPipelineTesterConfig, PipelineTesterMixin):
    def assert_rejected_before_rng(self, pipe, inputs, error, match):
        before = inputs["generator"].get_state().clone()
        with pytest.raises(error, match=match):
            pipe(**inputs)
        assert torch.equal(inputs["generator"].get_state(), before)

    def coord_group_inputs(self):
        inputs = prepared_inputs(torch_device)
        groups = torch.zeros(1, 3, 37, dtype=torch.long)
        groups[:, 1] = 1
        inputs["conditioning_features"][CONDITION_KEYS["coordinate_group"]] = groups.to(torch_device)
        return inputs, groups

    def test_inference_batch_single_identical(self):
        # The shared test copies `num_inference_steps` into the batched inputs, and this pipeline has no such argument.
        super().test_inference_batch_single_identical(additional_params_copy_to_batched_inputs=[])

    def test_components_function(self):
        # The shared test expects every dummy component to be a module, but `recycle_depths` is a list of ints that the
        # pipeline stores in its config, so it is left out of the comparison.
        init_components = self.get_dummy_components()
        init_components = {k: v for k, v in init_components.items() if k != "recycle_depths"}
        pipe = self.get_pipeline(**init_components)
        assert set(pipe.components.keys()) == set(init_components.keys())

    @pytest.mark.skip(
        reason="The shared test passes `latents` and `output_type='latent'`, which this pipeline does not have; "
        "`test_callback_on_step_end` covers callbacks."
    )
    def test_callback_inputs(self):
        pass

    def test_pipeline_calls_model_once_per_step_and_recycles_inside_the_call(self):
        pipe = self.get_pipeline().to(torch_device)
        calls, cycles = [], []
        handles = [
            pipe.transformer.register_forward_hook(lambda *_: calls.append(None)),
            pipe.transformer.transformer_layers[0].register_forward_hook(lambda *_: cycles.append(None)),
        ]
        pipe(**self.get_dummy_inputs())
        for handle in handles:
            handle.remove()
        assert len(calls) == 2
        assert len(cycles) == 3  # recycle depths (1, 2) execute 1 + 2 trunk passes

    def test_callback_on_step_end(self):
        pipe = self.get_pipeline().to(torch_device)
        seen = []

        def callback(pipe, step, timestep, callback_kwargs):
            assert set(callback_kwargs) == {"bb_ca", "last_clean_bb_ca"}
            seen.append(step)
            if step == 0:
                return {"bb_ca": torch.zeros_like(callback_kwargs["bb_ca"])}
            return None

        output = pipe(
            **self.get_dummy_inputs(),
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["bb_ca", "last_clean_bb_ca"],
        )
        assert seen == [0, 1]
        assert torch.isfinite(output.bb_ca).all()
        with pytest.raises(ValueError, match="callback_on_step_end_tensor_inputs"):
            pipe(
                **self.get_dummy_inputs(),
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )

    def test_missing_sequence_condition_fails_before_sampling(self):
        inputs = {**self.get_dummy_inputs(), "generator": self.get_generator(7), "conditioning_features": {}}
        self.assert_rejected_before_rng(self.get_pipeline().to(torch_device), inputs, KeyError, SEQUENCE_KEY)

    def test_pipeline_prepares_random_coord_groups_outside_transformer(self):
        pipe = self.get_pipeline(
            **self.get_dummy_components(depths=(0, 1), feats_seq=["xt_bb_ca", "xt_local_latents", "coord_group_seq"])
        ).to(torch_device)
        inputs, groups = self.coord_group_inputs()
        expected_generator = self.get_generator(7)
        torch.randn((1, 3, 3), generator=expected_generator)
        torch.randn((1, 3), generator=expected_generator)  # the stochastic centering offset of the one molecule
        torch.randn((1, 3, 4), generator=expected_generator)
        expected = [torch.randperm(2, generator=expected_generator)[groups] for _ in range(2)]
        captured = []
        handle = pipe.transformer.register_forward_pre_hook(
            lambda module, args, kwargs: captured.append(kwargs["conditioning"]["_prepared_coord_group_seq"].cpu()),
            with_kwargs=True,
        )
        actual_generator = self.get_generator(7)
        pipe(**inputs, generator=actual_generator)
        handle.remove()
        assert len(captured) == 2
        assert all(torch.equal(actual, wanted) for actual, wanted in zip(captured, expected))
        assert torch.equal(actual_generator.get_state(), expected_generator.get_state())

    def test_coordinate_group_relabeling_uses_row_generators_without_global_rng_mutation(self):
        conditioning = {
            CONDITION_KEYS["coordinate_group"]: torch.tensor(
                [[[-1, 0, 7] + [-1] * 34, [-1, 0, 9] + [-1] * 34, [-1] * 37]]
            )
        }
        before = torch.get_rng_state().clone()
        first = prepare_coord_group_seq(conditioning, [self.get_generator(23)])
        replay = prepare_coord_group_seq(conditioning, [self.get_generator(23)])
        assert torch.equal(first, replay)
        assert torch.equal(before, torch.get_rng_state())

    def test_partial_and_inverse_initialization_use_clean_tensor_inputs(self):
        pipe = self.get_pipeline().to(torch_device)
        coords_nm = torch.zeros(1, 3, 37, 3, device=torch_device)
        coords_nm[:, 0, 1] = torch.tensor([1.0, 2.0, 3.0], device=torch_device)
        inputs = {**prepared_inputs(torch_device), "coords_nm": coords_nm}
        partial = pipe(**inputs, generator=self.get_generator(13), partial_t_bb_ca=0.5, partial_t_local_latents=0.5)
        inverse = pipe(
            **inputs,
            generator=self.get_generator(13),
            partial_t_bb_ca=0.0,
            partial_t_local_latents=0.0,
            inverse_folding=True,
        )
        assert torch.isfinite(partial.bb_ca).all() and torch.isfinite(partial.local_latents).all()
        assert torch.equal(inverse.bb_ca[:, 0], coords_nm[:, 0, 1])
        rejected = {**self.get_dummy_inputs(), "generator": self.get_generator(7), "coords_nm": coords_nm}
        self.assert_rejected_before_rng(
            pipe, {**rejected, "inverse_folding": True}, ValueError, "inverse_folding requires partial_t"
        )
        self.assert_rejected_before_rng(
            pipe,
            {**rejected, "partial_t_bb_ca": 0.5, "partial_t_local_latents": 0.0, "inverse_folding": True},
            ValueError,
            "zero start time",
        )
        self.assert_rejected_before_rng(pipe, {**rejected, "partial_t_bb_ca": 0.5}, ValueError, "both modalities")

    def test_decoder_sequence_prediction_obeys_token_type_and_user_condition(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = prepared_inputs(torch_device)
        inputs["conditioning_features"][SEQUENCE_KEY] = torch.tensor([[0, 8, 0]], device=torch_device)
        inputs["token_type"] = torch.zeros(1, 3, dtype=torch.long, device=torch_device)

        def fixed_logits(module, args, kwargs, output):
            logits = torch.zeros_like(output.sequence_logits)
            logits[..., 0], logits[..., 22], logits[..., 1] = 100, 99, 3
            return RFD4ProteinaTransformerOutput(
                backbone=output.backbone,
                local_latents=output.local_latents,
                sequence_logits=logits,
                insertion_logits=output.insertion_logits,
            )

        decoded_features = []
        handles = [
            pipe.transformer.register_forward_hook(fixed_logits, with_kwargs=True),
            pipe.vae.decoder.register_forward_pre_hook(
                lambda module, args: decoded_features.append(args[0][SEQUENCE_KEY].cpu())
            ),
        ]
        pipe(**inputs, generator=self.get_generator(7))
        for handle in handles:
            handle.remove()
        # Position 0 is filled with the best protein residue (not the mask token or an RNA type); position 1 is user-fixed.
        assert torch.equal(decoded_features[0], torch.tensor([[1, 8, 0]]))

    def test_pipeline_preserves_inputs_and_independent_batch_rows(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        original = {name: value.clone() for name, value in inputs["conditioning_features"].items()}
        single_generator = self.get_generator(7)
        single = pipe(**{**inputs, "generator": single_generator})
        for name, value in original.items():
            assert torch.equal(inputs["conditioning_features"][name], value)
        generators = [self.get_generator(7), self.get_generator(8)]
        batched = {name: self.batch_input(inputs[name], 2) for name in self.batch_input_params}
        output = pipe(**{**inputs, **batched, "generator": generators})
        torch.testing.assert_close(output.bb_ca[:1], single.bb_ca, rtol=0, atol=1e-6)
        torch.testing.assert_close(output.local_latents[:1], single.local_latents, rtol=0, atol=1e-6)
        assert torch.equal(generators[0].get_state(), single_generator.get_state())

    def test_num_designs_per_prompt_repeats_every_input_row(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs.pop("generator")
        two_rows = {name: self.batch_input(inputs[name], 2) for name in self.batch_input_params}
        # The second input row designs one more token, which shows up in the emit mask of its designs.
        two_rows["is_in_context_conditioning"][1, 1] = False
        generators = [self.get_generator(seed) for seed in range(4)]
        output = pipe(**{**inputs, **two_rows}, num_designs_per_prompt=2, generator=generators)
        assert output.coords_nm.shape[0] == 4
        # Rows 0 and 1 are the two designs of input row 0; rows 2 and 3 those of input row 1.
        assert output.emit_mask[0].equal(output.emit_mask[1])
        assert output.emit_mask[2].equal(output.emit_mask[3])
        assert not output.emit_mask[0].equal(output.emit_mask[2])
        # Repeating by hand with the same generators gives the same designs.
        repeated = {
            name: value.repeat_interleave(2, dim=0)
            if isinstance(value, torch.Tensor)
            else {key: tensor.repeat_interleave(2, dim=0) for key, tensor in value.items()}
            for name, value in two_rows.items()
        }
        by_hand = pipe(**{**inputs, **repeated}, generator=[self.get_generator(seed) for seed in range(4)])
        torch.testing.assert_close(output.bb_ca, by_hand.bb_ca, rtol=0, atol=0)
        with pytest.raises(ValueError, match="one distinct generator per batch row"):
            pipe(**{**inputs, **two_rows}, num_designs_per_prompt=2, generator=generators[:2])
        with pytest.raises(ValueError, match="num_designs_per_prompt"):
            pipe(**inputs, num_designs_per_prompt=0)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_numpy_output_from_reduced_precision(self, dtype):
        """numpy has no bfloat16 or float16 path, and the released recipe runs in bfloat16."""
        pipe = self.get_pipeline().to(torch_device).to(dtype)
        output = pipe(**self.get_dummy_inputs(), output_type="np")
        assert isinstance(output.coords_nm, np.ndarray)
        assert output.coords_nm.dtype == np.float32
        assert np.isfinite(output.coords_nm).all()

    def test_non_finite_coordinates_rejected_before_rng(self):
        """Unresolved residues are masked in by `atom37_mask` but have no coordinates, and encoding them
        turns the whole latent NaN. The encode paths must say so rather than return a NaN structure."""
        coords_nm = torch.zeros(1, 3, 37, 3, device=torch_device)
        coords_nm[0, 0, 1] = float("nan")
        inputs = {
            **self.get_dummy_inputs(),
            "generator": self.get_generator(7),
            "coords_nm": coords_nm,
            "partial_t_bb_ca": 0.3,
            "partial_t_local_latents": 0.6,
        }
        self.assert_rejected_before_rng(self.get_pipeline().to(torch_device), inputs, ValueError, "not finite")

    @pytest.mark.parametrize("invalid", ["rank", "dtype", "shape", "device", "empty"])
    def test_invalid_sampling_masks_rejected_before_rng(self, invalid):
        inputs = prepared_inputs(torch_device)
        if invalid == "rank":
            inputs["padding_mask"] = inputs["padding_mask"][0]
        elif invalid == "dtype":
            inputs["supports_local_latent"] = inputs["supports_local_latent"].float()
        elif invalid == "shape":
            inputs["is_in_context_conditioning"] = inputs["is_in_context_conditioning"][:, :1]
        elif invalid == "device":
            inputs["supports_local_latent"] = inputs["supports_local_latent"].to("meta")
        else:
            inputs["padding_mask"] = inputs["padding_mask"][:, :0]
        inputs = {**self.get_dummy_inputs(), "generator": self.get_generator(7), **inputs}
        self.assert_rejected_before_rng(
            self.get_pipeline().to(torch_device), inputs, ValueError, "sampling masks|padding_mask"
        )

    def test_generators_must_be_distinct_per_row(self):
        inputs = self.get_dummy_inputs()
        shared = self.get_generator(1)
        batched = {name: self.batch_input(inputs[name], 2) for name in self.batch_input_params}
        with pytest.raises(ValueError, match="one distinct generator per batch row"):
            self.get_pipeline().to(torch_device)(**{**inputs, **batched, "generator": [shared, shared]})

    def test_stochastic_centering_offsets_each_molecule_by_one_shared_vector(self):
        pipe = self.get_pipeline().to(torch_device)
        mask = torch.ones(1, 4, dtype=torch.bool, device=torch_device)
        conditioning = {"molecule_iid": torch.tensor([[0, 0, 1, 1]], device=torch_device)}
        centered, _ = pipe.prepare_latents(mask, mask, conditioning, [self.get_generator(3)])
        base = pipe._sample_noise((1, 4, 3), mask, [self.get_generator(3)], zero_com=True)
        offset = centered - base
        # Tokens of one molecule move together, and the two molecules move differently.
        torch.testing.assert_close(offset[:, 0], offset[:, 1])
        torch.testing.assert_close(offset[:, 2], offset[:, 3])
        assert not torch.allclose(offset[:, 0], offset[:, 2])

    def test_prepare_latents_replays_distinct_row_generators(self):
        pipe = self.get_pipeline().to(torch_device)
        mask = torch.tensor([[True, False], [True, False]], device=torch_device)
        conditioning = {"molecule_iid": torch.zeros(2, 2, dtype=torch.long, device=torch_device)}
        left = [torch.Generator().manual_seed(7), torch.Generator().manual_seed(11)]
        right = [torch.Generator().manual_seed(7), torch.Generator().manual_seed(11)]
        first = pipe.prepare_latents(mask, mask, conditioning, left)
        second = pipe.prepare_latents(mask, mask, conditioning, right)
        for first_value, second_value in zip(first, second):
            torch.testing.assert_close(first_value, second_value, rtol=0, atol=0)
            assert first_value[:, 1].count_nonzero() == 0
        assert all(a.get_state().equal(b.get_state()) for a, b in zip(left, right))


class TestRFD4ProteinaPipelineMemory(RFD4ProteinaPipelineTesterConfig, MemoryTesterMixin):
    pass
