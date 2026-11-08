# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def training(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "examples/rfd4_proteina/train_rfd4_proteina_lora.py"
    spec = importlib.util.spec_from_file_location("rfd4_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "logger", Mock())
    monkeypatch.setitem(sys.modules, "rfd4_proteina_prep", SimpleNamespace(prepare_cif=Mock()))
    return module


def features(tokens=2, resolved=True, emitted=True):
    return {
        "padding_mask": torch.ones(1, tokens, dtype=torch.bool),
        "is_in_context_conditioning": torch.full((1, tokens), not emitted, dtype=torch.bool),
        "atom37_mask": torch.ones(1, tokens, 37, dtype=torch.bool),
        "coords_nm": torch.full((1, tokens, 37, 3), 0.0 if resolved else float("nan")),
    }


def test_nested_cache_keys_and_file_changes(training, tmp_path):
    for subdir in ("a", "b"):
        (tmp_path / subdir).mkdir()
        (tmp_path / subdir / "model.cif").touch()
    prep = sys.modules["rfd4_proteina_prep"].prepare_cif
    prep.side_effect = lambda path, **kwargs: (features(2 if path.parent.name == "a" else 3), None)
    dataset = training.PreparedStructureDataset(tmp_path, 0, 4, tmp_path / "cache")
    assert [dataset[i]["padding_mask"].shape[1] for i in range(2)] == [2, 3]
    assert prep.call_count == 2
    assert len(list((tmp_path / "cache").glob("*.pt"))) == 2
    dataset.paths[0].write_text("changed source")
    dataset[0]
    assert prep.call_count == 3


def test_filter_before_sharding_and_cache_in_memory(training, tmp_path):
    for name in ("good", "invalid", "long", "unresolved", "context"):
        (tmp_path / f"{name}.cif").touch()

    def prepare(path, **kwargs):
        if path.stem == "invalid":
            raise ValueError("unusable structure")
        return features(5 if path.stem == "long" else 2, path.stem != "unresolved", path.stem != "context"), None

    prep = sys.modules["rfd4_proteina_prep"].prepare_cif
    prep.side_effect = prepare
    dataset = training.PreparedStructureDataset(tmp_path, 0, 4)
    assert [path.stem for path in dataset.paths] == ["good"]
    for _ in range(3):
        assert dataset[0]["padding_mask"].shape == (1, 2)
    assert prep.call_count == 5
    assert len(dataset._memory_cache) == 1
    (tmp_path / "good.cif").unlink()
    with pytest.raises(ValueError, match="No trainable structures"):
        training.PreparedStructureDataset(tmp_path, 0, 4)


def test_concurrent_cache_preparation_is_atomic(training, tmp_path, monkeypatch):
    (tmp_path / "model.cif").touch()
    cache = tmp_path / "cache"
    prep = sys.modules["rfd4_proteina_prep"].prepare_cif

    def prepare(*args, **kwargs):
        time.sleep(0.05)
        return features(), None

    prep.side_effect = prepare
    save = torch.save

    def slow_save(value, handle):
        assert not list(cache.glob("*.pt.lock"))[0].with_suffix("").exists()
        time.sleep(0.05)
        save(value, handle)

    monkeypatch.setattr(torch, "save", slow_save)
    with ThreadPoolExecutor(max_workers=4) as pool:
        datasets = list(pool.map(lambda _: training.PreparedStructureDataset(tmp_path, 0, 4, cache), range(4)))
    assert prep.call_count == 1
    assert len(list(cache.glob("*.pt"))) == 1
    for dataset in datasets:
        torch.testing.assert_close(dataset[0]["coords_nm"], features()["coords_nm"])


@pytest.mark.parametrize("main", [True, False])
def test_checkpoint_all_ranks_save_only_main_prunes(training, tmp_path, main):
    (tmp_path / "checkpoint-1").mkdir()
    accelerator = Mock(is_main_process=main)
    args = SimpleNamespace(output_dir=str(tmp_path), checkpoints_total_limit=1)
    training.save_checkpoint(accelerator, args, 2)
    accelerator.save_state.assert_called_once_with(str(tmp_path / "checkpoint-2"))
    assert accelerator.wait_for_everyone.call_count == 2
    assert (tmp_path / "checkpoint-1").exists() is not main


@pytest.mark.parametrize("fail", [True, False])
@pytest.mark.parametrize("initial_training", [True, False])
def test_validation_eval_and_restore(training, tmp_path, monkeypatch, fail, initial_training):
    transformer = torch.nn.Sequential(torch.nn.Dropout(0.5)).train(initial_training)
    prep_features = {key: torch.zeros(1) for key in training.PIPELINE_INPUTS}
    writer = Mock()
    monkeypatch.setitem(
        sys.modules,
        "rfd4_proteina_prep",
        SimpleNamespace(prepare_cif=lambda *a, **k: (prep_features, None), write_cif=writer),
    )

    class Pipeline:
        def __init__(self, **kwargs):
            pass

        def set_progress_bar_config(self, **kwargs):
            pass

        def __call__(self, **kwargs):
            assert not transformer.training
            assert not torch.is_grad_enabled()
            if fail:
                raise RuntimeError("generation failed")
            return object()

    monkeypatch.setattr(training, "RFD4ProteinaPipeline", Pipeline)
    args = SimpleNamespace(validation_cif="input.cif", prep_seed=0, seed=0, output_dir=str(tmp_path))

    def call():
        training.run_validation(
            SimpleNamespace(device=torch.device("cpu")),
            args,
            transformer,
            None,
            {"bb_ca": None, "local_latents": None},
            1,
        )

    if fail:
        with pytest.raises(RuntimeError, match="generation failed"):
            call()
    else:
        call()
        writer.assert_called_once()
    assert transformer.training is initial_training


def test_lora_metadata_preserves_nondefault_scaling(training, tmp_path):
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    from diffusers import RFD4ProteinaTransformer

    config = {
        "nlayers": 1,
        "token_dim": 8,
        "pair_repr_dim": 4,
        "nheads": 2,
        "feats_seq": ["xt_bb_ca", "xt_local_latents"],
        "feats_cond_seq": [],
        "feats_pair_repr": [],
        "feats_pair_cond": [],
        "dim_cond": 8,
        "latent_dim": 4,
    }
    model = RFD4ProteinaTransformer(**config)
    model.add_adapter(
        LoraConfig(r=2, lora_alpha=7, lora_dropout=0.1, target_modules=["to_q", "to_k", "to_v", "to_out"])
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_()
    training.RFD4ProteinaPipeline.save_lora_weights(
        tmp_path,
        transformer_lora_layers=get_peft_model_state_dict(model),
        **training._collate_lora_metadata({"transformer": model}),
    )
    state, metadata = training.RFD4ProteinaPipeline.lora_state_dict(tmp_path, return_lora_metadata=True)
    restored = RFD4ProteinaTransformer(**config)
    training.RFD4ProteinaPipeline.load_lora_into_transformer(
        state, transformer=restored, metadata=metadata, adapter_name="default"
    )
    assert restored.peft_config["default"].lora_alpha == 7
    assert restored.peft_config["default"].lora_dropout == 0.1
    torch.testing.assert_close(get_peft_model_state_dict(model), get_peft_model_state_dict(restored))
    for name, module in model.named_modules():
        if hasattr(module, "scaling"):
            other = restored.get_submodule(name)
            assert other.scaling["default"] == module.scaling["default"] == 3.5
            value = torch.randn(1, 2, module.in_features)
            module.eval()
            other.eval()

            # Compare the adapter contribution without depending on the randomly initialized base layer.
            def adapter(layer):
                return layer.lora_B["default"](layer.lora_A["default"](value)) * layer.scaling["default"]

            torch.testing.assert_close(adapter(module), adapter(other), rtol=0, atol=0)


def test_failed_cache_write_does_not_publish_partial_file(training, tmp_path, monkeypatch):
    (tmp_path / "model.cif").touch()
    cache = tmp_path / "cache"
    sys.modules["rfd4_proteina_prep"].prepare_cif.return_value = (features(), None)
    save = torch.save

    def fail_save(value, handle):
        handle.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        training.PreparedStructureDataset(tmp_path, 0, 4, cache)
    assert not list(cache.glob("*.pt"))
    monkeypatch.setattr(torch, "save", save)
    dataset = training.PreparedStructureDataset(tmp_path, 0, 4, cache)
    torch.testing.assert_close(dataset[0]["coords_nm"], features()["coords_nm"])
