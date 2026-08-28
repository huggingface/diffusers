# coding=utf-8
# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import torch

from diffusers import AutoModel, ControlNetModel, DDIMScheduler, ModularPipeline, UNet2DConditionModel
from diffusers.modular_pipelines import ComponentSpec, ModularPipelineBlocks


class TestAutoModelLoadIdTagging:
    def test_automodel_tags_load_id(self):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe", subfolder="unet")

        assert hasattr(model, "_diffusers_load_id"), "Model should have _diffusers_load_id attribute"
        assert model._diffusers_load_id != "null", "_diffusers_load_id should not be 'null'"

        # Verify load_id contains the expected fields
        load_id = model._diffusers_load_id
        assert "hf-internal-testing/tiny-stable-diffusion-xl-pipe" in load_id
        assert "unet" in load_id

    def test_automodel_update_components(self):
        pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")
        pipe.load_components(dtype=torch.float32)

        auto_model = AutoModel.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe", subfolder="unet")

        pipe.update_components(unet=auto_model)

        assert pipe.unet is auto_model

        assert "unet" in pipe._component_specs
        spec = pipe._component_specs["unet"]
        assert spec.pretrained_model_name_or_path == "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        assert spec.subfolder == "unet"

    def test_load_components_loads_local_single_file_path(self, tmp_path):
        pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")

        model = ControlNetModel.from_pretrained("hf-internal-testing/tiny-controlnet")
        model.save_pretrained(tmp_path)

        local_ckpt_path = str(tmp_path / "diffusion_pytorch_model.safetensors")

        pipe._component_specs["controlnet"] = ComponentSpec(
            name="controlnet",
            type_hint=ControlNetModel,
            pretrained_model_name_or_path=local_ckpt_path,
        )
        pipe.load_components(names="controlnet", config=str(tmp_path))

        assert pipe.controlnet is not None
        assert isinstance(pipe.controlnet, ControlNetModel)
        assert pipe._component_specs["controlnet"].pretrained_model_name_or_path == local_ckpt_path
        assert getattr(pipe.controlnet, "_diffusers_load_id", None) not in (None, "null")


class TestLoadComponentsSkipBehavior:
    def test_load_components_skips_already_loaded(self):
        pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")
        pipe.load_components(dtype=torch.float32)

        original_unet = pipe.unet

        pipe.load_components()

        # Verify that the unet is the same object (not reloaded)
        assert pipe.unet is original_unet, "load_components should skip already loaded components"

    def test_load_components_selective_loading(self):
        pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")

        pipe.load_components(names="unet", dtype=torch.float32)

        # Verify only requested component was loaded.
        assert hasattr(pipe, "unet")
        assert pipe.unet is not None
        assert getattr(pipe, "vae", None) is None

    def test_load_components_selective_loading_incremental(self):
        """Loading a subset of components should not affect already-loaded components."""
        pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")

        pipe.load_components(names="unet", dtype=torch.float32)
        pipe.load_components(names="text_encoder", dtype=torch.float32)

        assert hasattr(pipe, "unet")
        assert pipe.unet is not None
        assert hasattr(pipe, "text_encoder")
        assert pipe.text_encoder is not None

    def test_load_components_skips_invalid_pretrained_path(self):
        pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")

        pipe._component_specs["test_component"] = ComponentSpec(
            name="test_component",
            type_hint=torch.nn.Module,
            pretrained_model_name_or_path=None,
            default_creation_method="from_pretrained",
        )
        pipe.load_components(dtype=torch.float32)

        # Verify test_component was not loaded
        assert not hasattr(pipe, "test_component") or pipe.test_component is None


class TestCustomModelSavePretrained:
    def test_save_pretrained_updates_index_for_local_model(self, tmp_path):
        """When a component without _diffusers_load_id (custom/local model) is saved,
        modular_model_index.json should point to the save directory."""
        pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")
        pipe.load_components(dtype=torch.float32)

        pipe.unet._diffusers_load_id = "null"

        save_dir = str(tmp_path / "my-pipeline")
        pipe.save_pretrained(save_dir)

        with open(os.path.join(save_dir, "modular_model_index.json")) as f:
            index = json.load(f)

        _library, _cls, unet_spec = index["unet"]
        assert unet_spec["pretrained_model_name_or_path"] == save_dir
        assert unet_spec["subfolder"] == "unet"

        _library, _cls, vae_spec = index["vae"]
        assert vae_spec["pretrained_model_name_or_path"] == "hf-internal-testing/tiny-stable-diffusion-xl-pipe"

    def test_save_pretrained_roundtrip_with_local_model(self, tmp_path):
        """A pipeline with a custom/local model should be saveable and re-loadable with identical outputs."""
        pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")
        pipe.load_components(dtype=torch.float32)

        pipe.unet._diffusers_load_id = "null"

        original_state_dict = pipe.unet.state_dict()

        save_dir = str(tmp_path / "my-pipeline")
        pipe.save_pretrained(save_dir)

        loaded_pipe = ModularPipeline.from_pretrained(save_dir)
        loaded_pipe.load_components(dtype=torch.float32)

        assert loaded_pipe.unet is not None
        assert loaded_pipe.unet.__class__.__name__ == pipe.unet.__class__.__name__

        loaded_state_dict = loaded_pipe.unet.state_dict()
        assert set(original_state_dict.keys()) == set(loaded_state_dict.keys())
        for key in original_state_dict:
            assert torch.equal(original_state_dict[key], loaded_state_dict[key]), f"Mismatch in {key}"

    def test_save_pretrained_updates_index_for_model_with_no_load_id(self, tmp_path):
        """testing the workflow of update the pipeline with a custom model and save the pipeline,
        the modular_model_index.json should point to the save directory."""
        pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")
        pipe.load_components(dtype=torch.float32)

        unet = UNet2DConditionModel.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe", subfolder="unet"
        )
        assert not hasattr(unet, "_diffusers_load_id")

        pipe.update_components(unet=unet)

        save_dir = str(tmp_path / "my-pipeline")
        pipe.save_pretrained(save_dir)

        with open(os.path.join(save_dir, "modular_model_index.json")) as f:
            index = json.load(f)

        _library, _cls, unet_spec = index["unet"]
        assert unet_spec["pretrained_model_name_or_path"] == save_dir
        assert unet_spec["subfolder"] == "unet"

        _library, _cls, vae_spec = index["vae"]
        assert vae_spec["pretrained_model_name_or_path"] == "hf-internal-testing/tiny-stable-diffusion-xl-pipe"

    def test_save_pretrained_overwrite_modular_index(self, tmp_path):
        """With overwrite_modular_index=True, all component references should point to the save directory."""
        pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")
        pipe.load_components(dtype=torch.float32)

        save_dir = str(tmp_path / "my-pipeline")
        pipe.save_pretrained(save_dir, overwrite_modular_index=True)

        with open(os.path.join(save_dir, "modular_model_index.json")) as f:
            index = json.load(f)

        for component_name in ["unet", "vae", "text_encoder", "text_encoder_2"]:
            if component_name not in index:
                continue
            _library, _cls, spec = index[component_name]
            assert spec["pretrained_model_name_or_path"] == save_dir, (
                f"{component_name} should point to save dir but got {spec['pretrained_model_name_or_path']}"
            )
            assert spec["subfolder"] == component_name

        loaded_pipe = ModularPipeline.from_pretrained(save_dir)
        loaded_pipe.load_components(dtype=torch.float32)

        assert loaded_pipe.unet is not None
        assert loaded_pipe.vae is not None


class TestModularPipelineInitFallback:
    """Test that ModularPipeline.__init__ falls back to default_blocks_name when
    _blocks_class_name is a base class (e.g. SequentialPipelineBlocks saved by from_blocks_dict)."""

    def test_init_fallback_when_blocks_class_name_is_base_class(self, tmp_path):
        # 1. Load pipeline and get a workflow (returns a base SequentialPipelineBlocks)
        pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")
        t2i_blocks = pipe.blocks.get_workflow("text2image")
        assert t2i_blocks.__class__.__name__ == "SequentialPipelineBlocks"

        # 2. Use init_pipeline to create a new pipeline from the workflow blocks
        t2i_pipe = t2i_blocks.init_pipeline("hf-internal-testing/tiny-stable-diffusion-xl-pipe")

        # 3. Save and reload — the saved config will have _blocks_class_name="SequentialPipelineBlocks"
        save_dir = str(tmp_path / "pipeline")
        t2i_pipe.save_pretrained(save_dir)
        loaded_pipe = ModularPipeline.from_pretrained(save_dir)

        # 4. Verify it fell back to default_blocks_name and has correct blocks
        assert loaded_pipe.__class__.__name__ == pipe.__class__.__name__
        assert loaded_pipe._blocks.__class__.__name__ == pipe._blocks.__class__.__name__
        assert len(loaded_pipe._blocks.sub_blocks) == len(pipe._blocks.sub_blocks)


_MISSING_HUB_ID = "org/this-repo-does-not-exist-14640"


class _LocalSnapshotBlocks(ModularPipelineBlocks):
    def __init__(self, component_names=("scheduler",)):
        self._component_names = component_names
        super().__init__()

    @property
    def expected_components(self):
        type_hints = {"scheduler": DDIMScheduler, "unet": UNet2DConditionModel}
        return [ComponentSpec(name, type_hints[name]) for name in self._component_names]


def _write_modular_index(snapshot_dir, components, repo_ids=None):
    index = {
        "_class_name": "ModularPipeline",
        "_diffusers_version": "0.40.0.dev0",
        "_blocks_class_name": "ModularPipelineBlocks",
    }
    for name, class_name in components.items():
        repo_id = (repo_ids or {}).get(name, _MISSING_HUB_ID)
        index[name] = [
            "diffusers",
            class_name,
            {
                "type_hint": ["diffusers", class_name],
                "pretrained_model_name_or_path": repo_id,
                "subfolder": name,
                "variant": None,
                "revision": None,
            },
        ]
    with open(os.path.join(snapshot_dir, "modular_model_index.json"), "w") as f:
        json.dump(index, f)


class TestLocalModularSnapshotLoading:
    def test_init_pipeline_rewrites_hub_ids_when_subfolder_exists(self, tmp_path):
        snapshot_dir = str(tmp_path / "snapshot")
        os.makedirs(snapshot_dir)
        DDIMScheduler().save_pretrained(os.path.join(snapshot_dir, "scheduler"))
        _write_modular_index(snapshot_dir, {"scheduler": "DDIMScheduler"})

        pipe = _LocalSnapshotBlocks().init_pipeline(snapshot_dir)

        assert pipe._component_specs["scheduler"].pretrained_model_name_or_path == snapshot_dir

        pipe.load_components(names="scheduler", local_files_only=True)
        assert pipe.scheduler is not None
        assert isinstance(pipe.scheduler, DDIMScheduler)

    def test_constructor_rewrites_hub_ids_when_subfolder_exists(self, tmp_path):
        snapshot_dir = str(tmp_path / "snapshot")
        os.makedirs(snapshot_dir)
        DDIMScheduler().save_pretrained(os.path.join(snapshot_dir, "scheduler"))
        _write_modular_index(snapshot_dir, {"scheduler": "DDIMScheduler"})

        pipe = ModularPipeline(
            blocks=_LocalSnapshotBlocks(),
            pretrained_model_name_or_path=snapshot_dir,
            local_files_only=True,
        )

        assert pipe._component_specs["scheduler"].pretrained_model_name_or_path == snapshot_dir
        pipe.load_components(names="scheduler", local_files_only=True)
        assert pipe.scheduler is not None

    def test_missing_local_subfolder_keeps_hub_id(self, tmp_path):
        snapshot_dir = str(tmp_path / "snapshot")
        os.makedirs(snapshot_dir)
        _write_modular_index(snapshot_dir, {"scheduler": "DDIMScheduler"})

        pipe = _LocalSnapshotBlocks().init_pipeline(snapshot_dir)

        assert pipe._component_specs["scheduler"].pretrained_model_name_or_path == _MISSING_HUB_ID

    def test_mixed_snapshot_rewrites_only_present_components(self, tmp_path):
        snapshot_dir = str(tmp_path / "snapshot")
        os.makedirs(snapshot_dir)
        DDIMScheduler().save_pretrained(os.path.join(snapshot_dir, "scheduler"))
        _write_modular_index(snapshot_dir, {"scheduler": "DDIMScheduler", "unet": "UNet2DConditionModel"})

        pipe = _LocalSnapshotBlocks(component_names=("scheduler", "unet")).init_pipeline(snapshot_dir)

        assert pipe._component_specs["scheduler"].pretrained_model_name_or_path == snapshot_dir
        assert pipe._component_specs["unet"].pretrained_model_name_or_path == _MISSING_HUB_ID

        pipe.load_components(names="scheduler", local_files_only=True)
        assert pipe.scheduler is not None
        assert pipe.unet is None

    def test_existing_local_spec_path_is_not_overwritten(self, tmp_path):
        other_dir = str(tmp_path / "other")
        snapshot_dir = str(tmp_path / "snapshot")
        os.makedirs(snapshot_dir)
        DDIMScheduler(num_train_timesteps=50).save_pretrained(os.path.join(other_dir, "scheduler"))
        DDIMScheduler(num_train_timesteps=1000).save_pretrained(os.path.join(snapshot_dir, "scheduler"))
        _write_modular_index(
            snapshot_dir,
            {"scheduler": "DDIMScheduler"},
            repo_ids={"scheduler": other_dir},
        )

        pipe = _LocalSnapshotBlocks().init_pipeline(snapshot_dir)

        assert pipe._component_specs["scheduler"].pretrained_model_name_or_path == other_dir
        pipe.load_components(names="scheduler", local_files_only=True)
        assert pipe.scheduler.config.num_train_timesteps == 50
