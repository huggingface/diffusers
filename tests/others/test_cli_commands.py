# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Essential unit tests for the non-``skills`` CLI commands.

One test per contract that would ship broken if regressed. Grouped by command.
"""

import json
import os
import subprocess
import wave
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from huggingface_hub.cli._output import OutputFormat, out
from PIL import Image

from diffusers.commands.custom_blocks import CustomBlocksCommand
from diffusers.commands.run import (
    RunCommand,
    _build_task_kwargs,
    _collapse_frame_dirs,
    _download_outputs_from_sandbox,
    _kwargs_to_argv,
    _load_lora,
    _load_pipeline,
    _parse_pipeline_kwargs,
    _resolve_dtype,
    _resolve_media_inputs,
    _save_output,
    _unwrap_pipeline_output,
    _upload_inputs_to_sandbox,
)
from diffusers.commands.schema import SchemaCommand, _parse_docstring_args
from diffusers.utils.testing_utils import (
    require_accelerator,
    require_kernels_version_greater_or_equal,
    require_torch_gpu,
)


AVAILABLE_COMMANDS = ("env", "fp16_safetensors", "custom_blocks", "run", "schema", "skills")


class TestRunCommand:
    def test_parse_pipeline_kwargs(self):
        assert _parse_pipeline_kwargs('{"prompt": "a cat", "steps": 50}') == {"prompt": "a cat", "steps": 50}
        with pytest.raises(SystemExit, match="must be valid JSON"):
            _parse_pipeline_kwargs('{"prompt": "unterminated')
        with pytest.raises(SystemExit, match="must decode to a JSON object"):
            _parse_pipeline_kwargs('["not", "an", "object"]')

    def test_resolve_dtype_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            _resolve_dtype("dummy-dtype")

    def test_resolve_media_inputs(self, monkeypatch):
        monkeypatch.setattr("diffusers.commands.run.load_image", lambda v: f"img({v})")
        monkeypatch.setattr("diffusers.commands.run.load_video", lambda v: [f"frame({v})"])
        # Scalar strings load individually; string-lists load each entry (batched inputs);
        # `prompt` isn't a media key so it passes through; non-string / non-string-list values
        # (e.g. a pre-loaded PIL object modeled here as a dict) pass through untouched.
        preloaded = {"pre": "loaded"}
        kwargs = {
            "image": "url1",
            # first/last-frame video pipelines (Wan I2V, MiniMax-H3 fl2va, LTX2) take `last_image`;
            # unresolved it reaches the pipeline as a bare path string.
            "last_image": "url3",
            "control_video": "url2",
            "prompt": "text",
            "mask_image": ["a.png", "b.png"],
            "control_image": ["c.png"],
            "video": ["v1.mp4", "v2.mp4"],
            "ip_adapter_image": preloaded,
        }
        _resolve_media_inputs(kwargs)
        assert kwargs == {
            "image": "img(url1)",
            "last_image": "img(url3)",
            "control_video": ["frame(url2)"],
            "prompt": "text",
            "mask_image": ["img(a.png)", "img(b.png)"],
            "control_image": ["img(c.png)"],
            "video": [["frame(v1.mp4)"], ["frame(v2.mp4)"]],
            "ip_adapter_image": preloaded,
        }

    def test_resolve_media_inputs_audio_batch(self, monkeypatch):
        # Batched audio: each entry loads, the paired sampling-rate kwarg defaults to the first entry's rate.
        monkeypatch.setattr("diffusers.commands.run._load_audio", lambda v: (f"wave({v})", 44100))
        kwargs = {"initial_audio_waveforms": ["a.wav", "b.wav"]}
        _resolve_media_inputs(kwargs)
        assert kwargs == {
            "initial_audio_waveforms": ["wave(a.wav)", "wave(b.wav)"],
            "initial_audio_sampling_rate": 44100,
        }

    def test_upload_inputs_to_sandbox_batched(self, tmp_path):
        # Batched media on --remote: `_upload_inputs_to_sandbox` uploads each local path in a list
        # and rewrites the JSON in place. URLs and non-existent paths pass through untouched.
        local_a = tmp_path / "a.png"
        local_b = tmp_path / "b.png"
        local_a.write_bytes(b"a")
        local_b.write_bytes(b"b")

        uploads: list[tuple[str, str]] = []

        class FakeFiles:
            def upload(self, src, dst):
                uploads.append((src, dst))

        class FakeSbx:
            files = FakeFiles()

        args = Namespace(
            pipeline_kwargs=json.dumps(
                {
                    "image": str(local_a),  # scalar local → uploaded
                    "mask_image": [str(local_a), "https://example.com/x.png", str(local_b)],  # mixed batch
                    "prompt": "unchanged",
                }
            )
        )
        _upload_inputs_to_sandbox(args, FakeSbx(), run_id="rid")

        assert len(uploads) == 3  # a (scalar), a (list[0]), b (list[2]); URL skipped
        rewritten = json.loads(args.pipeline_kwargs)
        assert rewritten["image"].startswith("/tmp/diffusers-cli/inputs/rid/image_")
        assert rewritten["mask_image"][0].startswith("/tmp/diffusers-cli/inputs/rid/mask_image_0_")
        assert rewritten["mask_image"][1] == "https://example.com/x.png"  # URL untouched
        assert rewritten["mask_image"][2].startswith("/tmp/diffusers-cli/inputs/rid/mask_image_2_")
        assert rewritten["prompt"] == "unchanged"

    def test_kwargs_to_argv(self):
        argv = _kwargs_to_argv(
            "run", {"cpu_offload": "model", "vae_tiling": True, "dependencies": ["torch", "accelerate"]}
        )
        assert argv[0] == "run"
        assert "--cpu-offload" in argv and argv[argv.index("--cpu-offload") + 1] == "model"
        assert "--vae-tiling" in argv
        assert argv.count("--dependencies") == 2 and "torch" in argv and "accelerate" in argv

    def test_remote_argv_omits_hf_job_args(self):
        # `_build_task_kwargs` must strip REMOTE_KEYS (flags that control how the remote run is
        # dispatched, not the sandbox CLI) plus None/False values before we forward argv to the sandbox.
        args = Namespace(
            remote=True,
            flavor="a100-large",
            timeout="10m",
            format="auto",
            image="pytorch/pytorch:2.10.0",
            func=object(),
            model="x",
            dtype="bf16",
            device_map="cuda",  # not in REMOTE_KEYS: forwarded to the sandbox
            revision=None,
            trust_remote_code=False,
            vae_tiling=True,
        )
        assert _build_task_kwargs(args) == {
            "model": "x",
            "dtype": "bf16",
            "device_map": "cuda",
            "vae_tiling": True,
        }

    # -----------------------------------------------------------------------
    # Route flags through the CLI parser and call `_load_pipeline` directly to assert
    # their effect on a real tiny pipeline. `hf-internal-testing/tiny-flux-pipe` is small
    # enough to load without a GPU and is already used across the pipeline suite.
    # -----------------------------------------------------------------------

    pretrained_model_name_or_path = "hf-internal-testing/tiny-flux-pipe"

    def _parse_run_argv(self, extra_argv: list[str]) -> Namespace:
        parser = ArgumentParser()
        subparsers = parser.add_subparsers()
        RunCommand.register_subcommand(subparsers)
        return parser.parse_args(
            [
                "run",
                "--model",
                self.pretrained_model_name_or_path,
                "--pipeline-kwargs",
                '{"prompt": "a cat"}',
                *extra_argv,
            ]
        )

    @require_torch_gpu
    def test_group_offload_arg(self):
        from diffusers.hooks.group_offloading import _is_group_offload_enabled

        args = self._parse_run_argv(["--cpu-offload", "group"])
        pipeline = _load_pipeline(args)
        assert _is_group_offload_enabled(pipeline.transformer)

    @require_accelerator
    def test_model_cpu_offload_arg(self):
        import accelerate

        args = self._parse_run_argv(["--cpu-offload", "model"])
        pipeline = _load_pipeline(args)
        assert isinstance(pipeline.transformer._hf_hook, accelerate.hooks.CpuOffload)

    def test_vae_tiling_arg(self):
        args = self._parse_run_argv(["--vae-tiling"])
        pipeline = _load_pipeline(args)
        assert pipeline.vae.use_tiling is True

    def test_vae_slicing_arg(self):
        args = self._parse_run_argv(["--vae-slicing"])
        pipeline = _load_pipeline(args)
        assert pipeline.vae.use_slicing is True

    @require_torch_gpu
    def test_compile_arg(self):
        args = self._parse_run_argv(["--compile"])
        pipeline = _load_pipeline(args)
        # `FluxTransformer2DModel` declares `_repeated_blocks`, so `_compile_denoiser` takes the
        # regional path: `compile_repeated_blocks` calls `nn.Module.compile()` on each repeated
        # block, which sets that block's `_compiled_call_impl` in place (no `OptimizedModule`
        # wrapper, hence no `_orig_mod`, is created).
        compiled_blocks = [
            m for m in pipeline.transformer.modules() if m.__class__.__name__ in pipeline.transformer._repeated_blocks
        ]
        assert compiled_blocks
        assert all(m._compiled_call_impl is not None for m in compiled_blocks)

    @require_torch_gpu
    # `--attention-backend` only exposes Hub-hosted kernels, all of which need `kernels>=0.12`.
    @require_kernels_version_greater_or_equal("0.12")
    def test_attention_backend_arg(self):
        from diffusers.models.attention_dispatch import AttentionBackendName

        args = self._parse_run_argv(["--attention-backend", "flash_hub"])
        try:
            pipeline = _load_pipeline(args)
        except FileNotFoundError as e:
            # The Hub kernel has no prebuilt variant for this torch/CUDA/arch combination.
            pytest.skip(f"`flash_hub` kernel unavailable in this environment: {e}")
        # `set_attention_backend` stamps each attention processor's `_attention_backend` attr.
        backends = {
            m.processor._attention_backend
            for m in pipeline.transformer.modules()
            if hasattr(m, "processor") and hasattr(m.processor, "_attention_backend")
        }
        assert backends == {AttentionBackendName.FLASH_HUB}

    def test_save_output_video_saves_mp4_and_frames(self, tmp_path, monkeypatch):
        # `output_type="pt"` video is (B, F, C, H, W) from `postprocess_video`: one mp4 per batch
        # item, plus every frame as `<video-stem>-frames/<NNNN>.png` beside it.
        exported: list[tuple[int, str]] = []
        monkeypatch.setattr(
            "diffusers.commands.run.export_to_video",
            lambda frames, path, fps: exported.append((len(frames), path)),
        )
        args = Namespace(output=str(tmp_path) + os.sep, fps=24, sampling_rate=None)
        saved = _save_output(torch.zeros((2, 4, 3, 8, 8)), args)
        names = sorted(Path(p).name for p in saved)
        assert [n for n, _ in exported] == [4, 4]
        assert [n for n in names if n.endswith(".mp4")] == ["0000.mp4", "0001.mp4"]
        frame_rel = sorted(str(Path(p).relative_to(tmp_path)) for p in saved if p.endswith(".png"))
        assert frame_rel == sorted(f"{v:04d}-frames/{i:04d}.png" for v in range(2) for i in range(4))
        assert all((tmp_path / rel).exists() for rel in frame_rel)

    def test_lora_spec_passes_weight_name(self):
        # A LoRA repo can ship several weight files (e.g. a ComfyUI variant next to the diffusers
        # one), so `weight_name` has to reach `load_lora_weights` to disambiguate.
        calls = []

        class FakePipeline:
            def load_lora_weights(self, lora_id, adapter_name=None, weight_name=None):
                calls.append((lora_id, adapter_name, weight_name))

            def set_adapters(self, names, adapter_weights=None):
                calls.append(("set_adapters", names, adapter_weights))

        spec = {"lora_id": "org/style", "lora_scale": 1.1, "weight_name": "style.safetensors"}
        _load_lora(FakePipeline(), Namespace(lora=[json.dumps(spec)]))
        assert calls[0] == ("org/style", "default", "style.safetensors")
        assert calls[1] == ("set_adapters", ["default"], [1.1])

    def test_output_key_is_repeatable(self):
        # A single --output-key returns that value directly; repeating it asks the modular
        # pipeline for several named outputs at once (e.g. a video and its soundtrack).
        args = self._parse_run_argv(["--output-key", "videos"])
        assert args.output_key == ["videos"]

        args = self._parse_run_argv(["--output-key", "videos", "--output-key", "audio"])
        assert args.output_key == ["videos", "audio"]
        # Repeated flags survive the rebuild of argv for the sandbox.
        argv = _kwargs_to_argv("run", {"output_key": args.output_key})
        assert argv.count("--output-key") == 2 and "videos" in argv and "audio" in argv

    def test_download_outputs_recurses_into_subdirs(self, tmp_path):
        # Frame folders are subdirectories of the sandbox output dir; the listing marks them
        # `dir` (not `directory`), and missing that means remote runs silently lose every frame.
        class FakeFiles:
            def list(self, path):
                if path == "/out":
                    return [
                        SimpleNamespace(type="file", path="/out/0000.mp4"),
                        SimpleNamespace(type="dir", path="/out/0000-frames"),
                    ]
                return [SimpleNamespace(type="file", path=f"/out/0000-frames/{i:04d}.png") for i in range(2)]

            def download(self, src, dst):
                Path(dst).write_text("x")

        class FakeSbx:
            files = FakeFiles()

        saved = _download_outputs_from_sandbox(FakeSbx(), "/out", tmp_path)
        assert sorted(str(Path(p).relative_to(tmp_path)) for p in saved) == [
            "0000-frames/0000.png",
            "0000-frames/0001.png",
            "0000.mp4",
        ]

    def test_collapse_frame_dirs(self):
        # Hundreds of frame paths would bury the mp4 and flood the terminal, so each
        # `<stem>-frames/` group is reported as its directory. Non-frame paths pass through.
        paths = ["/o/0000.mp4"] + [f"/o/0000-frames/{i:04d}.png" for i in range(120)] + ["/o/0000.wav"]
        assert _collapse_frame_dirs(paths) == ["/o/0000.mp4", "/o/0000-frames", "/o/0000.wav"]
        assert _collapse_frame_dirs(["/o/0000.png", "/o/0001.png"]) == ["/o/0000.png", "/o/0001.png"]

    def test_save_output_tensor_image_batch(self, tmp_path):
        # `output_type="pt"` images are channels-first (B, C, H, W): one png per batch item.
        args = Namespace(output=str(tmp_path) + os.sep, fps=24, sampling_rate=None)
        saved = _save_output(torch.zeros((2, 3, 8, 8)), args)
        assert [Path(p).suffix for p in saved] == [".png", ".png"]
        assert all(Path(p).exists() for p in saved)

    def test_save_output_nested_pil_video_batch(self, tmp_path, monkeypatch):
        # list[list[PIL]] (video pipelines under their default output_type="pil") saves one mp4
        # per inner sequence, plus the per-frame pngs.
        exported: list[str] = []
        monkeypatch.setattr("diffusers.commands.run.export_to_video", lambda frames, path, fps: exported.append(path))
        frames = [Image.new("RGB", (8, 8)) for _ in range(3)]
        args = Namespace(output=str(tmp_path) + os.sep, fps=24, sampling_rate=None)
        saved = _save_output([frames, frames], args)
        assert len(exported) == 2
        assert sorted(Path(p).suffix for p in saved) == [".mp4"] * 2 + [".png"] * 6

    def test_save_output_stereo_audio(self, tmp_path):
        # (B, C, samples) waveforms (e.g. StableAudio's native output_type="pt") save as
        # multi-channel wavs instead of falling through as unrecognized.
        args = Namespace(output=str(tmp_path) + os.sep, fps=24, sampling_rate=44100)
        saved = _save_output(torch.zeros((1, 2, 1000)), args)
        assert [Path(p).suffix for p in saved] == [".wav"]
        with wave.open(saved[0]) as w:
            assert w.getnchannels() == 2
            assert w.getnframes() == 1000

    def test_unwrap_pipeline_output_multi_media(self):
        # Every media field present on an output is saved, not just the first match (LTX2 returns
        # both `frames` and `audio`), and payloads keep their batch dimension.
        class Output:
            frames = torch.zeros((1, 4, 3, 8, 8))
            audio = torch.zeros((1, 2, 1000))

        payloads = _unwrap_pipeline_output(Output())
        assert len(payloads) == 2
        assert payloads[0].shape == (1, 4, 3, 8, 8)
        assert payloads[1].shape == (1, 2, 1000)

    def test_save_output_unrecognized_raises(self, tmp_path):
        # Unrecognized payloads (e.g. a modular PipelineState) raise instead of being pickled.
        args = Namespace(output=str(tmp_path) + os.sep, fps=24, sampling_rate=None)
        with pytest.raises(ValueError, match="--output-key"):
            _save_output({"not": "media"}, args)


class TestSchemaCommand:
    pretrained_model_name_or_path = "hf-internal-testing/tiny-flux-pipe"

    def test_parse_docstring_args(self):
        docstring = """Description.

        Args:
            prompt (str): The prompt text
                wraps across multiple lines
                for readability.
            steps (int, optional): Steps to run.
        """
        result = _parse_docstring_args(docstring)
        assert result["steps"] == "Steps to run."
        assert "wraps across multiple lines" in result["prompt"]
        assert "\n" not in result["prompt"]

    def test_schema(self, monkeypatch):
        # End-to-end: parse real argv → SchemaCommand.run → capture the emitted payload and
        # verify it contains the pipeline class + at least a `prompt` input parsed from the
        # pipeline's `__call__` signature.
        captured: dict = {}
        monkeypatch.setattr(out, "dict", lambda payload: captured.update(payload))

        parser = ArgumentParser()
        subparsers = parser.add_subparsers()
        SchemaCommand.register_subcommand(subparsers)
        args = parser.parse_args(["schema", "-m", self.pretrained_model_name_or_path])

        out.set_mode(OutputFormat.json)
        args.func(args).run()

        assert captured["pipeline_class"] == "FluxPipeline"
        assert captured["model"] == self.pretrained_model_name_or_path
        input_names = [p["name"] for p in captured["inputs"]]
        assert "prompt" in input_names


class TestCustomBlocksCommand:
    def test_class_discovery(self, tmp_path):
        block_py = tmp_path / "block.py"
        block_py.write_text(
            "class OtherBase:\n    pass\n"
            "class NotABlock(OtherBase):\n    pass\n"
            "class MyBlock(ModularPipelineBlocks):\n    pass\n"
        )
        cmd = CustomBlocksCommand()
        assert cmd._get_class_names(block_py) == [("MyBlock", "ModularPipelineBlocks")]

        broken = tmp_path / "broken.py"
        broken.write_text("class Broken(:\n    pass\n")
        with pytest.raises(ValueError, match="Could not parse"):
            cmd._get_class_names(broken)

    def test_packaging_writes_pipeline_index(self, tmp_path, monkeypatch):
        # The packaged dir must be loadable by `ModularPipeline.from_pretrained` (what
        # `diffusers-cli run` uses), which requires `modular_model_index.json` in addition to
        # the block-level `modular_config.json`.
        block_py = tmp_path / "block.py"
        block_py.write_text(
            "from diffusers.modular_pipelines import ModularPipelineBlocks\n"
            "\n"
            "class MyBlock(ModularPipelineBlocks):\n"
            "    model_name = 'test'\n"
        )
        monkeypatch.chdir(tmp_path)
        CustomBlocksCommand(str(block_py), "MyBlock").run()
        assert (tmp_path / "modular_config.json").exists()
        assert (tmp_path / "modular_model_index.json").exists()


class TestCli:
    def test_toplevel_help_lists_all_commands(self):
        result = subprocess.run(
            ["python", "-m", "diffusers.commands.diffusers_cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        for cmd in AVAILABLE_COMMANDS:
            assert cmd in result.stdout
