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

import subprocess
from argparse import ArgumentParser, Namespace

import pytest

from diffusers.commands.custom_blocks import CustomBlocksCommand
from diffusers.commands.run import (
    _build_task_kwargs,
    _kwargs_to_argv,
    _load_pipeline,
    _parse_pipeline_kwargs,
    _resolve_dtype,
    _resolve_media_inputs,
    _upload_inputs_to_sandbox,
)
from diffusers.commands.schema import _parse_docstring_args
from diffusers.utils.testing_utils import require_accelerator, require_torch_gpu


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
        import json as _json

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
            pipeline_kwargs=_json.dumps(
                {
                    "image": str(local_a),  # scalar local → uploaded
                    "mask_image": [str(local_a), "https://example.com/x.png", str(local_b)],  # mixed batch
                    "prompt": "unchanged",
                }
            )
        )
        _upload_inputs_to_sandbox(args, FakeSbx(), run_id="rid")

        assert len(uploads) == 3  # a (scalar), a (list[0]), b (list[2]); URL skipped
        rewritten = _json.loads(args.pipeline_kwargs)
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
        from diffusers.commands.run import RunCommand

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
        # `_compile_denoiser` either applies `torch.compile` (which wraps the module in an
        # `OptimizedModule` with an `_orig_mod` attribute), or calls `compile_repeated_blocks`
        # which mutates the transformer's inner blocks in place. Either path leaves
        # `_orig_mod` on either the transformer itself or on one of its repeated blocks.
        has_torch_compile_wrapper = hasattr(pipeline.transformer, "_orig_mod") or any(
            hasattr(m, "_orig_mod") for m in pipeline.transformer.modules()
        )
        assert has_torch_compile_wrapper

    @require_torch_gpu
    def test_attention_backend_arg(self):
        args = self._parse_run_argv(["--attention-backend", "flash_hub"])
        pipeline = _load_pipeline(args)
        # `set_attention_backend` stamps each attention processor's `_attention_backend` attr.
        backends = {
            m.processor._attention_backend
            for m in pipeline.transformer.modules()
            if hasattr(m, "processor") and hasattr(m.processor, "_attention_backend")
        }
        assert any(b.value == "flash_hub" for b in backends)


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
        from huggingface_hub.cli._output import OutputFormat, out

        from diffusers.commands.schema import SchemaCommand

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
