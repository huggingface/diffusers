import argparse
import gc
import importlib
import json
import os
import sys
import traceback
import yaml
from pathlib import Path
from datetime import datetime, timezone

import torch
from diffusers import DiffusionPipeline

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from runner_utils import detect_device, timer, validate_image, compare_with_reference


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_image(image, output_dir: str, filename: str) -> str:
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    filepath = os.path.join(output_dir, "images", filename)
    if hasattr(image, "save"):
        image.save(filepath)
    elif isinstance(image, list) and len(image) > 0 and hasattr(image[0], "save"):
        image = image[0]
        image.save(filepath)
    else:
        from PIL import Image
        Image.fromarray(image).save(filepath)
    return filepath


def load_pipeline(pipeline_class_name: str, module_path: str, weight_path: str, model_id: str, backend: str, device: str, torch_dtype: torch.dtype):
    module = importlib.import_module(module_path)
    pipeline_cls = getattr(module, pipeline_class_name)

    if backend == "local" and weight_path and os.path.isdir(weight_path):
        pipe = pipeline_cls.from_pretrained(
            weight_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    elif backend == "modelscope":
        from modelscope import snapshot_download
        local_path = snapshot_download(model_id or weight_path)
        pipe = pipeline_cls.from_pretrained(
            local_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    elif backend == "hf":
        pipe = pipeline_cls.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    else:
        pipe = pipeline_cls.from_pretrained(
            weight_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    pipe = pipe.to(device)
    return pipe


def apply_optimizations(pipe, parallel: str):
    if parallel == "single":
        if hasattr(pipe, "enable_attention_slicing"):
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        if hasattr(pipe, "enable_vae_slicing"):
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass


def build_kwargs(params_grid_entry: dict, prompt: str, negative_prompt: str, device: str, extra_params: dict | None = None) -> dict:
    grid_copy = dict(params_grid_entry)
    parallel = grid_copy.pop("parallel", "single")
    name = grid_copy.pop("name", "unnamed")

    kwargs = {k: v for k, v in grid_copy.items()}
    kwargs["prompt"] = prompt
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    if extra_params:
        kwargs.update(extra_params)

    if "image" in kwargs and isinstance(kwargs["image"], str):
        from PIL import Image
        image_path = kwargs["image"]
        if os.path.isfile(image_path):
            kwargs["image"] = Image.open(image_path).convert("RGB")
        else:
            del kwargs["image"]

    gen_device = "cuda" if device == "cuda" else "cpu"
    kwargs["generator"] = torch.Generator(device=gen_device).manual_seed(42)

    return kwargs, parallel, name


def run_single_config(pipe, kwargs: dict, parallel: str, name: str,
                      pipeline: str, variant_name: str, device: str, dtype_str: str,
                      output_dir: str, ref_image_path: str | None = None) -> dict:
    result = {
        "pipeline": pipeline,
        "variant": variant_name,
        "config_name": name,
        "parallel": parallel,
        "device": device,
        "dtype": dtype_str,
        "params": {k: v for k, v in kwargs.items()
                   if k not in ("prompt", "negative_prompt", "generator", "image")},
        "result": None,
        "error": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        with timer() as get_elapsed:
            output = pipe(**kwargs)

        inference_time = round(get_elapsed(), 2)
        image = output.images[0]

        filename = f"{pipeline}_{variant_name}_{name}.png"
        out_path = save_image(image, output_dir, filename)

        width = kwargs.get("width", 0)
        height = kwargs.get("height", 0)
        validation = validate_image(image, width, height)

        precision = {}
        if ref_image_path and os.path.isfile(ref_image_path):
            precision = compare_with_reference(image, ref_image_path)
        elif ref_image_path:
            precision = {"error": f"reference not found: {ref_image_path}"}

        result["result"] = {
            "status": "passed",
            "inference_time_s": inference_time,
            "output_file": os.path.relpath(out_path, output_dir),
            "validation": validation,
            "precision": precision,
        }
    except Exception as e:
        result["result"] = {"status": "failed"}
        result["error"] = "".join(traceback.format_exception(type(e), e, e.__traceback__))

    return result


def generate_reference_only(config_dir: str, output_dir: str):
    config = load_yaml(os.path.join(config_dir, "config.yaml"))
    ref_config_name = config.get("reference_config")
    if not ref_config_name:
        print(f"[SKIP] {config_dir}: no reference_config defined")
        return

    ref_entry = None
    for entry in config["params_grid"]:
        if entry.get("name") == ref_config_name:
            ref_entry = entry
            break
    if not ref_entry:
        print(f"[SKIP] {config_dir}: reference_config '{ref_config_name}' not found in params_grid")
        return

    pipeline_name = os.path.basename(config_dir)
    pipeline_class = config["pipeline_class"]
    module_path = config["module"]
    prompt = config.get("prompt", "")

    variants_dir = os.path.join(config_dir, "variants")
    variant_files = sorted(Path(variants_dir).glob("*.yaml"))

    pipeline_label = os.path.basename(config_dir).replace("/", "_")

    if not variant_files:
        print(f"[SKIP] {pipeline_name}: no variant files")
        return

    variant_data = load_yaml(str(variant_files[0]))
    device, torch_dtype = detect_device()

    print(f"[{pipeline_name}] loading model ({variant_files[0].stem})...")
    pipe = load_pipeline(
        pipeline_class, module_path,
        variant_data.get("weight_path", variant_data["model_id"]),
        variant_data["model_id"],
        variant_data.get("backend", "local"),
        device, torch_dtype,
    )
    apply_optimizations(pipe, ref_entry.get("parallel", "single"))

    kwargs, _, _ = build_kwargs(ref_entry, prompt, config.get("negative_prompt", ""), device,
                                 variant_data.get("extra_params"))
    output = pipe(**kwargs)
    image = output.images[0]

    ci_runners_dir = os.path.dirname(os.path.dirname(config_dir))
    ref_dir = os.path.join(ci_runners_dir, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    ref_path = os.path.join(ref_dir, f"{pipeline_label}_{ref_config_name}.png")
    image.save(ref_path)
    print(f"[{pipeline_name}] reference saved to {ref_path}")

    del pipe
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "npu":
        torch.npu.empty_cache()


def scan_pipelines(configs_root: str) -> list[str]:
    pipeline_dirs = []
    for entry in sorted(os.listdir(configs_root)):
        full_path = os.path.join(configs_root, entry)
        if os.path.isdir(full_path) and os.path.isfile(os.path.join(full_path, "config.yaml")):
            pipeline_dirs.append(full_path)
    return pipeline_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True, help="path to configs/ directory")
    parser.add_argument("--output", required=True, help="output directory for results")
    parser.add_argument("--generate-reference", action="store_true",
                        help="generate reference images instead of running full CI")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.generate_reference:
        for pipe_dir in scan_pipelines(args.configs):
            generate_reference_only(pipe_dir, args.output)
        return

    all_results = []
    device, torch_dtype = detect_device()
    dtype_str = str(torch_dtype).split(".")[-1]

    for pipe_dir in scan_pipelines(args.configs):
        config = load_yaml(os.path.join(pipe_dir, "config.yaml"))
        pipeline_class = config["pipeline_class"]
        module_path = config["module"]
        prompt = config.get("prompt", "")
        negative_prompt = config.get("negative_prompt", "")
        ref_config_name = config.get("reference_config")
        pipeline_label = os.path.basename(pipe_dir).replace("/", "_")

        variants_dir = os.path.join(pipe_dir, "variants")
        variant_files = sorted(Path(variants_dir).glob("*.yaml"))

        if not variant_files:
            print(f"[WARN] {pipeline_label}: no variant files found, skipping")
            continue

        for vf in variant_files:
            variant_data = load_yaml(str(vf))
            variant_name = vf.stem

            print(f"[{pipeline_class}] loading model: {variant_name}")
            try:
                pipe = load_pipeline(
                    pipeline_class, module_path,
                    variant_data.get("weight_path", variant_data["model_id"]),
                    variant_data["model_id"],
                    variant_data.get("backend", "local"),
                    device, torch_dtype,
                )
            except Exception as e:
                result = {
                    "pipeline": pipeline_class,
                    "variant": variant_name,
                    "config_name": "model_load",
                    "parallel": "N/A",
                    "device": device,
                    "dtype": dtype_str,
                    "params": {},
                    "result": {"status": "failed"},
                    "error": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                all_results.append(result)
                continue

            for params_entry in config["params_grid"]:
                kwargs, parallel, config_name = build_kwargs(
                    params_entry, prompt, negative_prompt, device,
                    variant_data.get("extra_params"),
                )
                apply_optimizations(pipe, parallel)

                ref_image_path = None
                if config_name == ref_config_name:
                    ci_runners_dir = os.path.dirname(os.path.dirname(pipe_dir))
                    ref_image_path = os.path.join(
                        ci_runners_dir, "reference",
                        f"{pipeline_label}_{ref_config_name}.png"
                    )

                print(f"  [{config_name}] parallel={parallel} ...")
                result = run_single_config(
                    pipe, kwargs, parallel, config_name,
                    pipeline_class, variant_name, device, dtype_str,
                    args.output, ref_image_path,
                )
                all_results.append(result)

                status = result["result"].get("status", "unknown")
                if status == "passed":
                    t = result["result"].get("inference_time_s", "?")
                    print(f"    PASS ({t}s)")
                else:
                    print(f"    FAIL")

            del pipe
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "npu":
                torch.npu.empty_cache()

    results_path = os.path.join(args.output, "all_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    passed = sum(1 for r in all_results if r["result"] and r["result"].get("status") == "passed")
    failed = sum(1 for r in all_results if r["result"] and r["result"].get("status") == "failed")
    print(f"\nDone. {len(all_results)} runs: {passed} passed, {failed} failed.")
    print(f"Results written to {results_path}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
