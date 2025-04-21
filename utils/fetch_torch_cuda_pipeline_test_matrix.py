import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi

import diffusers


PATH_TO_REPO = Path(__file__).parent.parent.resolve()
ALWAYS_TEST_PIPELINE_MODULES = [
    "controlnet",
    "controlnet_flux",
    "controlnet_sd3",
    "stable_diffusion",
    "stable_diffusion_2",
    "stable_diffusion_3",
    "stable_diffusion_xl",
    "ip_adapters",
    "flux",
]
PIPELINE_USAGE_CUTOFF = int(os.getenv("PIPELINE_USAGE_CUTOFF", 50000))

logger = logging.getLogger(__name__)
api = HfApi()


def filter_pipelines(usage_dict, usage_cutoff=10000):
    output = []
    for diffusers_object, usage in usage_dict.items():
        if usage < usage_cutoff:
            continue

        is_diffusers_pipeline = hasattr(diffusers.pipelines, diffusers_object)
        if not is_diffusers_pipeline:
            continue

        output.append(diffusers_object)

    return output


def fetch_pipeline_objects():
    models = api.list_models(library="diffusers")
    downloads = defaultdict(int)

    for model in models:
        is_counted = False
        for tag in model.tags:
            if tag.startswith("diffusers:"):
                is_counted = True
                downloads[tag[len("diffusers:") :]] += model.downloads

        if not is_counted:
            downloads["other"] += model.downloads

    # Remove 0 downloads
    downloads = {k: v for k, v in downloads.items() if v > 0}
    pipeline_objects = filter_pipelines(downloads, PIPELINE_USAGE_CUTOFF)

    return pipeline_objects


def fetch_pipeline_modules_to_test():
    try:
        pipeline_objects = fetch_pipeline_objects()
    except Exception as e:
        logger.error(e)
        raise RuntimeError("Unable to fetch model list from HuggingFace Hub.")

    test_modules = []
    for pipeline_name in pipeline_objects:
        module = getattr(diffusers, pipeline_name)

        test_module = module.__module__.split(".")[-2].strip()
        test_modules.append(test_module)

    return test_modules


def main():
    test_modules = fetch_pipeline_modules_to_test()
    test_modules.extend(ALWAYS_TEST_PIPELINE_MODULES)

    # Get unique modules
    test_modules = sorted(set(test_modules))
    print(json.dumps(test_modules))

    save_path = f"{PATH_TO_REPO}/reports"
    os.makedirs(save_path, exist_ok=True)

    with open(f"{save_path}/test-pipelines.json", "w") as f:
        json.dump({"pipeline_test_modules": test_modules}, f)


if __name__ == "__main__":
    main()
