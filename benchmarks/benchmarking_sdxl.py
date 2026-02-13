from functools import partial

import torch
from benchmarking_utils import BenchmarkMixin, BenchmarkScenario, model_init_fn

from diffusers import UNet2DConditionModel
from diffusers.utils.testing_utils import torch_device


CKPT_ID = "stabilityai/stable-diffusion-xl-base-1.0"
RESULT_FILENAME = "sdxl.csv"


def get_input_dict(**device_dtype_kwargs):
    # height: 1024
    # width: 1024
    # max_sequence_length: 77
    hidden_states = torch.randn(1, 4, 128, 128, **device_dtype_kwargs)
    encoder_hidden_states = torch.randn(1, 77, 2048, **device_dtype_kwargs)
    timestep = torch.tensor([1.0], **device_dtype_kwargs)
    added_cond_kwargs = {
        "text_embeds": torch.randn(1, 1280, **device_dtype_kwargs),
        "time_ids": torch.ones(1, 6, **device_dtype_kwargs),
    }

    return {
        "sample": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "added_cond_kwargs": added_cond_kwargs,
    }


if __name__ == "__main__":
    scenarios = [
        BenchmarkScenario(
            name=f"{CKPT_ID}-bf16",
            model_cls=UNet2DConditionModel,
            model_init_kwargs={
                "pretrained_model_name_or_path": CKPT_ID,
                "torch_dtype": torch.bfloat16,
                "subfolder": "unet",
            },
            get_model_input_dict=partial(get_input_dict, device=torch_device, dtype=torch.bfloat16),
            model_init_fn=model_init_fn,
            compile_kwargs={"fullgraph": True},
        ),
        BenchmarkScenario(
            name=f"{CKPT_ID}-layerwise-upcasting",
            model_cls=UNet2DConditionModel,
            model_init_kwargs={
                "pretrained_model_name_or_path": CKPT_ID,
                "torch_dtype": torch.bfloat16,
                "subfolder": "unet",
            },
            get_model_input_dict=partial(get_input_dict, device=torch_device, dtype=torch.bfloat16),
            model_init_fn=partial(model_init_fn, layerwise_upcasting=True),
        ),
        BenchmarkScenario(
            name=f"{CKPT_ID}-group-offload-leaf",
            model_cls=UNet2DConditionModel,
            model_init_kwargs={
                "pretrained_model_name_or_path": CKPT_ID,
                "torch_dtype": torch.bfloat16,
                "subfolder": "unet",
            },
            get_model_input_dict=partial(get_input_dict, device=torch_device, dtype=torch.bfloat16),
            model_init_fn=partial(
                model_init_fn,
                group_offload_kwargs={
                    "onload_device": torch_device,
                    "offload_device": torch.device("cpu"),
                    "offload_type": "leaf_level",
                    "use_stream": True,
                    "non_blocking": True,
                },
            ),
        ),
    ]

    runner = BenchmarkMixin()
    runner.run_bencmarks_and_collate(scenarios, filename=RESULT_FILENAME)
