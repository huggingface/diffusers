from functools import partial

import torch
from benchmarking_utils import BenchmarkMixin, BenchmarkScenario, model_init_fn

from diffusers import WanTransformer3DModel
from diffusers.utils.testing_utils import torch_device


CKPT_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
RESULT_FILENAME = "wan.csv"


def get_input_dict(**device_dtype_kwargs):
    # height: 480
    # width: 832
    # num_frames: 81
    # max_sequence_length: 512
    hidden_states = torch.randn(1, 16, 21, 60, 104, **device_dtype_kwargs)
    encoder_hidden_states = torch.randn(1, 512, 4096, **device_dtype_kwargs)
    timestep = torch.tensor([1.0], **device_dtype_kwargs)

    return {"hidden_states": hidden_states, "encoder_hidden_states": encoder_hidden_states, "timestep": timestep}


if __name__ == "__main__":
    scenarios = [
        BenchmarkScenario(
            name=f"{CKPT_ID}-bf16",
            model_cls=WanTransformer3DModel,
            model_init_kwargs={
                "pretrained_model_name_or_path": CKPT_ID,
                "torch_dtype": torch.bfloat16,
                "subfolder": "transformer",
            },
            get_model_input_dict=partial(get_input_dict, device=torch_device, dtype=torch.bfloat16),
            model_init_fn=model_init_fn,
            compile_kwargs={"fullgraph": True},
        ),
        BenchmarkScenario(
            name=f"{CKPT_ID}-layerwise-upcasting",
            model_cls=WanTransformer3DModel,
            model_init_kwargs={
                "pretrained_model_name_or_path": CKPT_ID,
                "torch_dtype": torch.bfloat16,
                "subfolder": "transformer",
            },
            get_model_input_dict=partial(get_input_dict, device=torch_device, dtype=torch.bfloat16),
            model_init_fn=partial(model_init_fn, layerwise_upcasting=True),
        ),
        BenchmarkScenario(
            name=f"{CKPT_ID}-group-offload-leaf",
            model_cls=WanTransformer3DModel,
            model_init_kwargs={
                "pretrained_model_name_or_path": CKPT_ID,
                "torch_dtype": torch.bfloat16,
                "subfolder": "transformer",
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
