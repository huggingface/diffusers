from functools import partial

import torch
from benchmarking_utils import BenchmarkMixin, BenchmarkScenario, model_init_fn

from diffusers import LTXVideoTransformer3DModel
from diffusers.utils.testing_utils import torch_device


CKPT_ID = "Lightricks/LTX-Video-0.9.7-dev"
RESULT_FILENAME = "ltx.csv"


def get_input_dict(**device_dtype_kwargs):
    # 512x704 (161 frames)
    # `max_sequence_length`: 256
    hidden_states = torch.randn(1, 7392, 128, **device_dtype_kwargs)
    encoder_hidden_states = torch.randn(1, 256, 4096, **device_dtype_kwargs)
    encoder_attention_mask = torch.ones(1, 256, **device_dtype_kwargs)
    timestep = torch.tensor([1.0], **device_dtype_kwargs)
    video_coords = torch.randn(1, 3, 7392, **device_dtype_kwargs)

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "timestep": timestep,
        "video_coords": video_coords,
    }


if __name__ == "__main__":
    scenarios = [
        BenchmarkScenario(
            name=f"{CKPT_ID}-bf16",
            model_cls=LTXVideoTransformer3DModel,
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
            model_cls=LTXVideoTransformer3DModel,
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
            model_cls=LTXVideoTransformer3DModel,
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
