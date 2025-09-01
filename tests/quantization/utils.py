from diffusers.utils import is_torch_available

from ..testing_utils import (
    backend_empty_cache,
    backend_max_memory_allocated,
    backend_reset_peak_memory_stats,
    torch_device,
)


if is_torch_available():
    import torch
    import torch.nn as nn

    class LoRALayer(nn.Module):
        """Wraps a linear layer with LoRA-like adapter - Used for testing purposes only

        Taken from
        https://github.com/huggingface/transformers/blob/566302686a71de14125717dea9a6a45b24d42b37/tests/quantization/bnb/test_4bit.py#L62C5-L78C77
        """

        def __init__(self, module: nn.Module, rank: int):
            super().__init__()
            self.module = module
            self.adapter = nn.Sequential(
                nn.Linear(module.in_features, rank, bias=False),
                nn.Linear(rank, module.out_features, bias=False),
            )
            small_std = (2.0 / (5 * min(module.in_features, module.out_features))) ** 0.5
            nn.init.normal_(self.adapter[0].weight, std=small_std)
            nn.init.zeros_(self.adapter[1].weight)
            self.adapter.to(module.weight.device)

        def forward(self, input, *args, **kwargs):
            return self.module(input, *args, **kwargs) + self.adapter(input)

    @torch.no_grad()
    @torch.inference_mode()
    def get_memory_consumption_stat(model, inputs):
        backend_reset_peak_memory_stats(torch_device)
        backend_empty_cache(torch_device)

        model(**inputs)
        max_mem_allocated = backend_max_memory_allocated(torch_device)
        return max_mem_allocated
