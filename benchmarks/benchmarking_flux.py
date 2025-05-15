import torch
from benchmarking_utils import BenchmarkMixin

from diffusers import FluxTransformer2DModel
from diffusers.utils.testing_utils import torch_device


class BenchmarkFlux(BenchmarkMixin):
    model_class = FluxTransformer2DModel
    compile_kwargs = {"fullgraph": True, "mode": "max-autotune"}

    def get_model_init_dict(self):
        return {
            "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
            "subfolder": "transformer",
            "torch_dtype": torch.bfloat16,
        }

    def initialize_model(self):
        model = self.model_class.from_pretrained(**self.get_model_init_dict())
        model = model.to(torch_device).eval()
        return model

    def get_input_dict(self):
        # resolution: 1024x1024
        # maximum sequence length 512
        hidden_states = torch.randn(1, 4096, 64, device=torch_device, dtype=torch.bfloat16)
        encoder_hidden_states = torch.randn(1, 512, 4096, device=torch_device, dtype=torch.bfloat16)
        pooled_prompt_embeds = torch.randn(1, 768, device=torch_device, dtype=torch.bfloat16)
        image_ids = torch.ones(512, 3, device=torch_device, dtype=torch.bfloat16)
        text_ids = torch.ones(4096, 3, device=torch_device, dtype=torch.bfloat16)
        timestep = torch.tensor([1.0], device=torch_device)
        guidance = torch.tensor([1.0], device=torch_device)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            "pooled_projections": pooled_prompt_embeds,
            "timestep": timestep,
            "guidance": guidance,
        }
