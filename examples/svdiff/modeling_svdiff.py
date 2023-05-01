from typing import Optional, Union, List
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import inspect
import safetensors
from accelerate.utils import set_module_tensor_to_device


class SVDiffModule(nn.Module):
    def __init__(self, weight: torch.Tensor, weight_type: Optional[str] = None, init_delta: Optional[torch.Tensor] = None, scale: float = 1.0):
        """
        SVDiff module to be registered as hook 

        Parameters:
            weight (`torch.Tensor`, *required*): pre-trained weight 
            weight_type (`string`, *optional*): choose from [None, 'conv', '1d']
            init_delta (`string, *optional*`): init values for delta
            scale (`float`): spectral shifts scale. This is intended to use during inference. 
        """
        super().__init__()
        self.weight_type = weight_type
        weight = self._reshape_weight_for_svd(weight)
        # perform SVD
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        self.register_buffer("U", U.detach())
        self.U.requires_grad = False
        self.register_buffer("S", S.detach())
        self.S.requires_grad = False
        self.register_buffer("Vh", Vh.detach())
        self.Vh.requires_grad = False
        # initialize to 0 for smooth tuning 
        self.delta = nn.Parameter(torch.zeros_like(S)) if init_delta is None else nn.Parameter(init_delta)
        self.scale = scale

    def _reshape_weight_for_svd(self, updated_weight: torch.Tensor, original_weight: torch.Tensor = None, reverse: bool = False, **kwargs):
        if self.weight_type is not None:
            if self.weight_type == "conv":
                if not reverse:
                    return rearrange(updated_weight, 'co cin h w -> co (cin h w)')
                else:
                    return rearrange(updated_weight, 'co (cin h w) -> co cin h w', cin=original_weight.size(1), h=original_weight.size(2), w=original_weight.size(3))
            elif self.weight_type == "1d":
                if not reverse:
                    return updated_weight.unsqueeze(0)
                else:
                    return updated_weight.squeeze(0)
            else:
                raise ValueError(f"`weight_type`={self.weight_type} is invalid!")
        else:
            return updated_weight
        
    def forward(self, weight: torch.Tensor):
        updated_weight = self.U @ torch.diag(F.relu(self.S + self.scale * self.delta)) @ self.Vh
        updated_weight = self._reshape_weight_for_svd(updated_weight, weight, reverse=True)
        return updated_weight


def set_spectral_shifts(model: nn.Module, spectral_shifts_ckpt: str = None, **kwargs):
    # key to module
    svdiff_modules = {}
    for name, module in model.named_modules():
        svdiff_module = None
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            svdiff_module = SVDiffModule(module.weight, weight_type="conv")
        elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.GroupNorm):
            svdiff_module = SVDiffModule(module.weight, weight_type="1d")
        elif isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            svdiff_module = SVDiffModule(module.weight)
        if svdiff_module:
            svdiff_modules[name+".delta"] = svdiff_module
            # register parametrization 
            nn.utils.parametrize.register_parametrization(module, "weight", svdiff_module)
            
    if spectral_shifts_ckpt:
        param_device = "cpu"
        torch_dtype = kwargs["torch_dtype"] if "torch_dtype" in kwargs else None
        with safetensors.safe_open(spectral_shifts_ckpt, framework="pt", device="cpu") as f:
            for key in f.keys():
                accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
                if accepts_dtype:
                    set_module_tensor_to_device(svdiff_modules[key], "delta", param_device, value=f.get_tensor(key), dtype=torch_dtype)
                else:
                    set_module_tensor_to_device(svdiff_modules[key], "delta", param_device, value=f.get_tensor(key))
        print(f"Resumed from {spectral_shifts_ckpt}")
    return model, svdiff_modules


def slerp(val, low, high):
    """ taken from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/4
    """
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


def slerp_tensor(val, low, high):
    shape = low.shape
    res = slerp(val, low.flatten(1), high.flatten(1))
    return res.reshape(shape)


if __name__ == '__main__':
    import os
    from transformers import CLIPTokenizer, CLIPTextModel
    from diffusers import UNet2DConditionModel, StableDiffusionPipeline
    # load pre-trained model
    model_id = "CompVis/stable-diffusion-v1-4"
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # set spectral shifts 
    unet, svdiff_modules = set_spectral_shifts(unet)
    text_encoder, svdiff_modules_te = set_spectral_shifts(text_encoder)
    # put spectral shifts into optimizer
    params_to_optimize = []
    for m in svdiff_modules.values():
        for p in m.parameters():
            params_to_optimize.append(p)
    for m in svdiff_modules_te.values():
        for p in m.parameters():
            params_to_optimize.append(p)
    total_params = sum(p.numel() for p in params_to_optimize)
    print(f"{total_params * 1.e-6:.2f} M trainable params") # 0.28M
    optimizer = torch.optim.AdamW(params_to_optimize, lr=1e-3)

    # dummy forward
    unet.train()
    text_encoder.train()
    for step in range(2):
        bsz = 2
        latents = torch.randn(bsz, 4, 32, 32)
        noisy_latents = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        input_ids = tokenizer("", padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
        input_ids = input_ids.expand(bsz, -1)
        encoder_hidden_states = text_encoder(input_ids)[0]
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = torch.nn.functional.mse_loss(model_pred.float(), noisy_latents.float(), reduction="mean")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # check!
    origin = unet.conv_in.weight
    module = svdiff_modules['conv_in.delta']
    weight = unet.conv_in.parametrizations.weight.original
    updated_weight = module.U @ torch.diag(F.relu(module.S + module.scale * module.delta)) @ module.Vh
    updated_weight = module._reshape_weight_for_svd(updated_weight, weight, reverse=True)
    print(torch.equal(updated_weight, origin)) # True
    print(svdiff_modules["conv_in.delta"].delta) # not zero
    print(svdiff_modules_te["text_model.embeddings.position_embedding.delta"].delta) # not zero

    # save weights
    ckpt_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "checkpoint-x")
    os.makedirs(ckpt_dir, exist_ok=True)
    state_dict = {}
    for name, module in svdiff_modules.items():
        state_dict[name] = module.delta
    safetensors.torch.save_file(state_dict, os.path.join(ckpt_dir, "spectral_shifts.safetensors"))
    state_dict = {}
    for name, module in svdiff_modules_te.items():
        state_dict[name] = module.delta
    safetensors.torch.save_file(state_dict, os.path.join(ckpt_dir, "spectral_shifts_te.safetensors"))

    # inference 
    del unet, text_encoder, svdiff_modules, svdiff_modules_te
    ckpt_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "checkpoint-x")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    unet, _ = set_spectral_shifts(unet, spectral_shifts_ckpt=os.path.join(ckpt_dir, "spectral_shifts.safetensors"))
    text_encoder, _ = set_spectral_shifts(text_encoder, spectral_shifts_ckpt=os.path.join(ckpt_dir, "spectral_shifts_te.safetensors"))
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        unet=unet,
        text_encoder=text_encoder,
    )
    images = pipe("test", num_inference_steps=1).images

