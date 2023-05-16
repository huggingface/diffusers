#
#
# TODO: REMOVE THIS FILE
# This file is intended to be used for initial development of new features.
#
#

import math

import safetensors
import torch
from PIL import Image

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline


# modified from https://github.com/kohya-ss/sd-scripts/blob/ad5f318d066c52e5b27306b399bc87e41f2eef2b/networks/lora.py#L17
class LoRAModule(torch.nn.Module):
    def __init__(self, org_module: torch.nn.Module, lora_dim=4, alpha=1.0, multiplier=1.0):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if alpha is None or alpha == 0:
            self.alpha = self.lora_dim
        else:
            if type(alpha) == torch.Tensor:
                alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
            self.register_buffer("alpha", torch.tensor(alpha))  # Treatable as a constant.

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier

    def forward(self, x):
        scale = self.alpha / self.lora_dim
        return self.multiplier * scale * self.lora_up(self.lora_down(x))


class LoRAModuleContainer(torch.nn.Module):
    def __init__(self, hooks, state_dict, multiplier):
        super().__init__()
        self.multiplier = multiplier

        # Create LoRAModule from state_dict information
        for key, value in state_dict.items():
            if "lora_down" in key:
                lora_name = key.split(".")[0]
                lora_dim = value.size()[0]
                lora_name_alpha = key.split(".")[0] + ".alpha"
                alpha = None
                if lora_name_alpha in state_dict:
                    alpha = state_dict[lora_name_alpha].item()
                if lora_name in hooks:
                    hook = hooks[lora_name]
                    lora_module = LoRAModule(hook.orig_module, lora_dim=lora_dim, alpha=alpha, multiplier=multiplier)
                    self.register_module(lora_name, lora_module)

        # Load whole LoRA weights
        self.load_state_dict(state_dict, strict=False)

        # Register LoRAModule to LoRAHook
        for name, module in self.named_modules():
            if module.__class__.__name__ == "LoRAModule":
                if name in hooks:
                    hook = hooks[name]
                    hook.append_lora(module)

    @property
    def alpha(self):
        return self.multiplier

    @alpha.setter
    def alpha(self, multiplier):
        self.multiplier = multiplier
        for name, module in self.named_modules():
            if module.__class__.__name__ == "LoRAModule":
                module.multiplier = multiplier

    def remove_from_hooks(self, hooks):
        for name, module in self.named_modules():
            if module.__class__.__name__ == "LoRAModule":
                hook = hooks[name]
                hook.remove_lora(module)
                del module


class LoRAHook(torch.nn.Module):
    """
    replaces forward method of the original Linear,
    instead of replacing the original Linear module.
    """

    def __init__(self):
        super().__init__()
        self.lora_modules = []

    def install(self, orig_module):
        assert not hasattr(self, "orig_module")
        self.orig_module = orig_module
        self.orig_forward = self.orig_module.forward
        self.orig_module.forward = self.forward

    def uninstall(self):
        assert hasattr(self, "orig_module")
        self.orig_module.forward = self.orig_forward
        del self.orig_forward
        del self.orig_module

    def append_lora(self, lora_module):
        self.lora_modules.append(lora_module)

    def remove_lora(self, lora_module):
        self.lora_modules.remove(lora_module)

    def forward(self, x):
        if len(self.lora_modules) == 0:
            return self.orig_forward(x)
        lora = torch.sum(torch.stack([lora(x) for lora in self.lora_modules]), dim=0)
        return self.orig_forward(x) + lora


class LoRAHookInjector(object):
    def __init__(self):
        super().__init__()
        self.hooks = {}
        self.device = None
        self.dtype = None

    def _get_target_modules(self, root_module, prefix, target_replace_modules):
        target_modules = []
        for name, module in root_module.named_modules():
            if (
                module.__class__.__name__ in target_replace_modules and "transformer_blocks" not in name
            ):  # to adapt latest diffusers:
                for child_name, child_module in module.named_modules():
                    is_linear = child_module.__class__.__name__ == "Linear"
                    is_conv2d = child_module.__class__.__name__ == "Conv2d"
                    # if is_linear or is_conv2d:
                    if is_linear and not is_conv2d and "ff.net" not in child_name:
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        target_modules.append((lora_name, child_module))
        return target_modules

    def install_hooks(self, pipe):
        """Install LoRAHook to the pipe."""
        assert len(self.hooks) == 0
        # text_encoder_targets = self._get_target_modules(pipe.text_encoder, "lora_te", ["CLIPAttention", "CLIPMLP"])
        # unet_targets = self._get_target_modules(pipe.unet, "lora_unet", ["Transformer2DModel", "Attention"])
        text_encoder_targets = self._get_target_modules(pipe.text_encoder, "lora_te", ["CLIPAttention"])
        unet_targets = self._get_target_modules(pipe.unet, "lora_unet", ["Transformer2DModel"])

        for name, target_module in text_encoder_targets + unet_targets:
            hook = LoRAHook()
            hook.install(target_module)
            self.hooks[name] = hook
            print(name)

        self.device = pipe.device
        self.dtype = pipe.unet.dtype

    def uninstall_hooks(self):
        """Uninstall LoRAHook from the pipe."""
        for k, v in self.hooks.items():
            v.uninstall()
        self.hooks = {}

    def apply_lora(self, filename, alpha=1.0):
        """Load LoRA weights and apply LoRA to the pipe."""
        assert len(self.hooks) != 0
        state_dict = safetensors.torch.load_file(filename)
        container = LoRAModuleContainer(self.hooks, state_dict, alpha)
        container.to(self.device, self.dtype)
        return container

    def remove_lora(self, container):
        """Remove the individual LoRA from the pipe."""
        container.remove_from_hooks(self.hooks)


def install_lora_hook(pipe: DiffusionPipeline):
    """Install LoRAHook to the pipe."""
    assert not hasattr(pipe, "lora_injector")
    assert not hasattr(pipe, "apply_lora")
    assert not hasattr(pipe, "remove_lora")
    injector = LoRAHookInjector()
    injector.install_hooks(pipe)
    pipe.lora_injector = injector
    pipe.apply_lora = injector.apply_lora
    pipe.remove_lora = injector.remove_lora


def uninstall_lora_hook(pipe: DiffusionPipeline):
    """Uninstall LoRAHook from the pipe."""
    pipe.lora_injector.uninstall_hooks()
    del pipe.lora_injector
    del pipe.apply_lora
    del pipe.remove_lora


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


if __name__ == "__main__":
    pipe = StableDiffusionPipeline.from_pretrained(
        "gsdf/Counterfeit-V2.5", torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.enable_xformers_memory_efficient_attention()

    prompt = "masterpeace, best quality, highres, 1girl, at dusk"
    negative_prompt = (
        "(low quality, worst quality:1.4), (bad anatomy), (inaccurate limb:1.2), "
        "bad composition, inaccurate eyes, extra digit, fewer digits, (extra arms:1.2) "
    )
    lora_fn = "../stable-diffusion-study/models/lora/light_and_shadow.safetensors"

    # Without Lora
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=512,
        height=768,
        num_inference_steps=15,
        num_images_per_prompt=4,
        generator=torch.manual_seed(0),
    ).images
    image_grid(images, 1, 4).save("test_orig.png")

    # Hook version (some restricted apply)
    install_lora_hook(pipe)
    pipe.apply_lora(lora_fn)
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=512,
        height=768,
        num_inference_steps=15,
        num_images_per_prompt=4,
        generator=torch.manual_seed(0),
    ).images
    image_grid(images, 1, 4).save("test_lora_hook.png")
    uninstall_lora_hook(pipe)

    # Diffusers dev version
    pipe.load_lora_weights(lora_fn)
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=512,
        height=768,
        num_inference_steps=15,
        num_images_per_prompt=4,
        generator=torch.manual_seed(0),
        # cross_attention_kwargs={"scale": 0.5},  # lora scale
    ).images
    image_grid(images, 1, 4).save("test_lora_dev.png")
