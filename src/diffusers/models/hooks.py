# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import functools
from typing import Any, Callable, Dict, Tuple

import torch


# Reference: https://github.com/huggingface/accelerate/blob/ba7ab93f5e688466ea56908ea3b056fae2f9a023/src/accelerate/hooks.py
class ModelHook:
    r"""
    A hook that contains callbacks to be executed just before and after the forward method of a model. The difference
    with PyTorch existing hooks is that they get passed along the kwargs.
    """

    def init_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        r"""
        Hook that is executed when a model is initialized.
        Args:
            module (`torch.nn.Module`):
                The module attached to this hook.
        """
        return module

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs) -> Tuple[Tuple[Any], Dict[str, Any]]:
        r"""
        Hook that is executed just before the forward method of the model.
        Args:
            module (`torch.nn.Module`):
                The module whose forward pass will be executed just after this event.
            args (`Tuple[Any]`):
                The positional arguments passed to the module.
            kwargs (`Dict[Str, Any]`):
                The keyword arguments passed to the module.
        Returns:
            `Tuple[Tuple[Any], Dict[Str, Any]]`:
                A tuple with the treated `args` and `kwargs`.
        """
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output: Any) -> Any:
        r"""
        Hook that is executed just after the forward method of the model.
        Args:
            module (`torch.nn.Module`):
                The module whose forward pass been executed just before this event.
            output (`Any`):
                The output of the module.
        Returns:
            `Any`: The processed `output`.
        """
        return output

    def detach_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        r"""
        Hook that is executed when the hook is detached from a module.
        Args:
            module (`torch.nn.Module`):
                The module detached from this hook.
        """
        return module

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        return module


class SequentialHook(ModelHook):
    r"""A hook that can contain several hooks and iterates through them at each event."""

    def __init__(self, *hooks):
        self.hooks = hooks

    def init_hook(self, module):
        for hook in self.hooks:
            module = hook.init_hook(module)
        return module

    def pre_forward(self, module, *args, **kwargs):
        for hook in self.hooks:
            args, kwargs = hook.pre_forward(module, *args, **kwargs)
        return args, kwargs

    def post_forward(self, module, output):
        for hook in self.hooks:
            output = hook.post_forward(module, output)
        return output

    def detach_hook(self, module):
        for hook in self.hooks:
            module = hook.detach_hook(module)
        return module

    def reset_state(self, module):
        for hook in self.hooks:
            module = hook.reset_state(module)
        return module


class FasterCacheHook(ModelHook):
    def __init__(
        self,
        skip_callback: Callable[[torch.nn.Module], bool],
    ) -> None:
        super().__init__()

        self.skip_callback = skip_callback

        self.cache = None
        self._iteration = 0

    def new_forward(self, module: torch.nn.Module, *args, **kwargs) -> Any:
        args, kwargs = module._diffusers_hook.pre_forward(module, *args, **kwargs)

        if self.cache is not None and self.skip_callback(module):
            output = self.cache
        else:
            output = module._old_forward(*args, **kwargs)

        return module._diffusers_hook.post_forward(module, output)

    def post_forward(self, module: torch.nn.Module, output: Any) -> Any:
        self.cache = output
        return output

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.cache = None
        self._iteration = 0
        return module


def add_hook_to_module(module: torch.nn.Module, hook: ModelHook, append: bool = False):
    r"""
    Adds a hook to a given module. This will rewrite the `forward` method of the module to include the hook, to remove
    this behavior and restore the original `forward` method, use `remove_hook_from_module`.
    <Tip warning={true}>
    If the module already contains a hook, this will replace it with the new hook passed by default. To chain two hooks
    together, pass `append=True`, so it chains the current and new hook into an instance of the `SequentialHook` class.
    </Tip>
    Args:
        module (`torch.nn.Module`):
            The module to attach a hook to.
        hook (`ModelHook`):
            The hook to attach.
        append (`bool`, *optional*, defaults to `False`):
            Whether the hook should be chained with an existing one (if module already contains a hook) or not.
    Returns:
        `torch.nn.Module`:
            The same module, with the hook attached (the module is modified in place, so the result can be discarded).
    """
    original_hook = hook

    if append and getattr(module, "_diffusers_hook", None) is not None:
        old_hook = module._diffusers_hook
        remove_hook_from_module(module)
        hook = SequentialHook(old_hook, hook)

    if hasattr(module, "_diffusers_hook") and hasattr(module, "_old_forward"):
        # If we already put some hook on this module, we replace it with the new one.
        old_forward = module._old_forward
    else:
        old_forward = module.forward
        module._old_forward = old_forward

    module = hook.init_hook(module)
    module._diffusers_hook = hook

    if hasattr(original_hook, "new_forward"):
        new_forward = original_hook.new_forward
    else:

        def new_forward(module, *args, **kwargs):
            args, kwargs = module._diffusers_hook.pre_forward(module, *args, **kwargs)
            output = module._old_forward(*args, **kwargs)
            return module._diffusers_hook.post_forward(module, output)

    # Overriding a GraphModuleImpl forward freezes the forward call and later modifications on the graph will fail.
    # Reference: https://pytorch.slack.com/archives/C3PDTEV8E/p1705929610405409
    if "GraphModuleImpl" in str(type(module)):
        module.__class__.forward = functools.update_wrapper(functools.partial(new_forward, module), old_forward)
    else:
        module.forward = functools.update_wrapper(functools.partial(new_forward, module), old_forward)

    return module


def remove_hook_from_module(module: torch.nn.Module, recurse: bool = False) -> torch.nn.Module:
    """
    Removes any hook attached to a module via `add_hook_to_module`.
    Args:
        module (`torch.nn.Module`):
            The module to attach a hook to.
        recurse (`bool`, defaults to `False`):
            Whether to remove the hooks recursively
    Returns:
        `torch.nn.Module`:
            The same module, with the hook detached (the module is modified in place, so the result can be discarded).
    """

    if hasattr(module, "_diffusers_hook"):
        module._diffusers_hook.detach_hook(module)
        delattr(module, "_diffusers_hook")

    if hasattr(module, "_old_forward"):
        # Overriding a GraphModuleImpl forward freezes the forward call and later modifications on the graph will fail.
        # Reference: https://pytorch.slack.com/archives/C3PDTEV8E/p1705929610405409
        if "GraphModuleImpl" in str(type(module)):
            module.__class__.forward = module._old_forward
        else:
            module.forward = module._old_forward
        delattr(module, "_old_forward")

    if recurse:
        for child in module.children():
            remove_hook_from_module(child, recurse)

    return module
