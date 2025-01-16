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
from typing import Any, Dict, Optional, Tuple

import torch

from ..utils.logging import get_logger


logger = get_logger(__name__)  # pylint: disable=invalid-name


class ModelHook:
    r"""
    A hook that contains callbacks to be executed just before and after the forward method of a model.
    """

    _is_stateful = False

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        r"""
        Hook that is executed when a model is initialized.
        Args:
            module (`torch.nn.Module`):
                The module attached to this hook.
        """
        return module

    def deinitalize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        r"""
        Hook that is executed when a model is deinitalized.
        Args:
            module (`torch.nn.Module`):
                The module attached to this hook.
        """
        module.forward = module._old_forward
        del module._old_forward
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

    def reset_state(self, module: torch.nn.Module):
        if self._is_stateful:
            raise NotImplementedError("This hook is stateful and needs to implement the `reset_state` method.")
        return module


class HookRegistry:
    def __init__(self, module_ref: torch.nn.Module) -> None:
        super().__init__()

        self.hooks: Dict[str, ModelHook] = {}

        self._module_ref = module_ref
        self._hook_order = []

    def register_hook(self, hook: ModelHook, name: str) -> None:
        if name in self.hooks.keys():
            logger.warning(f"Hook with name {name} already exists, replacing it.")

        if hasattr(self._module_ref, "_old_forward"):
            old_forward = self._module_ref._old_forward
        else:
            old_forward = self._module_ref.forward
            self._module_ref._old_forward = self._module_ref.forward

        self._module_ref = hook.initialize_hook(self._module_ref)

        if hasattr(hook, "new_forward"):
            rewritten_forward = hook.new_forward

            def new_forward(module, *args, **kwargs):
                args, kwargs = hook.pre_forward(module, *args, **kwargs)
                output = rewritten_forward(module, *args, **kwargs)
                return hook.post_forward(module, output)
        else:

            def new_forward(module, *args, **kwargs):
                args, kwargs = hook.pre_forward(module, *args, **kwargs)
                output = old_forward(*args, **kwargs)
                return hook.post_forward(module, output)

        self._module_ref.forward = functools.update_wrapper(
            functools.partial(new_forward, self._module_ref), old_forward
        )

        self.hooks[name] = hook
        self._hook_order.append(name)

    def get_hook(self, name: str) -> Optional[ModelHook]:
        if name not in self.hooks.keys():
            return None
        return self.hooks[name]

    def remove_hook(self, name: str, recurse: bool = True) -> None:
        if name in self.hooks.keys():
            hook = self.hooks[name]
            self._module_ref = hook.deinitalize_hook(self._module_ref)
            del self.hooks[name]
            self._hook_order.remove(name)

        if recurse:
            for module_name, module in self._module_ref.named_modules():
                if module_name == "":
                    continue
                if hasattr(module, "_diffusers_hook"):
                    module._diffusers_hook.remove_hook(name, recurse=False)

    def reset_stateful_hooks(self, recurse: bool = True) -> None:
        for hook_name in self._hook_order:
            hook = self.hooks[hook_name]
            if hook._is_stateful:
                hook.reset_state(self._module_ref)

        if recurse:
            for module_name, module in self._module_ref.named_modules():
                if module_name == "":
                    continue
                if hasattr(module, "_diffusers_hook"):
                    module._diffusers_hook.reset_stateful_hooks(recurse=False)

    @classmethod
    def check_if_exists_or_initialize(cls, module: torch.nn.Module) -> "HookRegistry":
        if not hasattr(module, "_diffusers_hook"):
            module._diffusers_hook = cls(module)
        return module._diffusers_hook

    def __repr__(self) -> str:
        hook_repr = ""
        for i, hook_name in enumerate(self._hook_order):
            hook_repr += f"  ({i}) {hook_name} - ({self.hooks[hook_name].__class__.__name__})"
            if i < len(self._hook_order) - 1:
                hook_repr += "\n"
        return f"HookRegistry(\n{hook_repr}\n)"
