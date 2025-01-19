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
import gc
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


class FunctionReference:
    def __init__(self) -> None:
        self.pre_forward = None
        self.post_forward = None
        self.old_forward = None


class HookRegistry:
    def __init__(self, module_ref: torch.nn.Module) -> None:
        super().__init__()

        self.hooks: Dict[str, ModelHook] = {}

        self._module_ref = module_ref
        self._hook_order = []
        self._fn_refs = []

    def register_hook(self, hook: ModelHook, name: str) -> None:
        if name in self.hooks.keys():
            logger.warning(f"Hook with name {name} already exists, replacing it.")

        forward = self._module_ref.forward

        fn_ref = FunctionReference()
        fn_ref.pre_forward = hook.pre_forward
        fn_ref.post_forward = hook.post_forward
        fn_ref.old_forward = forward

        self._module_ref = hook.initialize_hook(self._module_ref)

        def create_new_forward(function_reference: FunctionReference):
            def new_forward(module, *args, **kwargs):
                args, kwargs = function_reference.pre_forward(module, *args, **kwargs)
                output = function_reference.old_forward(*args, **kwargs)
                return function_reference.post_forward(module, output)

            return new_forward

        # if hasattr(hook, "new_forward"):
        #     fn_ref.old_forward = hook.new_forward

        #     def new_forward(module, *args, **kwargs):
        #         args, kwargs = hook.pre_forward(module, *args, **kwargs)
        #         output = rewritten_forward(module, *args, **kwargs)
        #         return hook.post_forward(module, output)
        # else:

        #     def new_forward(module, *args, **kwargs):
        #         args, kwargs = hook.pre_forward(module, *args, **kwargs)
        #         output = forward(*args, **kwargs)
        #         return hook.post_forward(module, output)

        new_forward = create_new_forward(fn_ref)
        self._module_ref.forward = functools.update_wrapper(functools.partial(new_forward, self._module_ref), forward)

        self.hooks[name] = hook
        self._hook_order.append(name)
        self._fn_refs.append(fn_ref)

    def get_hook(self, name: str) -> Optional[ModelHook]:
        return self.hooks.get(name, None)

    def remove_hook(self, name: str, recurse: bool = True) -> None:
        num_hooks = len(self._hook_order)
        if name in self.hooks.keys():
            hook = self.hooks[name]
            index = self._hook_order.index(name)

            fn_ref = self._fn_refs[index]

            if index == num_hooks - 1:
                self._module_ref.forward = fn_ref.old_forward
            else:
                next_fn_ref = self._fn_refs[index + 1]
                next_fn_ref.old_forward = fn_ref.old_forward

            self._module_ref = hook.deinitalize_hook(self._module_ref)
            del self.hooks[name]
            self._hook_order.pop(index)
            self._fn_refs.pop(index)

        if recurse:
            for module_name, module in self._module_ref.named_modules():
                if module_name == "":
                    continue
                if hasattr(module, "_diffusers_hook"):
                    module._diffusers_hook.remove_hook(name, recurse=False)

        gc.collect()

    def reset_stateful_hooks(self, recurse: bool = True) -> None:
        for hook_name in reversed(self._hook_order):
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
        num_hooks = len(self._hook_order)
        hook_repr = ""
        for i, hook_name in enumerate(self._hook_order):
            hook_invocation_index = num_hooks - i - 1
            hook_repr += f"  ({hook_invocation_index}) {hook_name} - ({self.hooks[hook_name].__class__.__name__})"
            if i < len(self._hook_order) - 1:
                hook_repr += "\n"
        return f"HookRegistry(\n{hook_repr}\n)"
