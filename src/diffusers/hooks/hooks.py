# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from ..utils.torch_utils import unwrap_module


logger = get_logger(__name__)  # pylint: disable=invalid-name


class BaseState:
    def reset(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "BaseState::reset is not implemented. Please implement this method in the derived class."
        )


class StateManager:
    def __init__(self, state_cls: BaseState, init_args=None, init_kwargs=None):
        self._state_cls = state_cls
        self._init_args = init_args if init_args is not None else ()
        self._init_kwargs = init_kwargs if init_kwargs is not None else {}
        self._state_cache = {}
        self._current_context = None

    def get_state(self):
        if self._current_context is None:
            raise ValueError("No context is set. Please set a context before retrieving the state.")
        if self._current_context not in self._state_cache.keys():
            self._state_cache[self._current_context] = self._state_cls(*self._init_args, **self._init_kwargs)
        return self._state_cache[self._current_context]

    def set_context(self, name: str) -> None:
        self._current_context = name

    def reset(self, *args, **kwargs) -> None:
        for name, state in list(self._state_cache.items()):
            state.reset(*args, **kwargs)
            self._state_cache.pop(name)
        self._current_context = None


class ModelHook:
    r"""
    A hook that contains callbacks to be executed just before and after the forward method of a model.
    """

    _is_stateful = False

    def __init__(self):
        self.fn_ref: "HookFunctionReference" = None

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
        Hook that is executed when a model is deinitialized.

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

    def _set_context(self, module: torch.nn.Module, name: str) -> None:
        # Iterate over all attributes of the hook to see if any of them have the type `StateManager`. If so, call `set_context` on them.
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, StateManager):
                attr.set_context(name)
        return module


class HookFunctionReference:
    def __init__(self) -> None:
        """A container class that maintains mutable references to forward pass functions in a hook chain.

        Its mutable nature allows the hook system to modify the execution chain dynamically without rebuilding the
        entire forward pass structure.

        Attributes:
            pre_forward: A callable that processes inputs before the main forward pass.
            post_forward: A callable that processes outputs after the main forward pass.
            forward: The current forward function in the hook chain.
            original_forward: The original forward function, stored when a hook provides a custom new_forward.

        The class enables hook removal by allowing updates to the forward chain through reference modification rather
        than requiring reconstruction of the entire chain. When a hook is removed, only the relevant references need to
        be updated, preserving the execution order of the remaining hooks.
        """
        self.pre_forward = None
        self.post_forward = None
        self.forward = None
        self.original_forward = None


class HookRegistry:
    def __init__(self, module_ref: torch.nn.Module) -> None:
        super().__init__()

        self.hooks: Dict[str, ModelHook] = {}

        self._module_ref = module_ref
        self._hook_order = []
        self._fn_refs = []

    def register_hook(self, hook: ModelHook, name: str) -> None:
        if name in self.hooks.keys():
            raise ValueError(
                f"Hook with name {name} already exists in the registry. Please use a different name or "
                f"first remove the existing hook and then add a new one."
            )

        self._module_ref = hook.initialize_hook(self._module_ref)

        def create_new_forward(function_reference: HookFunctionReference):
            def new_forward(module, *args, **kwargs):
                args, kwargs = function_reference.pre_forward(module, *args, **kwargs)
                output = function_reference.forward(*args, **kwargs)
                return function_reference.post_forward(module, output)

            return new_forward

        forward = self._module_ref.forward

        fn_ref = HookFunctionReference()
        fn_ref.pre_forward = hook.pre_forward
        fn_ref.post_forward = hook.post_forward
        fn_ref.forward = forward

        if hasattr(hook, "new_forward"):
            fn_ref.original_forward = forward
            fn_ref.forward = functools.update_wrapper(
                functools.partial(hook.new_forward, self._module_ref), hook.new_forward
            )

        rewritten_forward = create_new_forward(fn_ref)
        self._module_ref.forward = functools.update_wrapper(
            functools.partial(rewritten_forward, self._module_ref), rewritten_forward
        )

        hook.fn_ref = fn_ref
        self.hooks[name] = hook
        self._hook_order.append(name)
        self._fn_refs.append(fn_ref)

    def get_hook(self, name: str) -> Optional[ModelHook]:
        return self.hooks.get(name, None)

    def remove_hook(self, name: str, recurse: bool = True) -> None:
        if name in self.hooks.keys():
            num_hooks = len(self._hook_order)
            hook = self.hooks[name]
            index = self._hook_order.index(name)
            fn_ref = self._fn_refs[index]

            old_forward = fn_ref.forward
            if fn_ref.original_forward is not None:
                old_forward = fn_ref.original_forward

            if index == num_hooks - 1:
                self._module_ref.forward = old_forward
            else:
                self._fn_refs[index + 1].forward = old_forward

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

    def reset_stateful_hooks(self, recurse: bool = True) -> None:
        for hook_name in reversed(self._hook_order):
            hook = self.hooks[hook_name]
            if hook._is_stateful:
                hook.reset_state(self._module_ref)

        if recurse:
            for module_name, module in unwrap_module(self._module_ref).named_modules():
                if module_name == "":
                    continue
                module = unwrap_module(module)
                if hasattr(module, "_diffusers_hook"):
                    module._diffusers_hook.reset_stateful_hooks(recurse=False)

    @classmethod
    def check_if_exists_or_initialize(cls, module: torch.nn.Module) -> "HookRegistry":
        if not hasattr(module, "_diffusers_hook"):
            module._diffusers_hook = cls(module)
        return module._diffusers_hook

    def _set_context(self, name: Optional[str] = None) -> None:
        for hook_name in reversed(self._hook_order):
            hook = self.hooks[hook_name]
            if hook._is_stateful:
                hook._set_context(self._module_ref, name)

        for module_name, module in unwrap_module(self._module_ref).named_modules():
            if module_name == "":
                continue
            module = unwrap_module(module)
            if hasattr(module, "_diffusers_hook"):
                module._diffusers_hook._set_context(name)

    def __repr__(self) -> str:
        registry_repr = ""
        for i, hook_name in enumerate(self._hook_order):
            if self.hooks[hook_name].__class__.__repr__ is not object.__repr__:
                hook_repr = self.hooks[hook_name].__repr__()
            else:
                hook_repr = self.hooks[hook_name].__class__.__name__
            registry_repr += f"  ({i}) {hook_name} - {hook_repr}"
            if i < len(self._hook_order) - 1:
                registry_repr += "\n"
        return f"HookRegistry(\n{registry_repr}\n)"
