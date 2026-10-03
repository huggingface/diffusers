import copy
import threading
from typing import Any, Iterable, List, Optional

import torch

from diffusers.utils import logging

from .scheduler import BaseAsyncScheduler, async_retrieve_timesteps
from .wrappers import ThreadSafeImageProcessorWrapper, ThreadSafeTokenizerWrapper, ThreadSafeVAEWrapper


logger = logging.get_logger(__name__)


class RequestScopedPipeline:
    DEFAULT_MUTABLE_ATTRS = [
        "_all_hooks",
        "_offload_device",
        "_progress_bar_config",
        "_progress_bar",
        "_rng_state",
        "_last_seed",
        "latents",
    ]

    def __init__(
        self,
        pipeline: Any,
        mutable_attrs: Optional[Iterable[str]] = None,
        auto_detect_mutables: bool = True,
        tensor_numel_threshold: int = 1_000_000,
        tokenizer_lock: Optional[threading.Lock] = None,
        wrap_scheduler: bool = True,
    ):
        self._base = pipeline

        self.unet = getattr(pipeline, "unet", None)
        self.vae = getattr(pipeline, "vae", None)
        self.text_encoder = getattr(pipeline, "text_encoder", None)
        self.components = getattr(pipeline, "components", None)

        self.transformer = getattr(pipeline, "transformer", None)

        if wrap_scheduler and hasattr(pipeline, "scheduler") and pipeline.scheduler is not None:
            if not isinstance(pipeline.scheduler, BaseAsyncScheduler):
                pipeline.scheduler = BaseAsyncScheduler(pipeline.scheduler)

        self._mutable_attrs = list(mutable_attrs) if mutable_attrs is not None else list(self.DEFAULT_MUTABLE_ATTRS)

        self._tokenizer_lock = tokenizer_lock if tokenizer_lock is not None else threading.Lock()

        self._vae_lock = threading.Lock()
        self._image_lock = threading.Lock()

        self._auto_detect_mutables = bool(auto_detect_mutables)
        self._tensor_numel_threshold = int(tensor_numel_threshold)
        self._auto_detected_attrs: List[str] = []

    def _detect_kernel_pipeline(self, pipeline) -> bool:
        kernel_indicators = [
            "text_encoding_cache",
            "memory_manager",
            "enable_optimizations",
            "_create_request_context",
            "get_optimization_stats",
        ]

        return any(hasattr(pipeline, attr) for attr in kernel_indicators)

    def _make_local_scheduler(self, num_inference_steps: int, device: str | None = None, **clone_kwargs):
        base_sched = getattr(self._base, "scheduler", None)
        if base_sched is None:
            return None

        if not isinstance(base_sched, BaseAsyncScheduler):
            wrapped_scheduler = BaseAsyncScheduler(base_sched)
        else:
            wrapped_scheduler = base_sched

        try:
            return wrapped_scheduler.clone_for_request(
                num_inference_steps=num_inference_steps, device=device, **clone_kwargs
            )
        except Exception as e:
            logger.debug(f"clone_for_request failed: {e}; trying shallow copy fallback")
            try:
                if hasattr(wrapped_scheduler, "scheduler"):
                    try:
                        copied_scheduler = copy.copy(wrapped_scheduler.scheduler)
                        return BaseAsyncScheduler(copied_scheduler)
                    except Exception:
                        return wrapped_scheduler
                else:
                    copied_scheduler = copy.copy(wrapped_scheduler)
                    return BaseAsyncScheduler(copied_scheduler)
            except Exception as e2:
                logger.warning(
                    f"Shallow copy of scheduler also failed: {e2}. Using original scheduler (*thread-unsafe but functional*)."
                )
                return wrapped_scheduler

    def _autodetect_mutables(self, max_attrs: int = 40):
        if not self._auto_detect_mutables:
            return []

        if self._auto_detected_attrs:
            return self._auto_detected_attrs

        candidates: List[str] = []
        seen = set()

        for name in dir(self._base):
            if name.startswith("__"):
                continue
            if name in self._mutable_attrs:
                continue
            if name in ("to", "save_pretrained", "from_pretrained"):
                continue

            try:
                val = getattr(self._base, name)
            except Exception:
                continue

            import types

            if callable(val) or isinstance(val, (types.ModuleType, types.FunctionType, types.MethodType)):
                continue

            if isinstance(val, (dict, list, set, tuple, bytearray)):
                candidates.append(name)
                seen.add(name)
            else:
                # try Tensor detection
                try:
                    if isinstance(val, torch.Tensor):
                        if val.numel() <= self._tensor_numel_threshold:
                            candidates.append(name)
                            seen.add(name)
                        else:
                            logger.debug(f"Ignoring large tensor attr '{name}', numel={val.numel()}")
                except Exception:
                    continue

            if len(candidates) >= max_attrs:
                break

        self._auto_detected_attrs = candidates
        logger.debug(f"Autodetected mutable attrs to clone: {self._auto_detected_attrs}")
        return self._auto_detected_attrs

    def _is_readonly_property(self, base_obj, attr_name: str) -> bool:
        try:
            cls = type(base_obj)
            descriptor = getattr(cls, attr_name, None)
            if isinstance(descriptor, property):
                return descriptor.fset is None
            if hasattr(descriptor, "__set__") is False and descriptor is not None:
                return False
        except Exception:
            pass
        return False

    def _clone_mutable_attrs(self, base, local):
        attrs_to_clone = list(self._mutable_attrs)
        attrs_to_clone.extend(self._autodetect_mutables())

        EXCLUDE_ATTRS = {
            "components",
        }

        for attr in attrs_to_clone:
            if attr in EXCLUDE_ATTRS:
                logger.debug(f"Skipping excluded attr '{attr}'")
                continue
            if not hasattr(base, attr):
                continue
            if self._is_readonly_property(base, attr):
                logger.debug(f"Skipping read-only property '{attr}'")
                continue

            try:
                val = getattr(base, attr)
            except Exception as e:
                logger.debug(f"Could not getattr('{attr}') on base pipeline: {e}")
                continue

            try:
                if isinstance(val, dict):
                    setattr(local, attr, dict(val))
                elif isinstance(val, (list, tuple, set)):
                    setattr(local, attr, list(val))
                elif isinstance(val, bytearray):
                    setattr(local, attr, bytearray(val))
                else:
                    # small tensors or atomic values
                    if isinstance(val, torch.Tensor):
                        if val.numel() <= self._tensor_numel_threshold:
                            setattr(local, attr, val.clone())
                        else:
                            # don't clone big tensors, keep reference
                            setattr(local, attr, val)
                    else:
                        try:
                            setattr(local, attr, copy.copy(val))
                        except Exception:
                            setattr(local, attr, val)
            except (AttributeError, TypeError) as e:
                logger.debug(f"Skipping cloning attribute '{attr}' because it is not settable: {e}")
                continue
            except Exception as e:
                logger.debug(f"Unexpected error cloning attribute '{attr}': {e}")
                continue

    def _is_tokenizer_component(self, component) -> bool:
        if component is None:
            return False

        tokenizer_methods = ["encode", "decode", "tokenize", "__call__"]
        has_tokenizer_methods = any(hasattr(component, method) for method in tokenizer_methods)

        class_name = component.__class__.__name__.lower()
        has_tokenizer_in_name = "tokenizer" in class_name

        tokenizer_attrs = ["vocab_size", "pad_token", "eos_token", "bos_token"]
        has_tokenizer_attrs = any(hasattr(component, attr) for attr in tokenizer_attrs)

        return has_tokenizer_methods and (has_tokenizer_in_name or has_tokenizer_attrs)

    def _should_wrap_tokenizers(self) -> bool:
        return True

    def generate(self, *args, num_inference_steps: int = 50, device: str | None = None, **kwargs):
        local_scheduler = self._make_local_scheduler(num_inference_steps=num_inference_steps, device=device)

        try:
            local_pipe = copy.copy(self._base)
        except Exception as e:
            logger.warning(f"copy.copy(self._base) failed: {e}. Falling back to deepcopy (may increase memory).")
            local_pipe = copy.deepcopy(self._base)

        try:
            if (
                hasattr(local_pipe, "vae")
                and local_pipe.vae is not None
                and not isinstance(local_pipe.vae, ThreadSafeVAEWrapper)
            ):
                local_pipe.vae = ThreadSafeVAEWrapper(local_pipe.vae, self._vae_lock)

            if (
                hasattr(local_pipe, "image_processor")
                and local_pipe.image_processor is not None
                and not isinstance(local_pipe.image_processor, ThreadSafeImageProcessorWrapper)
            ):
                local_pipe.image_processor = ThreadSafeImageProcessorWrapper(
                    local_pipe.image_processor, self._image_lock
                )
        except Exception as e:
            logger.debug(f"Could not wrap vae/image_processor: {e}")

        if local_scheduler is not None:
            try:
                timesteps, num_steps, configured_scheduler = async_retrieve_timesteps(
                    local_scheduler.scheduler,
                    num_inference_steps=num_inference_steps,
                    device=device,
                    return_scheduler=True,
                    **{k: v for k, v in kwargs.items() if k in ["timesteps", "sigmas"]},
                )

                final_scheduler = BaseAsyncScheduler(configured_scheduler)
                setattr(local_pipe, "scheduler", final_scheduler)
            except Exception:
                logger.warning("Could not set scheduler on local pipe; proceeding without replacing scheduler.")

        self._clone_mutable_attrs(self._base, local_pipe)

        original_tokenizers = {}

        if self._should_wrap_tokenizers():
            try:
                for name in dir(local_pipe):
                    if "tokenizer" in name and not name.startswith("_"):
                        tok = getattr(local_pipe, name, None)
                        if tok is not None and self._is_tokenizer_component(tok):
                            if not isinstance(tok, ThreadSafeTokenizerWrapper):
                                original_tokenizers[name] = tok
                                wrapped_tokenizer = ThreadSafeTokenizerWrapper(tok, self._tokenizer_lock)
                                setattr(local_pipe, name, wrapped_tokenizer)

                if hasattr(local_pipe, "components") and isinstance(local_pipe.components, dict):
                    for key, val in local_pipe.components.items():
                        if val is None:
                            continue

                        if self._is_tokenizer_component(val):
                            if not isinstance(val, ThreadSafeTokenizerWrapper):
                                original_tokenizers[f"components[{key}]"] = val
                                wrapped_tokenizer = ThreadSafeTokenizerWrapper(val, self._tokenizer_lock)
                                local_pipe.components[key] = wrapped_tokenizer

            except Exception as e:
                logger.debug(f"Tokenizer wrapping step encountered an error: {e}")

        result = None
        cm = getattr(local_pipe, "model_cpu_offload_context", None)

        try:
            if callable(cm):
                try:
                    with cm():
                        result = local_pipe(*args, num_inference_steps=num_inference_steps, **kwargs)
                except TypeError:
                    try:
                        with cm:
                            result = local_pipe(*args, num_inference_steps=num_inference_steps, **kwargs)
                    except Exception as e:
                        logger.debug(f"model_cpu_offload_context usage failed: {e}. Proceeding without it.")
                        result = local_pipe(*args, num_inference_steps=num_inference_steps, **kwargs)
            else:
                result = local_pipe(*args, num_inference_steps=num_inference_steps, **kwargs)

            return result

        finally:
            try:
                for name, tok in original_tokenizers.items():
                    if name.startswith("components["):
                        key = name[len("components[") : -1]
                        if hasattr(local_pipe, "components") and isinstance(local_pipe.components, dict):
                            local_pipe.components[key] = tok
                    else:
                        setattr(local_pipe, name, tok)
            except Exception as e:
                logger.debug(f"Error restoring original tokenizers: {e}")
