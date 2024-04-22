from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

import torch

from ..image_processor import PipelineImageInput
from ..pipelines.pipeline_utils import DiffusionPipeline
from ..utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class CallbackInput:
    prompt: Union[str, List[str]] = None
    negative_prompt: Union[str, List[str]] = None
    prompt_embeds: torch.FloatTensor = None
    negative_prompt_embeds: torch.FloatTensor = None
    height: int = None
    width: int = None
    image: PipelineImageInput = None
    control_image: PipelineImageInput = None
    ip_adapter_image: PipelineImageInput = None
    ip_adapter_image_embeds: torch.FloatTensor = None
    num_inference_steps: int = None
    guidance_scale: float = None
    guidance_rescale: float = None
    num_images_per_prompt: int = None
    num_videos_per_prompt: int = None
    generator: torch.Generator = None
    eta: float = None


class BaseCallback:
    r"""
    A class for objects that will inspect the state of the inference loop at some events and take some decisions. At
    each of those events the following arguments are available:
    """

    def on_inference_begin(
        self, pipe: DiffusionPipeline, args: CallbackInput, control: Any, **kwargs: Dict[str, Any]
    ) -> Any:
        r"""
        Event triggered at the beginning of the inference loop.
        """
        pass

    def on_inference_end(
        self, pipe: DiffusionPipeline, args: CallbackInput, control: Any, **kwargs: Dict[str, Any]
    ) -> Any:
        r"""
        Event triggered at the end of the inference loop.
        """
        pass

    def on_step_begin(
        self, pipe: DiffusionPipeline, args: CallbackInput, control: Any, **kwargs: Dict[str, Any]
    ) -> Any:
        r"""
        Event triggered at the beginning of the inference step.
        """
        pass

    def on_step_end(self, pipe: DiffusionPipeline, args: CallbackInput, control: Any, **kwargs: Dict[str, Any]) -> Any:
        r"""
        Event triggered at the end of the inference step.
        """
        pass

    def on_load(self, pipe: DiffusionPipeline, args: CallbackInput, control: Any, **kwargs: Dict[str, Any]) -> Any:
        r"""
        Event triggered when the pipeline is loaded.
        """
        pass

    def on_save(self, pipe: DiffusionPipeline, args: CallbackInput, control: Any, **kwargs: Dict[str, Any]) -> Any:
        r"""
        Event triggered when the pipeline is saved.
        """
        pass


CallbackType = Union[BaseCallback, Type[BaseCallback]]


class CallbackResult:
    r"""
    Class to store the result of a callback. This is used to store the results of the callbacks in the CallbackHandler.
    """

    event: str
    result: Any


class CallbackHandler:
    r"""
    Internal class to handle a list of callbacks. Callbacks are objects that listen to specific events during the
    inference loop and take some decisions thereby influencing the inference process. Callbacks are called in the order
    they are added to the handler.
    """

    def __init__(self, pipe: DiffusionPipeline, callbacks: List[BaseCallback], track_callback_results: bool) -> None:
        self.pipe = pipe
        self.track_callback_results = track_callback_results

        self.callbacks = []
        self._callback_types = set()
        self._callback_results = []

        if not isinstance(callbacks, list):
            callbacks = [callbacks]

        for callback in callbacks:
            self.add_callback(callback)

    def enable_callback_result_tracking(self) -> None:
        self.track_callback_results = True

    def disable_callback_result_tracking(self) -> None:
        self.track_callback_results = False

    def add_callback(self, callback: CallbackType) -> None:
        cb = callback() if isinstance(callback, type) else callback
        cb_class = cb.__class__

        if not isinstance(cb, BaseCallback):
            logger.warning(f"Callback {cb_class} is not an instance of BaseCallback. This is not recommended.")
        if cb_class in self._callback_types:
            logger.warning(
                f"You are adding the callback {cb_class} multiple times. The current list of callbacks is:\n"
                + self.callbacks
            )

        self.callbacks.append(cb)
        self._callback_types.add(cb_class)

    def pop_callback(self, callback: CallbackType) -> Optional[CallbackType]:
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback: CallbackType) -> None:
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_results(self) -> List[CallbackResult]:
        return self._callback_results

    def clear_callback_results(self) -> None:
        self._callback_results = []

    def _call_event(self, event: str, control: Any, **kwargs: Dict[str, Any]) -> Any:
        for callback in self.callbacks:
            cb = getattr(callback, event)
            result = cb(pipe=self.pipe, control=control, **kwargs)
            if self.track_callback_results:
                self._callback_results.append(CallbackResult(event=event, result=result))
            if result is not None:
                control = result
        return control

    def on_inference_begin(self, args: CallbackInput, control: Any = None, **kwargs: Dict[str, Any]) -> Any:
        return self._call_event("on_inference_begin", args=args, control=control, **kwargs)

    def on_inference_end(self, args: CallbackInput, control: Any = None, **kwargs: Dict[str, Any]) -> Any:
        return self._call_event("on_inference_end", args=args, control=control, **kwargs)

    def on_step_begin(self, args: CallbackInput, control: Any = None, **kwargs: Dict[str, Any]) -> Any:
        return self._call_event("on_step_begin", args=args, control=control, **kwargs)

    def on_step_end(self, args: CallbackInput, control: Any = None, **kwargs: Dict[str, Any]) -> Any:
        return self._call_event("on_step_end", args=args, control=control, **kwargs)

    def on_load(self, args: CallbackInput, control: Any = None, **kwargs: Dict[str, Any]) -> Any:
        return self._call_event("on_load", args=args, control=control, **kwargs)

    def on_save(self, args: CallbackInput, control: Any = None, **kwargs: Dict[str, Any]) -> Any:
        return self._call_event("on_save", args=args, control=control, **kwargs)

    def __len__(self) -> int:
        return len(self.callbacks)
