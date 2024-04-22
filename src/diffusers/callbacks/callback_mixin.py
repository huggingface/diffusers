from typing import Any, Dict, Optional

from .base_callback import CallbackHandler, CallbackInput, CallbackType


class CallbackMixin:
    _possible_callback_kwargs = [
        "prompt",
        "negative_prompt",
        "prompt_embeds",
        "negative_prompt_embeds",
        "height",
        "width",
        "image",
        "control_image",
        "ip_adapter_image",
        "ip_adapter_image_embeds",
        "num_inference_steps",
        "guidance_scale",
        "guidance_rescale" "num_images_per_prompt",
        "num_videos_per_prompt",
        "generator",
        "eta",
    ]

    def __init__(self) -> None:
        self._callback_handler = CallbackHandler(
            pipe=self,
            callbacks=[],
            track_callback_results=False,
        )

    def enable_tracking_callback_results(self) -> None:
        self._callback_handler.enable_callback_result_tracking()

    def disable_tracking_callback_results(self) -> None:
        self._callback_handler.disable_callback_result_tracking()

    def add_callback(self, callback: CallbackType) -> None:
        self._callback_handler.add_callback(callback)

    def pop_callback(self, callback: CallbackType) -> Optional[CallbackType]:
        return self.pop_callback(callback)

    def remove_callback(self, callback: CallbackType) -> None:
        self._callback_handler.remove_callback(callback)

    def _get_callback_inputs(self, locals: Dict[str, Any]) -> CallbackInput:
        kwargs = {}
        for possible_kwarg in self._possible_callback_kwargs:
            if possible_kwarg in locals:
                kwargs[possible_kwarg] = locals[possible_kwarg]
        return CallbackInput(**kwargs)
