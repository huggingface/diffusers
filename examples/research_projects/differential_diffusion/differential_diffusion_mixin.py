import inspect
from typing import Any, Dict

import torch


def _get_default_value(func, arg_name):
    return inspect.signature(func).parameters[arg_name].default


class DifferentialDiffusionMixin:
    def __init__(self):
        if not hasattr(self, "prepare_latents"):
            raise ValueError("`prepare_latents` must be implemented in the model class.")

        prepare_latents_possible_kwargs = inspect.signature(self.prepare_latents).parameters.keys()
        prepare_latents_required_kwargs = [
            "image",
            "num_inference_steps",
            "num_images_per_prompt",
            "generator",
            "denoising_start",
        ]

        if not all(kwarg in prepare_latents_possible_kwargs for kwarg in prepare_latents_required_kwargs):
            raise ValueError(f"`prepare_latents` must have the following arguments: {prepare_latents_required_kwargs}")

    @torch.no_grad()
    def __call__(self, map: torch.FloatTensor, **kwargs):
        if map is None:
            raise ValueError("`map` must be provided for differential diffusion.")

        original_with_noise = thresholds = masks = None
        original_callback_on_step_end = kwargs.pop("callback_on_step_end", None)
        original_callback_on_step_end_tensor_inputs = kwargs.pop("callback_on_step_end_tensor_inputs", [])

        callback_on_step_end_tensor_inputs_required = ["timesteps", "batch_size", "prompt_embeds", "device", "latents"]
        callback_on_step_end_tensor_inputs = list(
            set(callback_on_step_end_tensor_inputs_required + original_callback_on_step_end_tensor_inputs)
        )

        image = kwargs.pop("image", _get_default_value(self.__call__, "image"))
        num_inference_steps = kwargs.pop(
            "num_inference_steps", _get_default_value(self.__call__, "num_inference_steps")
        )
        num_images_per_prompt = kwargs.pop(
            "num_images_per_prompt", _get_default_value(self.__call__, "num_images_per_prompt")
        )
        generator = kwargs.pop("generator", _get_default_value(self.__call__, "generator"))
        denoising_start = kwargs.pop("denoising_start", _get_default_value(self.__call__, "denoising_start"))

        def callback(pipe, i: int, t: int, callback_kwargs: Dict[str, Any]):
            nonlocal original_with_noise, thresholds, masks

            timesteps = callback_kwargs.get("timesteps")
            batch_size = callback_kwargs.get("batch_size")
            prompt_embeds = callback_kwargs.get("prompt_embeds")
            latents = callback_kwargs.get("latents")

            if i < 0:
                original_with_noise = self.prepare_latents(
                    image=image,
                    timestep=timesteps,
                    batch_size=batch_size,
                    num_images_per_prompt=num_images_per_prompt,
                    dtype=prompt_embeds.dtype,
                    device=prompt_embeds.device,
                    generator=generator,
                )
                thresholds = torch.arange(num_inference_steps, dtype=map.dtype) / num_inference_steps
                thresholds = thresholds.unsqueeze(1).unsqueeze(1).to(prompt_embeds.device)
                masks = map > (thresholds + (denoising_start or 0))

                if denoising_start is None:
                    latents = original_with_noise[:1]
            elif i == 0:
                pass
            else:
                mask = masks[i].unsqueeze(0)
                mask = mask.to(latents.dtype)
                mask = mask.unsqueeze(1)
                latents = original_with_noise[i] * mask + latents * (1 - mask)

            callback_results = {}

            if original_callback_on_step_end is not None:
                callback_kwargs["latents"] = latents
                result = original_callback_on_step_end(pipe, i, t, callback_kwargs)
                callback_results.update(result)

                if "latents" in result:
                    latents = result["latents"]

            callback_results["latents"] = latents

            return callback_results

        return super().__call__(
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            callback_before_step_begin=True,
            **kwargs,
        )
