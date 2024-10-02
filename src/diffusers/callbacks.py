from typing import Any, Dict, List

from .configuration_utils import ConfigMixin, register_to_config
from .utils import CONFIG_NAME


class PipelineCallback(ConfigMixin):
    """
    Base class for all the official callbacks used in a pipeline. This class provides a structure for implementing
    custom callbacks and ensures that all callbacks have a consistent interface.

    Please implement the following:
        `tensor_inputs`: This should return a list of tensor inputs specific to your callback. You will only be able to
        include
            variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.
        `callback_fn`: This method defines the core functionality of your callback.
    """

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(self, cutoff_step_ratio=1.0, cutoff_step_index=None):
        super().__init__()

        if (cutoff_step_ratio is None and cutoff_step_index is None) or (
            cutoff_step_ratio is not None and cutoff_step_index is not None
        ):
            raise ValueError("Either cutoff_step_ratio or cutoff_step_index should be provided, not both or none.")

        if cutoff_step_ratio is not None and (
            not isinstance(cutoff_step_ratio, float) or not (0.0 <= cutoff_step_ratio <= 1.0)
        ):
            raise ValueError("cutoff_step_ratio must be a float between 0.0 and 1.0.")

    @property
    def tensor_inputs(self) -> List[str]:
        raise NotImplementedError(f"You need to set the attribute `tensor_inputs` for {self.__class__}")

    def callback_fn(self, pipeline, step_index, timesteps, callback_kwargs) -> Dict[str, Any]:
        raise NotImplementedError(f"You need to implement the method `callback_fn` for {self.__class__}")

    def __call__(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        return self.callback_fn(pipeline, step_index, timestep, callback_kwargs)


class MultiPipelineCallbacks:
    """
    This class is designed to handle multiple pipeline callbacks. It accepts a list of PipelineCallback objects and
    provides a unified interface for calling all of them.
    """

    def __init__(self, callbacks: List[PipelineCallback]):
        self.callbacks = callbacks

    @property
    def tensor_inputs(self) -> List[str]:
        return [input for callback in self.callbacks for input in callback.tensor_inputs]

    def __call__(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        """
        Calls all the callbacks in order with the given arguments and returns the final callback_kwargs.
        """
        for callback in self.callbacks:
            callback_kwargs = callback(pipeline, step_index, timestep, callback_kwargs)

        return callback_kwargs


class SDCFGCutoffCallback(PipelineCallback):
    """
    Callback function for Stable Diffusion Pipelines. After certain number of steps (set by `cutoff_step_ratio` or
    `cutoff_step_index`), this callback will disable the CFG.

    Note: This callback mutates the pipeline by changing the `_guidance_scale` attribute to 0.0 after the cutoff step.
    """

    tensor_inputs = ["prompt_embeds"]

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        cutoff_step_ratio = self.config.cutoff_step_ratio
        cutoff_step_index = self.config.cutoff_step_index

        # Use cutoff_step_index if it's not None, otherwise use cutoff_step_ratio
        cutoff_step = (
            cutoff_step_index if cutoff_step_index is not None else int(pipeline.num_timesteps * cutoff_step_ratio)
        )

        if step_index == cutoff_step:
            prompt_embeds = callback_kwargs[self.tensor_inputs[0]]
            prompt_embeds = prompt_embeds[-1:]  # "-1" denotes the embeddings for conditional text tokens.

            pipeline._guidance_scale = 0.0

            callback_kwargs[self.tensor_inputs[0]] = prompt_embeds
        return callback_kwargs


class SDXLCFGCutoffCallback(PipelineCallback):
    """
    Callback function for the base Stable Diffusion XL Pipelines. After certain number of steps (set by
    `cutoff_step_ratio` or `cutoff_step_index`), this callback will disable the CFG.

    Note: This callback mutates the pipeline by changing the `_guidance_scale` attribute to 0.0 after the cutoff step.
    """

    tensor_inputs = [
        "prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
    ]

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        cutoff_step_ratio = self.config.cutoff_step_ratio
        cutoff_step_index = self.config.cutoff_step_index

        # Use cutoff_step_index if it's not None, otherwise use cutoff_step_ratio
        cutoff_step = (
            cutoff_step_index if cutoff_step_index is not None else int(pipeline.num_timesteps * cutoff_step_ratio)
        )

        if step_index == cutoff_step:
            prompt_embeds = callback_kwargs[self.tensor_inputs[0]]
            prompt_embeds = prompt_embeds[-1:]  # "-1" denotes the embeddings for conditional text tokens.

            add_text_embeds = callback_kwargs[self.tensor_inputs[1]]
            add_text_embeds = add_text_embeds[-1:]  # "-1" denotes the embeddings for conditional pooled text tokens

            add_time_ids = callback_kwargs[self.tensor_inputs[2]]
            add_time_ids = add_time_ids[-1:]  # "-1" denotes the embeddings for conditional added time vector

            pipeline._guidance_scale = 0.0

            callback_kwargs[self.tensor_inputs[0]] = prompt_embeds
            callback_kwargs[self.tensor_inputs[1]] = add_text_embeds
            callback_kwargs[self.tensor_inputs[2]] = add_time_ids

        return callback_kwargs


class SDXLControlnetCFGCutoffCallback(PipelineCallback):
    """
    Callback function for the Controlnet Stable Diffusion XL Pipelines. After certain number of steps (set by
    `cutoff_step_ratio` or `cutoff_step_index`), this callback will disable the CFG.

    Note: This callback mutates the pipeline by changing the `_guidance_scale` attribute to 0.0 after the cutoff step.
    """

    tensor_inputs = [
        "prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "image",
    ]

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        cutoff_step_ratio = self.config.cutoff_step_ratio
        cutoff_step_index = self.config.cutoff_step_index

        # Use cutoff_step_index if it's not None, otherwise use cutoff_step_ratio
        cutoff_step = (
            cutoff_step_index if cutoff_step_index is not None else int(pipeline.num_timesteps * cutoff_step_ratio)
        )

        if step_index == cutoff_step:
            prompt_embeds = callback_kwargs[self.tensor_inputs[0]]
            prompt_embeds = prompt_embeds[-1:]  # "-1" denotes the embeddings for conditional text tokens.

            add_text_embeds = callback_kwargs[self.tensor_inputs[1]]
            add_text_embeds = add_text_embeds[-1:]  # "-1" denotes the embeddings for conditional pooled text tokens

            add_time_ids = callback_kwargs[self.tensor_inputs[2]]
            add_time_ids = add_time_ids[-1:]  # "-1" denotes the embeddings for conditional added time vector

            # For Controlnet
            image = callback_kwargs[self.tensor_inputs[3]]
            image = image[-1:]

            pipeline._guidance_scale = 0.0

            callback_kwargs[self.tensor_inputs[0]] = prompt_embeds
            callback_kwargs[self.tensor_inputs[1]] = add_text_embeds
            callback_kwargs[self.tensor_inputs[2]] = add_time_ids
            callback_kwargs[self.tensor_inputs[3]] = image

        return callback_kwargs


class IPAdapterScaleCutoffCallback(PipelineCallback):
    """
    Callback function for any pipeline that inherits `IPAdapterMixin`. After certain number of steps (set by
    `cutoff_step_ratio` or `cutoff_step_index`), this callback will set the IP Adapter scale to `0.0`.

    Note: This callback mutates the IP Adapter attention processors by setting the scale to 0.0 after the cutoff step.
    """

    tensor_inputs = []

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        cutoff_step_ratio = self.config.cutoff_step_ratio
        cutoff_step_index = self.config.cutoff_step_index

        # Use cutoff_step_index if it's not None, otherwise use cutoff_step_ratio
        cutoff_step = (
            cutoff_step_index if cutoff_step_index is not None else int(pipeline.num_timesteps * cutoff_step_ratio)
        )

        if step_index == cutoff_step:
            pipeline.set_ip_adapter_scale(0.0)
        return callback_kwargs
