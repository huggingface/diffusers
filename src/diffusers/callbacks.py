from typing import List

from .configuration_utils import ConfigMixin, register_to_config
from .utils import CONFIG_NAME


class PipelineCallback(ConfigMixin):
    config_name = CONFIG_NAME

    @register_to_config
    def __init__(self, cutoff_step_ratio=1.0):
        super().__init__()

        if not isinstance(cutoff_step_ratio, float) or not (0.0 <= cutoff_step_ratio <= 1.0):
            raise ValueError("cutoff_step_ratio must be a float between 0.0 and 1.0.")

    @property
    def tensor_inputs(self) -> List:
        raise NotImplementedError(f"You need to set the attribute `tensor_inputs` for {self.__class__}")

    def callback_fn(self, pipeline, step_index, timesteps, callback_kwargs):
        raise NotImplementedError(f"You need to implement the method `callback_fn` for {self.__class__}")

    def __call__(self, pipeline, step_index, timestep, callback_kwargs):
        return self.callback_fn(pipeline, step_index, timestep, callback_kwargs)


class MultiPipelineCallbacks:
    def __init__(self, callbacks: List[PipelineCallback]):
        self.callbacks = callbacks

    @property
    def tensor_inputs(self) -> List[str]:
        return [input for callback in self.callbacks for input in callback.tensor_inputs]

    def __call__(self, pipeline, step_index, timestep, callback_kwargs):
        for callback in self.callbacks:
            callback_kwargs = callback(pipeline, step_index, timestep, callback_kwargs)

        return callback_kwargs


class SDCFGCutoffCallback(PipelineCallback):
    tensor_inputs = ["prompt_embeds"]

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs):
        cutoff_step_ratio = self.config.cutoff_step_ratio

        if step_index == int(pipeline.num_timesteps * cutoff_step_ratio):
            prompt_embeds = callback_kwargs[self.tensor_inputs[0]]
            prompt_embeds = prompt_embeds[-1:]

            pipeline._guidance_scale = 0.0

            callback_kwargs[self.tensor_inputs[0]] = prompt_embeds
        return callback_kwargs


class SDXLCFGCutoffCallback(PipelineCallback):
    tensor_inputs = ["prompt_embeds", "add_text_embeds", "add_time_ids"]

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs):
        cutoff_step_ratio = self.config.cutoff_step_ratio

        if step_index == int(pipeline.num_timesteps * cutoff_step_ratio):
            prompt_embeds = callback_kwargs[self.tensor_inputs[0]]
            prompt_embeds = prompt_embeds[-1:]

            add_text_embeds = callback_kwargs[self.tensor_inputs[1]]
            add_text_embeds = add_text_embeds[-1:]

            add_time_ids = callback_kwargs[self.tensor_inputs[2]]
            add_time_ids = add_time_ids[-1:]

            pipeline._guidance_scale = 0.0

            callback_kwargs[self.tensor_inputs[0]] = prompt_embeds
            callback_kwargs[self.tensor_inputs[1]] = add_text_embeds
            callback_kwargs[self.tensor_inputs[2]] = add_time_ids
        return callback_kwargs


class IPAdapterScaleCutoffCallback(PipelineCallback):
    tensor_inputs = []

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs):
        cutoff_step_ratio = self.config.cutoff_step_ratio

        if step_index == int(pipeline.num_timesteps * cutoff_step_ratio):
            pipeline.set_ip_adapter_scale(0.0)
        return callback_kwargs
