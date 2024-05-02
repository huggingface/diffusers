from functools import partial
from typing import Dict, List


class PipelineCallback:
    def __init__(self, **kwargs):
        self.func = partial(self.callback_fn, **kwargs)

    @property
    def tensor_inputs(self) -> List:
        raise NotImplementedError(f"You need to set the attribute `tensor_inputs` for {self.__class__}")

    def callback_fn(self, pipeline, step_index, timesteps, callback_kwargs, **kwargs):
        pass

    def __call__(self, pipeline, step_index, timestep, callback_kwargs):
        return self.func(pipeline, step_index, timestep, callback_kwargs)


class MultiPipelineCallbacks:
    def __init__(self, callbacks: List[PipelineCallback]):
        self.callbacks = callbacks

    @property
    def tensor_inputs(self) -> List[str]:
        return [input for callback in self.callbacks for input in callback.tensor_inputs]

    def __call__(self, pipeline, step_index, timestep, callback_kwargs):
        return {
            key: value
            for callback in self.callbacks
            for key, value in callback(pipeline, step_index, timestep, callback_kwargs).items()
        }


class SDCFGCutoutCallback(PipelineCallback):
    tensor_inputs = ["prompt_embeds"]

    def __init__(self, step_ratio: float = 1.0):
        if not isinstance(step_ratio, float) or not (0.0 <= step_ratio <= 1.0):
            raise ValueError("step_ratio must be a float between 0.0 and 1.0.")

        super().__init__()
        self.step_ratio = step_ratio

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs):
        if step_index == int(pipeline.num_timesteps * self.step_ratio):
            prompt_embeds = callback_kwargs[self.tensor_inputs[0]]
            prompt_embeds = prompt_embeds[-1:]

            pipeline._guidance_scale = 0.0

            callback_kwargs[self.tensor_inputs[0]] = prompt_embeds
        return callback_kwargs


class SDXLCFGCutoutCallback(PipelineCallback):
    tensor_inputs = ["prompt_embeds", "add_text_embeds", "add_time_ids"]

    def __init__(self, step_ratio: int = 1):
        if not isinstance(step_ratio, float) or not (0.0 <= step_ratio <= 1.0):
            raise ValueError("step_ratio must be a float between 0.0 and 1.0.")

        super().__init__()
        self.step_ratio = step_ratio

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs):
        if step_index == int(pipeline.num_timesteps * self.step_ratio):
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


class IPAdapterScaleCutoutCallback(PipelineCallback):
    tensor_inputs = []

    def __init__(self, step_ratio: int = 1):
        if not isinstance(step_ratio, float) or not (0.0 <= step_ratio <= 1.0):
            raise ValueError("step_ratio must be a float between 0.0 and 1.0.")

        super().__init__()
        self.step_ratio = step_ratio

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs):
        if step_index == int(pipeline.num_timesteps * self.step_ratio):
            pipeline.set_ip_adapter_scale(0.0)
        return callback_kwargs


class StepwiseIPAdapterScalerCallback(PipelineCallback):
    tensor_inputs = []

    def __init__(self, steps_scales: Dict):
        if not all(0.0 <= step <= 1.0 for step in steps_scales.keys()):
            raise ValueError("All steps must be a float between 0.0 and 1.0.")

        super().__init__()
        self.steps_scales = steps_scales

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs):
        for step, scale in self.steps_scales.items():
            if step_index == int(pipeline.num_timesteps * step):
                pipeline.set_ip_adapter_scale(scale)
        return callback_kwargs
