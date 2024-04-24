def callback_sd_cfg_cutout(pipeline, step_index, timestep, callback_kwargs, step_ratio):
    if step_index == int(pipeline.num_timesteps * step_ratio):
        prompt_embeds = callback_kwargs["prompt_embeds"]
        prompt_embeds = prompt_embeds[-1:]

        pipeline._guidance_scale = 0.0
        callback_kwargs["prompt_embeds"] = prompt_embeds
    return callback_kwargs


def callback_sdxl_cfg_cutout(pipeline, step_index, timestep, callback_kwargs, step_ratio):
    if step_index == int(pipeline.num_timesteps * step_ratio):
        prompt_embeds = callback_kwargs["prompt_embeds"]
        prompt_embeds = prompt_embeds[-1:]

        add_text_embeds = callback_kwargs["add_text_embeds"]
        add_text_embeds = add_text_embeds[-1:]

        add_time_ids = callback_kwargs["add_time_ids"]
        add_time_ids = add_time_ids[-1:]

        pipeline._guidance_scale = 0.0

        callback_kwargs["prompt_embeds"] = prompt_embeds
        callback_kwargs["add_text_embeds"] = add_text_embeds
        callback_kwargs["add_time_ids"] = add_time_ids
    return callback_kwargs


def callback_ip_scale_cutout(pipeline, step_index, timestep, callback_kwargs, step_ratio):
    if step_index == int(pipeline.num_timesteps * step_ratio):
        pipeline.set_ip_adapter_scale(0.0)
    return callback_kwargs


class Callback:
    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def __call__(self, pipeline, step_index, timestep, callback_kwargs):
        return self.func(pipeline, step_index, timestep, callback_kwargs, **self.kwargs)
