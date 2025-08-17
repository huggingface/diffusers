


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps



def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

    return latents

class QwenImagePrepareLatentsStep(PipelineBlock):

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Prepare latents step that prepares the latents for the text-to-image generation process"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="height"),
            InputParam(name="width"),
            InputParam(name="latents"),
            InputParam(name="num_images_per_prompt", default=1),
        ]
    
    @property
    def intermediate_inputs(self) -> List[InputParam]:
        return [
            InputParam(
                name="batch_size",
                required=True,
                type_hint=int,
                description="Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be generated in input step.",
            ),
            InputParam(name="generator"),
            InputParam(name="dtype", type_hint=torch.dtype, description="The dtype of the model inputs"),
        ]
    
    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(name="latents", type_hint=torch.Tensor, description="The initial latents to use for the denoising process"),
        ]

    
    def check_inputs(self, height, width, components):

        if height is not None and height % (components.vae_scale_factor * 2) != 0:
            raise ValueError(f"Height must be divisible by {components.vae_scale_factor * 2} but is {height}")

        if width is not None and width % (components.vae_scale_factor * 2) != 0:
            raise ValueError(f"Width must be divisible by {components.vae_scale_factor * 2} but is {width}")
    
    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline.prepare_latents with self->components
    def prepare_latents(
        components,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (components.vae_scale_factor * 2))
        width = 2 * (int(width) // (components.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = pack_latents(latents, batch_size, num_channels_latents, height, width)

        return latents


    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:

        block_state = self.get_block_state(state)

        device = components._execution_device
        dtype = block_state.dtype 

        height = block_state.height or components.default_height
        width = block_state.width or components.default_width
        final_batch_size = block_state.batch_size * block_state.num_images_per_prompt

        latents = self.prepare_latents(
            components=components,
            batch_size=final_batch_size,
            num_channels_latents=components.num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=block_state.generator)
        
        self.set_block_state(state, block_state)

        return components, state



class QwenImageSetTimestepsStep(PipelineBlock):

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that sets the the scheduler's timesteps for inference"
    
    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(name="scheduler", FlowMatchEulerDiscreteScheduler),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="num_inference_steps", default=50),
            InputParam(name="sigmas"),
        ]

    @property
    def intermediate_inputs(self) -> List[InputParam]:
        return [
            InputParam(name="latents", required=True, type_hint=torch.Tensor, description="The latents to use for the denoising process"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(name="timesteps", type_hint=torch.Tensor, description="The timesteps to use for the denoising process"),
            OutputParam(name="num_inference_steps", type_hint=int, description="The number of inference steps to use for the denoising process"),
        ]
    
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device

        sigmas = np.linspace(1.0, 1 / block_state.num_inference_steps, block_state.num_inference_steps) if block_state.sigmas is None else block_state.sigmas
        
        mu = calculate_shift(
            image_seq_len=block_state.latents.shape[1],
            base_seq_len= components.scheduler.config.get("base_image_seq_len", 256),
            max_seq_len= components.scheduler.config.get("max_image_seq_len", 4096),
            base_shift= components.scheduler.config.get("base_shift", 0.5),
            max_shift= components.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler=components.scheduler,
            num_inference_steps=block_state.num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        self.set_block_state(state, block_state)

        return components, state
        

class QwenImagePrepareAdditionalConditioningStep(PipelineBlock):

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Step that prepares the additional conditioning for the text-to-image generation process"
    
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:

        block_state = self.get_block_state(state)

        height = block_state.height or components.default_height
        width = block_state.width or components.default_width

        block_state.img_shapes = [(1, height // components.vae_scale_factor // 2, width // components.vae_scale_factor // 2)] * block_state.final_batch_size
        image_seq_len = block_state.latents.shape[1]
        txt_seq_lens = block_state.prompt_embeds_mask.sum(dim=1).tolist() if block_state.prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            block_state.negative_prompt_embeds_mask.sum(dim=1).tolist() if block_state.negative_prompt_embeds_mask is not None else None
        )


        self.set_block_state(state, block_state)

        return components, state