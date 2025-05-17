# Based on stable_diffusion_xl_reference.py and stable_diffusion_controlnet_reference.py

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch

from diffusers import StableDiffusionXLControlNetPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.models import ControlNetModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.utils import PIL_INTERPOLATION, deprecate, logging, replace_example_docstring
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # !pip install opencv-python transformers accelerate
        >>> from diffusers import ControlNetModel, AutoencoderKL
        >>> from diffusers.schedulers import UniPCMultistepScheduler
        >>> from diffusers.utils import load_image
        >>> import numpy as np
        >>> import torch

        >>> import cv2
        >>> from PIL import Image

        >>> # download an image for the Canny controlnet
        >>> canny_image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl_reference_input_cat.jpg"
        ... )

        >>> # download an image for the Reference controlnet
        >>> ref_image = load_image(
        ...     "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
        ... )

        >>> # initialize the models and pipeline
        >>> controlnet_conditioning_scale = 0.5  # recommended for good generalization
        >>> controlnet = ControlNetModel.from_pretrained(
        ...     "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
        ... )
        >>> vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        >>> pipe = StableDiffusionXLControlNetReferencePipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16
        ... ).to("cuda:0")

        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        >>> # get canny image
        >>> image = np.array(canny_image)
        >>> image = cv2.Canny(image, 100, 200)
        >>> image = image[:, :, None]
        >>> image = np.concatenate([image, image, image], axis=2)
        >>> canny_image = Image.fromarray(image)

        >>> # generate image
        >>> image = pipe(
        ...     prompt="a cat",
        ...     num_inference_steps=20,
        ...     controlnet_conditioning_scale=controlnet_conditioning_scale,
        ...     image=canny_image,
        ...     ref_image=ref_image,
        ...     reference_attn=True,
        ...     reference_adain=True
        ...     style_fidelity=1.0,
        ...     generator=torch.Generator("cuda").manual_seed(42)
        ... ).images[0]
        ```
"""


def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


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


class StableDiffusionXLControlNetReferencePipeline(StableDiffusionXLControlNetPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL with ControlNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        text_encoder_2 ([`~transformers.CLIPTextModelWithProjection`]):
            Second frozen text-encoder
            ([laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        tokenizer_2 ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings should always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark](https://github.com/ShieldMnt/invisible-watermark/) library to
            watermark output images. If not defined, it defaults to `True` if the package is installed; otherwise no
            watermarker is used.
    """

    def prepare_ref_latents(self, refimage, batch_size, dtype, device, generator, do_classifier_free_guidance):
        refimage = refimage.to(device=device)
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.upcast_vae()
            refimage = refimage.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        if refimage.dtype != self.vae.dtype:
            refimage = refimage.to(dtype=self.vae.dtype)
        # encode the mask image into latents space so we can concatenate it to the latents
        if isinstance(generator, list):
            ref_image_latents = [
                self.vae.encode(refimage[i : i + 1]).latent_dist.sample(generator=generator[i])
                for i in range(batch_size)
            ]
            ref_image_latents = torch.cat(ref_image_latents, dim=0)
        else:
            ref_image_latents = self.vae.encode(refimage).latent_dist.sample(generator=generator)
        ref_image_latents = self.vae.config.scaling_factor * ref_image_latents

        # duplicate mask and ref_image_latents for each generation per prompt, using mps friendly method
        if ref_image_latents.shape[0] < batch_size:
            if not batch_size % ref_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {ref_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            ref_image_latents = ref_image_latents.repeat(batch_size // ref_image_latents.shape[0], 1, 1, 1)

        ref_image_latents = torch.cat([ref_image_latents] * 2) if do_classifier_free_guidance else ref_image_latents

        # aligning device to prevent device errors when concating it with the latent model input
        ref_image_latents = ref_image_latents.to(device=device, dtype=dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        return ref_image_latents

    def prepare_ref_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = (image - 0.5) / 0.5
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)

            elif isinstance(image[0], torch.Tensor):
                image = torch.stack(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def check_ref_inputs(
        self,
        ref_image,
        reference_guidance_start,
        reference_guidance_end,
        style_fidelity,
        reference_attn,
        reference_adain,
    ):
        ref_image_is_pil = isinstance(ref_image, PIL.Image.Image)
        ref_image_is_tensor = isinstance(ref_image, torch.Tensor)

        if not ref_image_is_pil and not ref_image_is_tensor:
            raise TypeError(
                f"ref image must be passed and be one of PIL image or torch tensor, but is {type(ref_image)}"
            )

        if not reference_attn and not reference_adain:
            raise ValueError("`reference_attn` or `reference_adain` must be True.")

        if style_fidelity < 0.0:
            raise ValueError(f"style fidelity: {style_fidelity} can't be smaller than 0.")
        if style_fidelity > 1.0:
            raise ValueError(f"style fidelity: {style_fidelity} can't be larger than 1.0.")

        if reference_guidance_start >= reference_guidance_end:
            raise ValueError(
                f"reference guidance start: {reference_guidance_start} cannot be larger or equal to reference guidance end: {reference_guidance_end}."
            )
        if reference_guidance_start < 0.0:
            raise ValueError(f"reference guidance start: {reference_guidance_start} can't be smaller than 0.")
        if reference_guidance_end > 1.0:
            raise ValueError(f"reference guidance end: {reference_guidance_end} can't be larger than 1.0.")

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        ref_image: Union[torch.Tensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        attention_auto_machine_weight: float = 1.0,
        gn_auto_machine_weight: float = 1.0,
        reference_guidance_start: float = 0.0,
        reference_guidance_end: float = 1.0,
        style_fidelity: float = 0.5,
        reference_attn: bool = True,
        reference_adain: bool = True,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders.
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.Tensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.Tensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be accepted
                as an image. The dimensions of the output image defaults to `image`'s dimensions. If height and/or
                width are passed, `image` is resized accordingly. If multiple ControlNets are specified in `init`,
                images must be passed as a list such that each element of the list can be correctly batched for input
                to a single ControlNet.
            ref_image (`torch.Tensor`, `PIL.Image.Image`):
                The Reference Control input condition. Reference Control uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.Tensor`, it is passed to Reference Control as is. `PIL.Image.Image` can
                also be accepted as an image.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image. Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image. Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. This is sent to `tokenizer_2`
                and `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://huggingface.co/papers/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, pooled text embeddings are generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs (prompt
                weighting). If not provided, pooled `negative_prompt_embeds` are generated from `negative_prompt` input
                argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            attention_auto_machine_weight (`float`):
                Weight of using reference query for self attention's context.
                If attention_auto_machine_weight=1.0, use reference query for all self attention's context.
            gn_auto_machine_weight (`float`):
                Weight of using reference adain. If gn_auto_machine_weight=2.0, use all reference adain plugins.
            reference_guidance_start (`float`, *optional*, defaults to 0.0):
                The percentage of total steps at which the reference ControlNet starts applying.
            reference_guidance_end (`float`, *optional*, defaults to 1.0):
                The percentage of total steps at which the reference ControlNet stops applying.
            style_fidelity (`float`):
                style fidelity of ref_uncond_xt. If style_fidelity=1.0, control more important,
                elif style_fidelity=0.0, prompt more important, else balanced.
            reference_attn (`bool`):
                Whether to use reference query for self attention's context.
            reference_adain (`bool`):
                Whether to use reference adain.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned containing the output images.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            image,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            negative_pooled_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs,
        )

        self.check_ref_inputs(
            ref_image,
            reference_guidance_start,
            reference_guidance_end,
            style_fidelity,
            reference_attn,
            reference_adain,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3.1 Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt,
            prompt_2,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        # 3.2 Encode ip_adapter_image
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []

            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False

        # 5. Preprocess reference image
        ref_image = self.prepare_ref_image(
            image=ref_image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=prompt_embeds.dtype,
        )

        # 6. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        self._num_timesteps = len(timesteps)

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 8. Prepare reference latent variables
        ref_image_latents = self.prepare_ref_latents(
            ref_image,
            batch_size * num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
            self.do_classifier_free_guidance,
        )

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        reference_keeps = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)
            reference_keep = 1.0 - float(
                i / len(timesteps) < reference_guidance_start or (i + 1) / len(timesteps) > reference_guidance_end
            )
            reference_keeps.append(reference_keep)

        # 9.2 Modify self attention and group norm
        MODE = "write"
        uc_mask = (
            torch.Tensor([1] * batch_size * num_images_per_prompt + [0] * batch_size * num_images_per_prompt)
            .type_as(ref_image_latents)
            .bool()
        )

        do_classifier_free_guidance = self.do_classifier_free_guidance

        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
        ):
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if MODE == "write":
                    self.bank.append(norm_hidden_states.detach().clone())
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                if MODE == "read":
                    if attention_auto_machine_weight > self.attn_weight:
                        attn_output_uc = self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=torch.cat([norm_hidden_states] + self.bank, dim=1),
                            # attention_mask=attention_mask,
                            **cross_attention_kwargs,
                        )
                        attn_output_c = attn_output_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:
                            attn_output_c[uc_mask] = self.attn1(
                                norm_hidden_states[uc_mask],
                                encoder_hidden_states=norm_hidden_states[uc_mask],
                                **cross_attention_kwargs,
                            )
                        attn_output = style_fidelity * attn_output_c + (1.0 - style_fidelity) * attn_output_uc
                        self.bank.clear()
                    else:
                        attn_output = self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                            attention_mask=attention_mask,
                            **cross_attention_kwargs,
                        )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

        def hacked_mid_forward(self, *args, **kwargs):
            eps = 1e-6
            x = self.original_forward(*args, **kwargs)
            if MODE == "write":
                if gn_auto_machine_weight >= self.gn_weight:
                    var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                    self.mean_bank.append(mean)
                    self.var_bank.append(var)
            if MODE == "read":
                if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                    var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                    std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                    mean_acc = sum(self.mean_bank) / float(len(self.mean_bank))
                    var_acc = sum(self.var_bank) / float(len(self.var_bank))
                    std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                    x_uc = (((x - mean) / std) * std_acc) + mean_acc
                    x_c = x_uc.clone()
                    if do_classifier_free_guidance and style_fidelity > 0:
                        x_c[uc_mask] = x[uc_mask]
                    x = style_fidelity * x_c + (1.0 - style_fidelity) * x_uc
                self.mean_bank = []
                self.var_bank = []
            return x

        def hack_CrossAttnDownBlock2D_forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
        ):
            eps = 1e-6

            # TODO(Patrick, William) - attention mask is not used
            output_states = ()

            for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                if MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        self.mean_bank.append([mean])
                        self.var_bank.append([var])
                if MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:
                            hidden_states_c[uc_mask] = hidden_states[uc_mask]
                        hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

                output_states = output_states + (hidden_states,)

            if MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)

                output_states = output_states + (hidden_states,)

            return hidden_states, output_states

        def hacked_DownBlock2D_forward(self, hidden_states, temb=None, *args, **kwargs):
            eps = 1e-6

            output_states = ()

            for i, resnet in enumerate(self.resnets):
                hidden_states = resnet(hidden_states, temb)

                if MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        self.mean_bank.append([mean])
                        self.var_bank.append([var])
                if MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:
                            hidden_states_c[uc_mask] = hidden_states[uc_mask]
                        hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

                output_states = output_states + (hidden_states,)

            if MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)

                output_states = output_states + (hidden_states,)

            return hidden_states, output_states

        def hacked_CrossAttnUpBlock2D_forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
        ):
            eps = 1e-6
            # TODO(Patrick, William) - attention mask is not used
            for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

                if MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        self.mean_bank.append([mean])
                        self.var_bank.append([var])
                if MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:
                            hidden_states_c[uc_mask] = hidden_states[uc_mask]
                        hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

            if MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        def hacked_UpBlock2D_forward(
            self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, *args, **kwargs
        ):
            eps = 1e-6
            for i, resnet in enumerate(self.resnets):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)

                if MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        self.mean_bank.append([mean])
                        self.var_bank.append([var])
                if MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:
                            hidden_states_c[uc_mask] = hidden_states[uc_mask]
                        hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

            if MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        if reference_attn:
            attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock)]
            attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))

        if reference_adain:
            gn_modules = [self.unet.mid_block]
            self.unet.mid_block.gn_weight = 0

            down_blocks = self.unet.down_blocks
            for w, module in enumerate(down_blocks):
                module.gn_weight = 1.0 - float(w) / float(len(down_blocks))
                gn_modules.append(module)

            up_blocks = self.unet.up_blocks
            for w, module in enumerate(up_blocks):
                module.gn_weight = float(w) / float(len(up_blocks))
                gn_modules.append(module)

            for i, module in enumerate(gn_modules):
                if getattr(module, "original_forward", None) is None:
                    module.original_forward = module.forward
                if i == 0:
                    # mid_block
                    module.forward = hacked_mid_forward.__get__(module, torch.nn.Module)
                elif isinstance(module, CrossAttnDownBlock2D):
                    module.forward = hack_CrossAttnDownBlock2D_forward.__get__(module, CrossAttnDownBlock2D)
                elif isinstance(module, DownBlock2D):
                    module.forward = hacked_DownBlock2D_forward.__get__(module, DownBlock2D)
                elif isinstance(module, CrossAttnUpBlock2D):
                    module.forward = hacked_CrossAttnUpBlock2D_forward.__get__(module, CrossAttnUpBlock2D)
                elif isinstance(module, UpBlock2D):
                    module.forward = hacked_UpBlock2D_forward.__get__(module, UpBlock2D)
                module.mean_bank = []
                module.var_bank = []
                module.gn_weight *= 2

        # 9.2 Prepare added time ids & embeddings
        if isinstance(image, list):
            original_size = original_size or image[0].shape[-2:]
        else:
            original_size = original_size or image.shape[-2:]
        target_size = target_size or (height, width)

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # 10.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                # controlnet(s) inference
                if guess_mode and self.do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                    controlnet_added_cond_kwargs = {
                        "text_embeds": add_text_embeds.chunk(2)[1],
                        "time_ids": add_time_ids.chunk(2)[1],
                    }
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds
                    controlnet_added_cond_kwargs = added_cond_kwargs

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    added_cond_kwargs=controlnet_added_cond_kwargs,
                    return_dict=False,
                )

                if guess_mode and self.do_classifier_free_guidance:
                    # Inferred ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds

                # ref only part
                if reference_keeps[i] > 0:
                    noise = randn_tensor(
                        ref_image_latents.shape, generator=generator, device=device, dtype=ref_image_latents.dtype
                    )
                    ref_xt = self.scheduler.add_noise(
                        ref_image_latents,
                        noise,
                        t.reshape(
                            1,
                        ),
                    )
                    ref_xt = self.scheduler.scale_model_input(ref_xt, t)

                    MODE = "write"
                    self.unet(
                        ref_xt,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )

                # predict the noise residual
                MODE = "read"
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
