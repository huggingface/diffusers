import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    T5EncoderModel,
    T5Tokenizer,
)

from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL, HiDreamImageTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from ...utils import is_torch_xla_available, logging
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import HiDreamImagePipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
        >>> from diffusers import UniPCMultistepScheduler, HiDreamImagePipeline, HiDreamImageTransformer2DModel

        >>> scheduler = UniPCMultistepScheduler(
        ...     flow_shift=3.0, prediction_type="flow_prediction", use_flow_sigmas=True
        ... )

        >>> tokenizer_4 = PreTrainedTokenizerFast.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        >>> text_encoder_4 = LlamaForCausalLM.from_pretrained(
        ...     "meta-llama/Meta-Llama-3.1-8B-Instruct",
        ...     output_hidden_states=True,
        ...     output_attentions=True,
        ...     torch_dtype=torch.bfloat16,
        ... )

        >>> transformer = HiDreamImageTransformer2DModel.from_pretrained(
        ...     "HiDream-ai/HiDream-I1-Full", subfolder="transformer", torch_dtype=torch.bfloat16
        ... )

        >>> pipe = HiDreamImagePipeline.from_pretrained(
        ...     "HiDream-ai/HiDream-I1-Full",
        ...     scheduler=scheduler,
        ...     tokenizer_4=tokenizer_4,
        ...     text_encoder_4=text_encoder_4,
        ...     transformer=transformer,
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> pipe.enable_model_cpu_offload()

        >>> image = pipe(
        ...     'A cat holding a sign that says "Hi-Dreams.ai".',
        ...     height=1024,
        ...     width=1024,
        ...     guidance_scale=5.0,
        ...     num_inference_steps=50,
        ...     generator=torch.Generator("cuda").manual_seed(0),
        ... ).images[0]
        >>> image.save("output.png")
        ```
"""


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
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


class HiDreamImagePipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->text_encoder_4->transformer->vae"
    _callback_tensor_inputs = ["latents", "t5_prompt_embeds", "llama3_prompt_embeds", "pooled_prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5Tokenizer,
        text_encoder_4: LlamaForCausalLM,
        tokenizer_4: PreTrainedTokenizerFast,
        transformer: HiDreamImageTransformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            text_encoder_4=text_encoder_4,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            tokenizer_4=tokenizer_4,
            scheduler=scheduler,
            transformer=transformer,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        # HiDreamImage latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.default_sample_size = 128
        if getattr(self, "tokenizer_4", None) is not None:
            self.tokenizer_4.pad_token = self.tokenizer_4.eos_token

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_3.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, self.tokenizer_3.model_max_length),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(
                untruncated_ids[:, min(max_sequence_length, self.tokenizer_3.model_max_length) - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {min(max_sequence_length, self.tokenizer_3.model_max_length)} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device), attention_mask=attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        tokenizer,
        text_encoder,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, 218),
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, 218 - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {218} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds

    def _get_llama3_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_4.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer_4(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, self.tokenizer_4.model_max_length),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer_4(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_4.batch_decode(
                untruncated_ids[:, min(max_sequence_length, self.tokenizer_4.model_max_length) - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {min(max_sequence_length, self.tokenizer_4.model_max_length)} tokens: {removed_text}"
            )

        outputs = self.text_encoder_4(
            text_input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
            output_attentions=True,
        )

        prompt_embeds = outputs.hidden_states[1:]
        prompt_embeds = torch.stack(prompt_embeds, dim=0)
        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_4: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        negative_prompt_4: Optional[Union[str, List[str]]] = None,
        t5_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        llama3_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_t5_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_llama3_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 128,
        lora_scale: Optional[float] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = pooled_prompt_embeds.shape[0]

        device = device or self._execution_device

        if pooled_prompt_embeds is None:
            pooled_prompt_embeds_1 = self._get_clip_prompt_embeds(
                self.tokenizer, self.text_encoder, prompt, max_sequence_length, device, dtype
            )

        if do_classifier_free_guidance and negative_pooled_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if len(negative_prompt) > 1 and len(negative_prompt) != batch_size:
                raise ValueError(f"negative_prompt must be of length 1 or {batch_size}")

            negative_pooled_prompt_embeds_1 = self._get_clip_prompt_embeds(
                self.tokenizer, self.text_encoder, negative_prompt, max_sequence_length, device, dtype
            )

            if negative_pooled_prompt_embeds_1.shape[0] == 1 and batch_size > 1:
                negative_pooled_prompt_embeds_1 = negative_pooled_prompt_embeds_1.repeat(batch_size, 1)

        if pooled_prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            if len(prompt_2) > 1 and len(prompt_2) != batch_size:
                raise ValueError(f"prompt_2 must be of length 1 or {batch_size}")

            pooled_prompt_embeds_2 = self._get_clip_prompt_embeds(
                self.tokenizer_2, self.text_encoder_2, prompt_2, max_sequence_length, device, dtype
            )

            if pooled_prompt_embeds_2.shape[0] == 1 and batch_size > 1:
                pooled_prompt_embeds_2 = pooled_prompt_embeds_2.repeat(batch_size, 1)

        if do_classifier_free_guidance and negative_pooled_prompt_embeds is None:
            negative_prompt_2 = negative_prompt_2 or ""
            negative_prompt_2 = [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2

            if len(negative_prompt_2) > 1 and len(negative_prompt_2) != batch_size:
                raise ValueError(f"negative_prompt_2 must be of length 1 or {batch_size}")

            negative_pooled_prompt_embeds_2 = self._get_clip_prompt_embeds(
                self.tokenizer_2, self.text_encoder_2, negative_prompt_2, max_sequence_length, device, dtype
            )

            if negative_pooled_prompt_embeds_2.shape[0] == 1 and batch_size > 1:
                negative_pooled_prompt_embeds_2 = negative_pooled_prompt_embeds_2.repeat(batch_size, 1)

        if pooled_prompt_embeds is None:
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)

        if do_classifier_free_guidance and negative_pooled_prompt_embeds is None:
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds_1, negative_pooled_prompt_embeds_2], dim=-1
            )

        if t5_prompt_embeds is None:
            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            if len(prompt_3) > 1 and len(prompt_3) != batch_size:
                raise ValueError(f"prompt_3 must be of length 1 or {batch_size}")

            t5_prompt_embeds = self._get_t5_prompt_embeds(prompt_3, max_sequence_length, device, dtype)

            if t5_prompt_embeds.shape[0] == 1 and batch_size > 1:
                t5_prompt_embeds = t5_prompt_embeds.repeat(batch_size, 1, 1)

        if do_classifier_free_guidance and negative_t5_prompt_embeds is None:
            negative_prompt_3 = negative_prompt_3 or ""
            negative_prompt_3 = [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3

            if len(negative_prompt_3) > 1 and len(negative_prompt_3) != batch_size:
                raise ValueError(f"negative_prompt_3 must be of length 1 or {batch_size}")

            negative_t5_prompt_embeds = self._get_t5_prompt_embeds(
                negative_prompt_3, max_sequence_length, device, dtype
            )

            if negative_t5_prompt_embeds.shape[0] == 1 and batch_size > 1:
                negative_t5_prompt_embeds = negative_t5_prompt_embeds.repeat(batch_size, 1, 1)

        if llama3_prompt_embeds is None:
            prompt_4 = prompt_4 or prompt
            prompt_4 = [prompt_4] if isinstance(prompt_4, str) else prompt_4

            if len(prompt_4) > 1 and len(prompt_4) != batch_size:
                raise ValueError(f"prompt_4 must be of length 1 or {batch_size}")

            llama3_prompt_embeds = self._get_llama3_prompt_embeds(prompt_4, max_sequence_length, device, dtype)

            if llama3_prompt_embeds.shape[0] == 1 and batch_size > 1:
                llama3_prompt_embeds = llama3_prompt_embeds.repeat(1, batch_size, 1, 1)

        if do_classifier_free_guidance and negative_llama3_prompt_embeds is None:
            negative_prompt_4 = negative_prompt_4 or ""
            negative_prompt_4 = [negative_prompt_4] if isinstance(negative_prompt_4, str) else negative_prompt_4

            if len(negative_prompt_4) > 1 and len(negative_prompt_4) != batch_size:
                raise ValueError(f"negative_prompt_4 must be of length 1 or {batch_size}")

            negative_llama3_prompt_embeds = self._get_llama3_prompt_embeds(
                negative_prompt_4, max_sequence_length, device, dtype
            )

            if negative_llama3_prompt_embeds.shape[0] == 1 and batch_size > 1:
                negative_llama3_prompt_embeds = negative_llama3_prompt_embeds.repeat(1, batch_size, 1, 1)

        # duplicate pooled_prompt_embeds for each generation per prompt
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        # duplicate t5_prompt_embeds for batch_size and num_images_per_prompt
        bs_embed, seq_len, _ = t5_prompt_embeds.shape
        if bs_embed == 1 and batch_size > 1:
            t5_prompt_embeds = t5_prompt_embeds.repeat(batch_size, 1, 1)
        elif bs_embed > 1 and bs_embed != batch_size:
            raise ValueError(f"cannot duplicate t5_prompt_embeds of batch size {bs_embed}")
        t5_prompt_embeds = t5_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        t5_prompt_embeds = t5_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # duplicate llama3_prompt_embeds for batch_size and num_images_per_prompt
        _, bs_embed, seq_len, dim = llama3_prompt_embeds.shape
        if bs_embed == 1 and batch_size > 1:
            llama3_prompt_embeds = llama3_prompt_embeds.repeat(1, batch_size, 1, 1)
        elif bs_embed > 1 and bs_embed != batch_size:
            raise ValueError(f"cannot duplicate llama3_prompt_embeds of batch size {bs_embed}")
        llama3_prompt_embeds = llama3_prompt_embeds.repeat(1, 1, num_images_per_prompt, 1)
        llama3_prompt_embeds = llama3_prompt_embeds.view(-1, batch_size * num_images_per_prompt, seq_len, dim)

        if do_classifier_free_guidance:
            # duplicate negative_pooled_prompt_embeds for batch_size and num_images_per_prompt
            bs_embed, seq_len = negative_pooled_prompt_embeds.shape
            if bs_embed == 1 and batch_size > 1:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(batch_size, 1)
            elif bs_embed > 1 and bs_embed != batch_size:
                raise ValueError(f"cannot duplicate negative_pooled_prompt_embeds of batch size {bs_embed}")
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

            # duplicate negative_t5_prompt_embeds for batch_size and num_images_per_prompt
            bs_embed, seq_len, _ = negative_t5_prompt_embeds.shape
            if bs_embed == 1 and batch_size > 1:
                negative_t5_prompt_embeds = negative_t5_prompt_embeds.repeat(batch_size, 1, 1)
            elif bs_embed > 1 and bs_embed != batch_size:
                raise ValueError(f"cannot duplicate negative_t5_prompt_embeds of batch size {bs_embed}")
            negative_t5_prompt_embeds = negative_t5_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_t5_prompt_embeds = negative_t5_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # duplicate negative_llama3_prompt_embeds for batch_size and num_images_per_prompt
            _, bs_embed, seq_len, dim = negative_llama3_prompt_embeds.shape
            if bs_embed == 1 and batch_size > 1:
                negative_llama3_prompt_embeds = negative_llama3_prompt_embeds.repeat(1, batch_size, 1, 1)
            elif bs_embed > 1 and bs_embed != batch_size:
                raise ValueError(f"cannot duplicate negative_llama3_prompt_embeds of batch size {bs_embed}")
            negative_llama3_prompt_embeds = negative_llama3_prompt_embeds.repeat(1, 1, num_images_per_prompt, 1)
            negative_llama3_prompt_embeds = negative_llama3_prompt_embeds.view(
                -1, batch_size * num_images_per_prompt, seq_len, dim
            )

        return (
            t5_prompt_embeds,
            llama3_prompt_embeds,
            negative_t5_prompt_embeds,
            negative_llama3_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def check_inputs(
        self,
        prompt,
        prompt_2,
        prompt_3,
        prompt_4,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        negative_prompt_4=None,
        t5_prompt_embeds=None,
        llama3_prompt_embeds=None,
        negative_t5_prompt_embeds=None,
        negative_llama3_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and pooled_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `pooled_prompt_embeds`: {pooled_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and pooled_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `pooled_prompt_embeds`: {pooled_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_3 is not None and t5_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_3`: {prompt_3} and `t5_prompt_embeds`: {t5_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_4 is not None and llama3_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_4`: {prompt_4} and `llama3_prompt_embeds`: {llama3_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and pooled_prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `pooled_prompt_embeds`. Cannot leave both `prompt` and `pooled_prompt_embeds` undefined."
            )
        elif prompt is None and t5_prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `t5_prompt_embeds`. Cannot leave both `prompt` and `t5_prompt_embeds` undefined."
            )
        elif prompt is None and llama3_prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `llama3_prompt_embeds`. Cannot leave both `prompt` and `llama3_prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")
        elif prompt_3 is not None and (not isinstance(prompt_3, str) and not isinstance(prompt_3, list)):
            raise ValueError(f"`prompt_3` has to be of type `str` or `list` but is {type(prompt_3)}")
        elif prompt_4 is not None and (not isinstance(prompt_4, str) and not isinstance(prompt_4, list)):
            raise ValueError(f"`prompt_4` has to be of type `str` or `list` but is {type(prompt_4)}")

        if negative_prompt is not None and negative_pooled_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_pooled_prompt_embeds`:"
                f" {negative_pooled_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_pooled_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_pooled_prompt_embeds`:"
                f" {negative_pooled_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_3 is not None and negative_t5_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_3`: {negative_prompt_3} and `negative_t5_prompt_embeds`:"
                f" {negative_t5_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_4 is not None and negative_llama3_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_4`: {negative_prompt_4} and `negative_llama3_prompt_embeds`:"
                f" {negative_llama3_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if pooled_prompt_embeds is not None and negative_pooled_prompt_embeds is not None:
            if pooled_prompt_embeds.shape != negative_pooled_prompt_embeds.shape:
                raise ValueError(
                    "`pooled_prompt_embeds` and `negative_pooled_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `pooled_prompt_embeds` {pooled_prompt_embeds.shape} != `negative_pooled_prompt_embeds`"
                    f" {negative_pooled_prompt_embeds.shape}."
                )
        if t5_prompt_embeds is not None and negative_t5_prompt_embeds is not None:
            if t5_prompt_embeds.shape != negative_t5_prompt_embeds.shape:
                raise ValueError(
                    "`t5_prompt_embeds` and `negative_t5_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `t5_prompt_embeds` {t5_prompt_embeds.shape} != `negative_t5_prompt_embeds`"
                    f" {negative_t5_prompt_embeds.shape}."
                )
        if llama3_prompt_embeds is not None and negative_llama3_prompt_embeds is not None:
            if llama3_prompt_embeds.shape != negative_llama3_prompt_embeds.shape:
                raise ValueError(
                    "`llama3_prompt_embeds` and `negative_llama3_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `llama3_prompt_embeds` {llama3_prompt_embeds.shape} != `negative_llama3_prompt_embeds`"
                    f" {negative_llama3_prompt_embeds.shape}."
                )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_4: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        negative_prompt_4: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        t5_prompt_embeds: Optional[torch.FloatTensor] = None,
        llama3_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_t5_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_llama3_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 128,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        division = self.vae_scale_factor * 2
        S_max = (self.default_sample_size * self.vae_scale_factor) ** 2
        scale = S_max / (width * height)
        scale = math.sqrt(scale)
        width, height = int(width * scale // division * division), int(height * scale // division * division)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            prompt_4,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            negative_prompt_4=negative_prompt_4,
            t5_prompt_embeds=t5_prompt_embeds,
            llama3_prompt_embeds=llama3_prompt_embeds,
            negative_t5_prompt_embeds=negative_t5_prompt_embeds,
            negative_llama3_prompt_embeds=negative_llama3_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif pooled_prompt_embeds is not None:
            batch_size = pooled_prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode prompt
        lora_scale = self.attention_kwargs.get("scale", None) if self.attention_kwargs is not None else None
        (
            t5_prompt_embeds,
            llama3_prompt_embeds,
            negative_t5_prompt_embeds,
            negative_llama3_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            prompt_4=prompt_4,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            negative_prompt_4=negative_prompt_4,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            t5_prompt_embeds=t5_prompt_embeds,
            llama3_prompt_embeds=llama3_prompt_embeds,
            negative_t5_prompt_embeds=negative_t5_prompt_embeds,
            negative_llama3_prompt_embeds=negative_llama3_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            t5_prompt_embeds = torch.cat([negative_t5_prompt_embeds, t5_prompt_embeds], dim=0)
            llama3_prompt_embeds = torch.cat([negative_llama3_prompt_embeds, llama3_prompt_embeds], dim=1)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            pooled_prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        if latents.shape[-2] != latents.shape[-1]:
            B, C, H, W = latents.shape
            pH, pW = H // self.transformer.config.patch_size, W // self.transformer.config.patch_size

            img_sizes = torch.tensor([pH, pW], dtype=torch.int64).reshape(-1)
            img_ids = torch.zeros(pH, pW, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW)[None, :]
            img_ids = img_ids.reshape(pH * pW, -1)
            img_ids_pad = torch.zeros(self.transformer.max_seq, 3)
            img_ids_pad[: pH * pW, :] = img_ids

            img_sizes = img_sizes.unsqueeze(0).to(latents.device)
            img_ids = img_ids_pad.unsqueeze(0).to(latents.device)
            if self.do_classifier_free_guidance:
                img_sizes = img_sizes.repeat(2 * B, 1)
                img_ids = img_ids.repeat(2 * B, 1, 1)
        else:
            img_sizes = img_ids = None

        # 5. Prepare timesteps
        mu = calculate_shift(self.transformer.max_seq)
        scheduler_kwargs = {"mu": mu}
        if isinstance(self.scheduler, UniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device)  # , shift=math.exp(mu))
            timesteps = self.scheduler.timesteps
        else:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                **scheduler_kwargs,
            )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timesteps=timestep,
                    t5_encoder_hidden_states=t5_prompt_embeds,
                    llama3_encoder_hidden_states=llama3_prompt_embeds,
                    pooled_embeds=pooled_prompt_embeds,
                    img_sizes=img_sizes,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]
                noise_pred = -noise_pred

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    t5_prompt_embeds = callback_outputs.pop("t5_prompt_embeds", t5_prompt_embeds)
                    llama3_prompt_embeds = callback_outputs.pop("llama3_prompt_embeds", llama3_prompt_embeds)
                    pooled_prompt_embeds = callback_outputs.pop("pooled_prompt_embeds", pooled_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return HiDreamImagePipelineOutput(images=image)
