from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ...models import DreamTransformer1DModel
from ...schedulers import DreamMaskedDiffusionScheduler
from ...utils import is_torch_xla_available, logging
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import DreamTextPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DreamTextPipeline(DiffusionPipeline):
    r"""
    Dream 7B diffusion based LLM.

    Introduced in https://hkunlp.github.io/blog/2025/dream/.
    """

    model_cpu_offload_seq = "transformer"
    _optional_components = []  # TODO: list any optional components here
    _callback_tensor_inputs = []  # TODO: what needs to be here?

    def __init__(
        self,
        tokenizer,
        transformer: DreamTransformer1DModel,
        scheduler: DreamMaskedDiffusionScheduler,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        # 131072 in original code
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 512
        )

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def tokenize_prompt(
        self,
        prompt: Union[str, List[str]],
        num_texts_per_prompt: int = 1,
        max_sequence_length: int = 512,
        apply_chat_template: bool = False,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if apply_chat_template:
            prompt_is_chat_template = isinstance(prompt[0], dict) or (isinstance(prompt[0], list) and isinstance(prompt[0][0], dict))
            if not prompt_is_chat_template:
                # Apply simple chat template for each supplied prompt
                prompt = [{"role": "user", "content": prompt_instance} for prompt_instance in prompt]
            # Call the PreTrainedTokenier's apply_chat_template method for chat generation
            text_inputs = self.tokenizer.apply_chat_template(
                prompt,
                return_tensors="pt",
                return_dict=False,  # List[int] output rather than Dict output
                add_generation_prompt=True,
            )
        else:
            # Call the tokenizer's normal __call__ method
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            )

        text_input_ids = text_inputs.input_ids.to(device=device)
        attention_mask = text_inputs.attention_mask.to(device=device)

        # duplicate text tokens and attention mask for each generation per prompt, using mps friendly method
        # TODO: this follows e.g. the Flux pipeline's encode_prompts, why do we repeat in the sequence length dim
        # rather than the batch length dim...?
        text_input_ids = text_input_ids.repeat(1, num_texts_per_prompt)
        text_input_ids = text_input_ids.view(batch_size * num_texts_per_prompt, -1)

        attention_mask = attention_mask.repeat(1, num_texts_per_prompt)
        attention_mask = attention_mask.view(batch_size * num_texts_per_prompt, -1)

        return text_input_ids, attention_mask

    def prepare_latents(
        self,
        batch_size: int,
        max_sequence_length: int,
        text_ids: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        latents_shape = (batch_size, max_sequence_length)
        if latents is None and text_ids is None:
            # Create all-masks latents of length max_sequence_length
            latents = torch.full(latents_shape, self.scheduler.config.mask_token_id, dtype=torch.long, device=device)
        elif latents is None and text_ids is not None:
            # Pad text_ids to max_sequence_length with mask tokens
            # NOTE: text_ids is assumed to have the correct batch dimension already
            latents = F.pad(
                text_ids, (0, max_sequence_length - text_ids.shape[1]), value=self.scheduler.config.mask_token_id
            )
        else:
            if latents.ndim == 1:
                # Unsqueeze a batch dim
                latents = latents.unsqueeze(0)
            # bring latents to the correct batch size
            current_batch_size = latents.shape[0]
            if batch_size % current_batch_size == 0:
                repeat_factor = batch_size // current_batch_size
                latents = latents.repeat(repeat_factor, 1)
            else:
                raise ValueError(
                    f"The `latents` batch size {current_batch_size} must evenly divide the total batch size"
                    f" {batch_size}."
                )

            # If latents is not max_sequence_length, pad to max_sequence_length with mask tokens
            latents = F.pad(
                latents, (0, max_sequence_length - latents.shape[1]), value=self.scheduler.config.mask_token_id
            )

            latents = latents.to(device)

        return latents

    def check_inputs(self, prompt, prompt_embeds, latents):
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is not None and latents is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `latents`: {latents}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        num_inference_steps: int = 512,
        num_texts_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.IntTensor] = None,  # TODO: does supporting both latents and prompt_embeds make sense?
        prompt_embeds: Optional[torch.Tensor] = None,
        temperature: Union[float, Tuple[float, float], List[float]] = 0.2,
        top_p: Union[float, Tuple[float, float], List[float]] = 0.95,
        max_sequence_length: int = 512,
        apply_chat_template: bool = False,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        output_type: str = "pil",  # TODO: replace with options appropriate for text
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ) -> Union[DreamTextPipelineOutput, Tuple[Any]]:
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide text generation. A chat template for
                `transformers.PreTrainedTokenizer.apply_chat_template` can be used if `apply_chat_template` is set to
                `True`.
            num_inference_steps (`int`, *optional*, defaults to 512):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            num_texts_per_prompt (`int`, *optional*, defaults to 1):
                The number of text outputs to generate per prompt. If neither `prompts` nor `prompt_embeds` is
                supplied, this will be interpreted as the batch size for generation.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.IntTensor`, *optional*):
                Pre-generated text tokens from which to start generation. If supplied, this should include any
                conditioning text tokens (analogous to a tokenized version of `prompt`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument. A single vector from the
                pooled and projected final hidden states.
            temperature (`Union[float, Tuple[float, float], List[float]]`, *optional*, defaults to 0.2):
                Configures the temperature scheduler on `self.scheduler`; see `DreamMaskedDiffusionScheduler#set_timesteps`.
            top_p (`Union[float, Tuple[float, float], List[float]]`, *optional*, defaults to 0.95):
                Configures the top-p probability scheduler on `self.scheduler`; see `DreamMaskedDiffusionScheduler#set_timesteps`.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            
            Returns:
            [`~pipelines.pipeline_utils.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.pipeline_utils.ImagePipelineOutput`] is returned, otherwise a
                `tuple` is returned where the first element is a list with the generated images.
        """
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, prompt_embeds, latents)

        # 2. Define call parameters
        # NOTE: it is possible for both prompt and prompt_embeds to be None (which corresponds to "unconditional" text
        # generation)
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            batch_size = 1

        device = self._execution_device

        self._current_timestep = None
        self._interrupt = False

        # 3. Tokenize input text prompts, if any
        if prompt is None:
            prompt = [prompt] if isinstance(prompt, str) else prompt
            text_ids, attention_mask = self.tokenize_prompt(
                prompt=prompt,
                num_texts_per_prompt=num_texts_per_prompt,
                apply_chat_template=apply_chat_template,
                device=device,
            )
        else:
            text_ids, attention_mask = None

        # 4. Prepare latent variables (e.g. the initial sample) for generation
        total_batch_size = batch_size * num_texts_per_prompt
        latents = self.prepare_latents(
            total_batch_size,
            max_sequence_length,
            text_ids=text_ids,
            latents=latents,
            device=device,
        )

        if prompt_embeds is not None:
            prompt_embeds = self.transformer.embed_tokens(latents)
        else:
            # If prompt_embeds's seq len is not max_sequence_length, concat with embedding of mask tokens for the
            # remaining length
            padding_length = max_sequence_length - prompt_embeds.shape[1]
            if padding_length > 0:
                padding_mask_tokens = torch.full(
                    (total_batch_size, padding_length), self.scheduler.config.mask_token_id, device=device
                )
                padding_mask_embedding = self.transformer.embed_tokens(padding_mask_tokens)
                prompt_embeds = torch.cat([prompt_embeds, padding_mask_embedding], dim=1)
            else:
                # Truncate to max_sequence_length, if necessary
                prompt_embeds = prompt_embeds[:, :max_sequence_length, :]

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, temperature=temperature, top_p=top_p, device=device)
        timesteps = self.scheduler.timesteps
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if i > 0:
                    model_output = self.transformer(
                        text_ids=latents,
                        attention_mask=attention_mask,
                        return_dict=False,
                    )[0]
                else:
                    # Use prompt_embeds only at the first step, to support supplying an initial prompt embedding
                    model_output = self.transformer(
                        hidden_states=prompt_embeds,
                        attention_mask=attention_mask,
                        return_dict=False,
                    )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=latents,
                    generator=generator,
                ).prev_sample

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
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 7. Post-processing and output handling
        self._current_timestep = None

        if output_type == "latent":
            texts = latents
        else:
            # TODO: should there be a text_processor class analogous to e.g. VaeImageProcessor???
            texts = self.tokenizer.batch_decode(latents)
            # TODO: if prompt or other conditioning is supplied, remove prompts from generated texts???

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (texts,)

        return DreamTextPipelineOutput(texts=texts)
