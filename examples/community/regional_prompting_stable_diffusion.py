import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torchvision.transforms.functional as FF
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.loaders.ip_adapter import IPAdapterMixin
from diffusers.loaders.lora_pipeline import LoraLoaderMixin
from diffusers.loaders.single_file import FromSingleFileMixin
from diffusers.loaders.textual_inversion import TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


try:
    from compel import Compel
except ImportError:
    Compel = None

KBASE = "ADDBASE"
KCOMM = "ADDCOMM"
KBRK = "BREAK"


class RegionalPromptingStableDiffusionPipeline(
    DiffusionPipeline,
    TextualInversionLoaderMixin,
    LoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
    StableDiffusionLoraLoaderMixin,
):
    r"""
    Args for Regional Prompting Pipeline:
        rp_args:dict
        Required
            rp_args["mode"]: cols, rows, prompt, prompt-ex
        for cols, rows mode
            rp_args["div"]: ex) 1;1;1(Divide into 3 regions)
        for prompt, prompt-ex mode
            rp_args["th"]: ex) 0.5,0.5,0.6 (threshold for prompt mode)

        Optional
            rp_args["save_mask"]: True/False (save masks in prompt mode)
            rp_args["power"]: int (power for attention maps in prompt mode)
            rp_args["base_ratio"]:
                float (Sets the ratio of the base prompt)
                ex) 0.2 (20%*BASE_PROMPT + 80%*REGION_PROMPT)
                [Use base prompt](https://github.com/hako-mikan/sd-webui-regional-prompter?tab=readme-ov-file#use-base-prompt)

    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        # Initialize additional properties needed for DiffusionPipeline
        self._num_timesteps = None
        self._interrupt = False
        self._guidance_scale = 7.5
        self._guidance_rescale = 0.0
        self._clip_skip = None
        self._cross_attention_kwargs = None

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: str = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        rp_args: Dict[str, str] = None,
    ):
        active = KBRK in prompt[0] if isinstance(prompt, list) else KBRK in prompt
        use_base = KBASE in prompt[0] if isinstance(prompt, list) else KBASE in prompt
        if negative_prompt is None:
            negative_prompt = "" if isinstance(prompt, str) else [""] * len(prompt)

        device = self._execution_device
        regions = 0

        self.base_ratio = float(rp_args["base_ratio"]) if "base_ratio" in rp_args else 0.0
        self.power = int(rp_args["power"]) if "power" in rp_args else 1

        prompts = prompt if isinstance(prompt, list) else [prompt]
        n_prompts = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt]
        self.batch = batch = num_images_per_prompt * len(prompts)

        if use_base:
            bases = prompts.copy()
            n_bases = n_prompts.copy()

            for i, prompt in enumerate(prompts):
                parts = prompt.split(KBASE)
                if len(parts) == 2:
                    bases[i], prompts[i] = parts
                elif len(parts) > 2:
                    raise ValueError(f"Multiple instances of {KBASE} found in prompt: {prompt}")
            for i, prompt in enumerate(n_prompts):
                n_parts = prompt.split(KBASE)
                if len(n_parts) == 2:
                    n_bases[i], n_prompts[i] = n_parts
                elif len(n_parts) > 2:
                    raise ValueError(f"Multiple instances of {KBASE} found in negative prompt: {prompt}")

            all_bases_cn, _ = promptsmaker(bases, num_images_per_prompt)
            all_n_bases_cn, _ = promptsmaker(n_bases, num_images_per_prompt)

        all_prompts_cn, all_prompts_p = promptsmaker(prompts, num_images_per_prompt)
        all_n_prompts_cn, _ = promptsmaker(n_prompts, num_images_per_prompt)

        equal = len(all_prompts_cn) == len(all_n_prompts_cn)

        if Compel:
            compel = Compel(tokenizer=self.tokenizer, text_encoder=self.text_encoder)

            def getcompelembs(prps):
                embl = []
                for prp in prps:
                    embl.append(compel.build_conditioning_tensor(prp))
                return torch.cat(embl)

            conds = getcompelembs(all_prompts_cn)
            unconds = getcompelembs(all_n_prompts_cn)
            base_embs = getcompelembs(all_bases_cn) if use_base else None
            base_n_embs = getcompelembs(all_n_bases_cn) if use_base else None
            # When using base, it seems more reasonable to use base prompts as prompt_embeddings rather than regional prompts
            embs = getcompelembs(prompts) if not use_base else base_embs
            n_embs = getcompelembs(n_prompts) if not use_base else base_n_embs

            if use_base and self.base_ratio > 0:
                conds = self.base_ratio * base_embs + (1 - self.base_ratio) * conds
                unconds = self.base_ratio * base_n_embs + (1 - self.base_ratio) * unconds

            prompt = negative_prompt = None
        else:
            conds = self.encode_prompt(prompts, device, 1, True)[0]
            unconds = (
                self.encode_prompt(n_prompts, device, 1, True)[0]
                if equal
                else self.encode_prompt(all_n_prompts_cn, device, 1, True)[0]
            )

            if use_base and self.base_ratio > 0:
                base_embs = self.encode_prompt(bases, device, 1, True)[0]
                base_n_embs = (
                    self.encode_prompt(n_bases, device, 1, True)[0]
                    if equal
                    else self.encode_prompt(all_n_bases_cn, device, 1, True)[0]
                )

                conds = self.base_ratio * base_embs + (1 - self.base_ratio) * conds
                unconds = self.base_ratio * base_n_embs + (1 - self.base_ratio) * unconds

            embs = n_embs = None

        if not active:
            pcallback = None
            mode = None
        else:
            if any(x in rp_args["mode"].upper() for x in ["COL", "ROW"]):
                mode = "COL" if "COL" in rp_args["mode"].upper() else "ROW"
                ocells, icells, regions = make_cells(rp_args["div"])

            elif "PRO" in rp_args["mode"].upper():
                regions = len(all_prompts_p[0])
                mode = "PROMPT"
                reset_attnmaps(self)
                self.ex = "EX" in rp_args["mode"].upper()
                self.target_tokens = target_tokens = tokendealer(self, all_prompts_p)
                thresholds = [float(x) for x in rp_args["th"].split(",")]

            orig_hw = (height, width)
            revers = True

            def pcallback(s_self, step: int, timestep: int, latents: torch.Tensor, selfs=None):
                if "PRO" in mode:  # in Prompt mode, make masks from sum of attention maps
                    self.step = step

                    if len(self.attnmaps_sizes) > 3:
                        self.history[step] = self.attnmaps.copy()
                        for hw in self.attnmaps_sizes:
                            allmasks = []
                            basemasks = [None] * batch
                            for tt, th in zip(target_tokens, thresholds):
                                for b in range(batch):
                                    key = f"{tt}-{b}"
                                    _, mask, _ = makepmask(self, self.attnmaps[key], hw[0], hw[1], th, step)
                                    mask = mask.unsqueeze(0).unsqueeze(-1)
                                    if self.ex:
                                        allmasks[b::batch] = [x - mask for x in allmasks[b::batch]]
                                        allmasks[b::batch] = [torch.where(x > 0, 1, 0) for x in allmasks[b::batch]]
                                    allmasks.append(mask)
                                    basemasks[b] = mask if basemasks[b] is None else basemasks[b] + mask
                            basemasks = [1 - mask for mask in basemasks]
                            basemasks = [torch.where(x > 0, 1, 0) for x in basemasks]
                            allmasks = basemasks + allmasks

                            self.attnmasks[hw] = torch.cat(allmasks)
                        self.maskready = True
                return latents

            def hook_forward(module):
                # diffusers==0.23.2
                def forward(
                    hidden_states: torch.Tensor,
                    encoder_hidden_states: Optional[torch.Tensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    temb: Optional[torch.Tensor] = None,
                    scale: float = 1.0,
                ) -> torch.Tensor:
                    attn = module
                    xshape = hidden_states.shape
                    self.hw = (h, w) = split_dims(xshape[1], *orig_hw)

                    if revers:
                        nx, px = hidden_states.chunk(2)
                    else:
                        px, nx = hidden_states.chunk(2)

                    if equal:
                        hidden_states = torch.cat(
                            [px for i in range(regions)] + [nx for i in range(regions)],
                            0,
                        )
                        encoder_hidden_states = torch.cat([conds] + [unconds])
                    else:
                        hidden_states = torch.cat([px for i in range(regions)] + [nx], 0)
                        encoder_hidden_states = torch.cat([conds] + [unconds])

                    residual = hidden_states

                    if attn.spatial_norm is not None:
                        hidden_states = attn.spatial_norm(hidden_states, temb)

                    input_ndim = hidden_states.ndim

                    if input_ndim == 4:
                        batch_size, channel, height, width = hidden_states.shape
                        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                    batch_size, sequence_length, _ = (
                        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                    )

                    if attention_mask is not None:
                        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

                    if attn.group_norm is not None:
                        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                    query = attn.to_q(hidden_states)

                    if encoder_hidden_states is None:
                        encoder_hidden_states = hidden_states
                    elif attn.norm_cross:
                        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                    key = attn.to_k(encoder_hidden_states)
                    value = attn.to_v(encoder_hidden_states)

                    inner_dim = key.shape[-1]
                    head_dim = inner_dim // attn.heads

                    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                    # the output of sdp = (batch, num_heads, seq_len, head_dim)
                    # TODO: add support for attn.scale when we move to Torch 2.1
                    hidden_states = scaled_dot_product_attention(
                        self,
                        query,
                        key,
                        value,
                        attn_mask=attention_mask,
                        dropout_p=0.0,
                        is_causal=False,
                        getattn="PRO" in mode,
                    )

                    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                    hidden_states = hidden_states.to(query.dtype)

                    # linear proj
                    hidden_states = attn.to_out[0](hidden_states)
                    # dropout
                    hidden_states = attn.to_out[1](hidden_states)

                    if input_ndim == 4:
                        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                    if attn.residual_connection:
                        hidden_states = hidden_states + residual

                    hidden_states = hidden_states / attn.rescale_output_factor

                    #### Regional Prompting Col/Row mode
                    if any(x in mode for x in ["COL", "ROW"]):
                        reshaped = hidden_states.reshape(hidden_states.size()[0], h, w, hidden_states.size()[2])
                        center = reshaped.shape[0] // 2
                        px = reshaped[0:center] if equal else reshaped[0:-batch]
                        nx = reshaped[center:] if equal else reshaped[-batch:]
                        outs = [px, nx] if equal else [px]
                        for out in outs:
                            c = 0
                            for i, ocell in enumerate(ocells):
                                for icell in icells[i]:
                                    if "ROW" in mode:
                                        out[
                                            0:batch,
                                            int(h * ocell[0]) : int(h * ocell[1]),
                                            int(w * icell[0]) : int(w * icell[1]),
                                            :,
                                        ] = out[
                                            c * batch : (c + 1) * batch,
                                            int(h * ocell[0]) : int(h * ocell[1]),
                                            int(w * icell[0]) : int(w * icell[1]),
                                            :,
                                        ]
                                    else:
                                        out[
                                            0:batch,
                                            int(h * icell[0]) : int(h * icell[1]),
                                            int(w * ocell[0]) : int(w * ocell[1]),
                                            :,
                                        ] = out[
                                            c * batch : (c + 1) * batch,
                                            int(h * icell[0]) : int(h * icell[1]),
                                            int(w * ocell[0]) : int(w * ocell[1]),
                                            :,
                                        ]
                                    c += 1
                        px, nx = (px[0:batch], nx[0:batch]) if equal else (px[0:batch], nx)
                        hidden_states = torch.cat([nx, px], 0) if revers else torch.cat([px, nx], 0)
                        hidden_states = hidden_states.reshape(xshape)

                    #### Regional Prompting Prompt mode
                    elif "PRO" in mode:
                        px, nx = (
                            torch.chunk(hidden_states) if equal else hidden_states[0:-batch],
                            hidden_states[-batch:],
                        )

                        if (h, w) in self.attnmasks and self.maskready:

                            def mask(input):
                                out = torch.multiply(input, self.attnmasks[(h, w)])
                                for b in range(batch):
                                    for r in range(1, regions):
                                        out[b] = out[b] + out[r * batch + b]
                                return out

                            px, nx = (mask(px), mask(nx)) if equal else (mask(px), nx)
                        px, nx = (px[0:batch], nx[0:batch]) if equal else (px[0:batch], nx)
                        hidden_states = torch.cat([nx, px], 0) if revers else torch.cat([px, nx], 0)
                    return hidden_states

                return forward

            def hook_forwards(root_module: torch.nn.Module):
                for name, module in root_module.named_modules():
                    if "attn2" in name and module.__class__.__name__ == "Attention":
                        module.forward = hook_forward(module)

            hook_forwards(self.unet)

        output = self.stable_diffusion_call(
            prompt=prompt,
            prompt_embeds=embs,
            negative_prompt=negative_prompt,
            negative_prompt_embeds=n_embs,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            output_type=output_type,
            return_dict=return_dict,
            callback_on_step_end=pcallback,
        )

        if "save_mask" in rp_args:
            save_mask = rp_args["save_mask"]
        else:
            save_mask = False

        if mode == "PROMPT" and save_mask:
            saveattnmaps(
                self,
                output,
                height,
                width,
                thresholds,
                num_inference_steps // 2,
                regions,
            )

        return output

    # copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    # copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion
    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    # copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion
    @torch.no_grad()
    def stable_diffusion_call(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
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
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
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
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
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

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        self.model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
        self._optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
        self._exclude_from_cpu_offload = ["safety_checker"]
        self._callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

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

        # 0. Default height and width to unet
        if not height or not width:
            height = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[0]
            )
            width = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[1]
            )
            height, width = height * self.vae_scale_factor, width * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
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

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://huggingface.co/papers/2305.08891
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

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

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    # copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        r"""Encodes the prompt into text encoder hidden states."""
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
            # cast text_encoder.dtype to prevent overflow when using bf16
            text_input_ids = text_input_ids.to(device=device, dtype=self.text_encoder.dtype)
            prompt_embeds = self.text_encoder(
                text_input_ids,
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]
        else:
            text_encoder_lora_scale = None
            if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
                text_encoder_lora_scale = lora_scale
            if text_encoder_lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
                # dynamically adjust the LoRA scale
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        # duplicate text embeddings for each generation per prompt
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""]
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Unscale LoRA weights to avoid overfitting. This is a hack
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        """Encodes the image into image encoder hidden states."""
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        """Prepares and processes IP-Adapter image embeddings."""
        image_embeds = []
        if do_classifier_free_guidance:
            negative_image_embeds = []
        if ip_adapter_image_embeds is None:
            for image in ip_adapter_image:
                if not isinstance(image, torch.Tensor):
                    image = self.image_processor.preprocess(image)
                    image = image.to(device=device)
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                image_emb, neg_image_emb = self.encode_image(image, device, num_images_per_prompt, True)
                image_embeds.append(image_emb)
                if do_classifier_free_guidance:
                    negative_image_embeds.append(neg_image_emb)

            if len(image_embeds) == 1:
                image_embeds = image_embeds[0]
                if do_classifier_free_guidance:
                    negative_image_embeds = negative_image_embeds[0]
            else:
                image_embeds = torch.cat(image_embeds, dim=0)
                if do_classifier_free_guidance:
                    negative_image_embeds = torch.cat(negative_image_embeds, dim=0)
        else:
            repeat_dim = 2 if do_classifier_free_guidance else 1
            image_embeds = ip_adapter_image_embeds.repeat_interleave(repeat_dim, dim=0)
            if do_classifier_free_guidance:
                negative_image_embeds = torch.zeros_like(image_embeds)

        if do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds])

        return image_embeds

    def run_safety_checker(self, image, device, dtype):
        """Runs the safety checker on the generated image."""
        if self.safety_checker is None:
            has_nsfw_concept = None
            return image, has_nsfw_concept

        if isinstance(self.safety_checker, StableDiffusionSafetyChecker):
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.to(dtype),
            )
        else:
            images_np = self.numpy_to_pil(image)
            safety_checker_input = self.safety_checker.feature_extractor(images_np, return_tensors="pt").to(device)
            has_nsfw_concept = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.to(dtype),
            )[1]

        return image, has_nsfw_concept

    # copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion
    def decode_latents(self, latents):
        """Decodes the latents to images."""
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    # copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ):
        """Gets the guidance scale embedding for classifier free guidance conditioning.
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                The guidance scale tensor used for classifier free guidance conditioning.
            embedding_dim (`int`, defaults to 512):
                The dimensionality of the guidance scale embedding.
            dtype (`torch.dtype`, defaults to torch.float32):
                The dtype of the embedding.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None


### Make prompt list for each regions
def promptsmaker(prompts, batch):
    out_p = []
    plen = len(prompts)
    for prompt in prompts:
        add = ""
        if KCOMM in prompt:
            add, prompt = prompt.split(KCOMM)
            add = add.strip() + " "
        prompts = [p.strip() for p in prompt.split(KBRK)]
        out_p.append([add + p for i, p in enumerate(prompts)])
    out = [None] * batch * len(out_p[0]) * len(out_p)
    for p, prs in enumerate(out_p):  # inputs prompts
        for r, pr in enumerate(prs):  # prompts for regions
            start = (p + r * plen) * batch
            out[start : start + batch] = [pr] * batch  # P1R1B1,P1R1B2...,P1R2B1,P1R2B2...,P2R1B1...
    return out, out_p


### make regions from ratios
### ";" makes outercells, "," makes inner cells
def make_cells(ratios):
    if ";" not in ratios and "," in ratios:
        ratios = ratios.replace(",", ";")
    ratios = ratios.split(";")
    ratios = [inratios.split(",") for inratios in ratios]

    icells = []
    ocells = []

    def startend(cells, array):
        current_start = 0
        array = [float(x) for x in array]
        for value in array:
            end = current_start + (value / sum(array))
            cells.append([current_start, end])
            current_start = end

    startend(ocells, [r[0] for r in ratios])

    for inratios in ratios:
        if 2 > len(inratios):
            icells.append([[0, 1]])
        else:
            add = []
            startend(add, inratios[1:])
            icells.append(add)
    return ocells, icells, sum(len(cell) for cell in icells)


def make_emblist(self, prompts):
    with torch.no_grad():
        tokens = self.tokenizer(
            prompts,
            max_length=self.tokenizer.model_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        embs = self.text_encoder(tokens, output_hidden_states=True).last_hidden_state.to(self.device, dtype=self.dtype)
    return embs


def split_dims(xs, height, width):
    def repeat_div(x, y):
        while y > 0:
            x = math.ceil(x / 2)
            y = y - 1
        return x

    scale = math.ceil(math.log2(math.sqrt(height * width / xs)))
    dsh = repeat_div(height, scale)
    dsw = repeat_div(width, scale)
    return dsh, dsw


##### for prompt mode
def get_attn_maps(self, attn):
    height, width = self.hw
    target_tokens = self.target_tokens
    if (height, width) not in self.attnmaps_sizes:
        self.attnmaps_sizes.append((height, width))

    for b in range(self.batch):
        for t in target_tokens:
            power = self.power
            add = attn[b, :, :, t[0] : t[0] + len(t)] ** (power) * (self.attnmaps_sizes.index((height, width)) + 1)
            add = torch.sum(add, dim=2)
            key = f"{t}-{b}"
            if key not in self.attnmaps:
                self.attnmaps[key] = add
            else:
                if self.attnmaps[key].shape[1] != add.shape[1]:
                    add = add.view(8, height, width)
                    add = FF.resize(add, self.attnmaps_sizes[0], antialias=None)
                    add = add.reshape_as(self.attnmaps[key])

                self.attnmaps[key] = self.attnmaps[key] + add


def reset_attnmaps(self):  # init parameters in every batch
    self.step = 0
    self.attnmaps = {}  # made from attention maps
    self.attnmaps_sizes = []  # height,width set of u-net blocks
    self.attnmasks = {}  # made from attnmaps for regions
    self.maskready = False
    self.history = {}


def saveattnmaps(self, output, h, w, th, step, regions):
    masks = []
    for i, mask in enumerate(self.history[step].values()):
        img, _, mask = makepmask(self, mask, h, w, th[i % len(th)], step)
        if self.ex:
            masks = [x - mask for x in masks]
            masks.append(mask)
            if len(masks) == regions - 1:
                output.images.extend([FF.to_pil_image(mask) for mask in masks])
                masks = []
        else:
            output.images.append(img)


def makepmask(
    self, mask, h, w, th, step
):  # make masks from attention cache return [for preview, for attention, for Latent]
    th = th - step * 0.005
    if 0.05 >= th:
        th = 0.05
    mask = torch.mean(mask, dim=0)
    mask = mask / mask.max().item()
    mask = torch.where(mask > th, 1, 0)
    mask = mask.float()
    mask = mask.view(1, *self.attnmaps_sizes[0])
    img = FF.to_pil_image(mask)
    img = img.resize((w, h))
    mask = FF.resize(mask, (h, w), interpolation=FF.InterpolationMode.NEAREST, antialias=None)
    lmask = mask
    mask = mask.reshape(h * w)
    mask = torch.where(mask > 0.1, 1, 0)
    return img, mask, lmask


def tokendealer(self, all_prompts):
    for prompts in all_prompts:
        targets = [p.split(",")[-1] for p in prompts[1:]]
        tt = []

        for target in targets:
            ptokens = (
                self.tokenizer(
                    prompts,
                    max_length=self.tokenizer.model_max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
            )[0]
            ttokens = (
                self.tokenizer(
                    target,
                    max_length=self.tokenizer.model_max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
            )[0]

            tlist = []

            for t in range(ttokens.shape[0] - 2):
                for p in range(ptokens.shape[0]):
                    if ttokens[t + 1] == ptokens[p]:
                        tlist.append(p)
            if tlist != []:
                tt.append(tlist)

    return tt


def scaled_dot_product_attention(
    self,
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    getattn=False,
) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=self.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if getattn:
        get_attn_maps(self, attn_weight)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


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


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://huggingface.co/papers/2305.08891).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg
