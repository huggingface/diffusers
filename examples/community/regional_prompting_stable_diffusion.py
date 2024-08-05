# tested for diffusers >=0.28.0
import math
from typing import Dict, List, Optional, Union

import torch
import torchvision.transforms.functional as FF
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.configuration_utils import FrozenDict, deprecate
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

try:
    from compel import Compel
except ImportError:
    Compel = None

KCOMM = "ADDCOMM"
KBRK = "BREAK"


class RegionalPromptingStableDiffusionPipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, IPAdapterMixin, FromSingleFileMixin
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
        requires_safety_checker: bool = True,
    ):
        # super().__init__(
        #     vae,
        #     text_encoder,
        #     tokenizer,
        #     unet,
        #     scheduler,
        #     safety_checker,
        #     feature_extractor,
        #     requires_safety_checker,
        # )

        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "skip_prk_steps") and scheduler.config.skip_prk_steps is False:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration"
                " `skip_prk_steps`. `skip_prk_steps` should be set to True in the configuration file. Please make"
                " sure to update the config accordingly as not setting `skip_prk_steps` in the config might lead to"
                " incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face"
                " Hub, it would be very nice if you could open a Pull request for the"
                " `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "skip_prk_steps not set",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            new_config = dict(scheduler.config)
            new_config["skip_prk_steps"] = True
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)
        # Check shapes, assume num_channels_latents == 4, num_channels_mask == 1, num_channels_masked == 4
        if unet.config.in_channels != 4:
            logger.warning(
                f"You have loaded a UNet with {unet.config.in_channels} input channels, whereas by default,"
                f" {self.__class__} assumes that `pipeline.unet` has 4 input channels: 4 for `num_channels_latents`,"
                ". If you did not intend to modify"
                " this behavior, please check whether you have loaded the right checkpoint."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Optional[str],
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ) -> torch.FloatTensor:
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
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
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
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
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

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

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
        if negative_prompt is None:
            negative_prompt = "" if isinstance(prompt, str) else [""] * len(prompt)

        device = self._execution_device
        regions = 0

        self.power = int(rp_args["power"]) if "power" in rp_args else 1

        prompts = prompt if isinstance(prompt, list) else [prompt]
        n_prompts = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt]
        self.batch = batch = num_images_per_prompt * len(prompts)
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
            embs = getcompelembs(prompts)
            n_embs = getcompelembs(n_prompts)
            prompt = negative_prompt = None
        else:
            conds = self.encode_prompt(prompts, device, 1, True)[0]
            unconds = (
                self.encode_prompt(n_prompts, device, 1, True)[0]
                if equal
                else self.encode_prompt(all_n_prompts_cn, device, 1, True)[0]
            )
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
                if "PRO" in mode:  # in Prompt mode, make masks from sum of attension maps
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

                    args = () if USE_PEFT_BACKEND else (scale,)

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

                    args = () if USE_PEFT_BACKEND else (scale,)
                    query = attn.to_q(hidden_states, *args)

                    if encoder_hidden_states is None:
                        encoder_hidden_states = hidden_states
                    elif attn.norm_cross:
                        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                    key = attn.to_k(encoder_hidden_states, *args)
                    value = attn.to_v(encoder_hidden_states, *args)

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
                    hidden_states = attn.to_out[0](hidden_states, *args)
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

        output = StableDiffusionPipeline(**self.components)(
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


### Make prompt list for each regions
def promptsmaker(prompts, batch):
    out_p = []
    plen = len(prompts)
    for prompt in prompts:
        add = ""
        if KCOMM in prompt:
            add, prompt = prompt.split(KCOMM)
            add = add + " "
        prompts = prompt.split(KBRK)
        out_p.append([add + p for p in prompts])
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
    self.attnmaps = {}  # maked from attention maps
    self.attnmaps_sizes = []  # height,width set of u-net blocks
    self.attnmasks = {}  # maked from attnmaps for regions
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
