from __future__ import annotations

import torch

from ...image_processor import VaeImageProcessor
from ...models import AutoencoderRAE
from ...models.transformers.transformer_rae_dit import RAEDiT2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import RAEDiTPipelineOutput


class RAEDiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for class-conditioned image generation in RAE latent space.

    Parameters:
        transformer ([`RAEDiT2DModel`]):
            Class-conditioned latent transformer used for Stage-2 denoising in RAE latent space.
        vae ([`AutoencoderRAE`]):
            Representation autoencoder used to decode latent samples back to RGB images.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Flow-matching scheduler used to integrate the latent denoising trajectory.
    """

    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        transformer: RAEDiT2DModel,
        vae: AutoencoderRAE,
        scheduler: FlowMatchEulerDiscreteScheduler,
        id2label: dict[int, str] | None = None,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )
        serialized_id2label = None
        if id2label is not None:
            serialized_id2label = {str(key): value for key, value in id2label.items()}
        self.register_to_config(id2label=serialized_id2label)

        self.labels = {}
        if self.config.id2label is not None:
            for key, value in self.config.id2label.items():
                for label in value.split(","):
                    self.labels[label.strip()] = int(key)
            self.labels = dict(sorted(self.labels.items()))

        self.image_processor = VaeImageProcessor(vae_scale_factor=1, do_resize=False, do_normalize=False)
        self._guidance_scale = 1.0

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1.0

    def get_label_ids(self, label: str | list[str]) -> list[int]:
        r"""
        Map ImageNet-style label strings to class ids.
        """

        if not isinstance(label, list):
            label = [label]

        for label_name in label:
            if label_name not in self.labels:
                raise ValueError(
                    f"{label_name} does not exist. Please make sure to select one of the following labels: \n {self.labels}."
                )

        return [self.labels[label_name] for label_name in label]

    def _prepare_class_labels(
        self,
        class_labels: int | list[int] | torch.Tensor,
        num_images_per_prompt: int,
        device: torch.device,
    ) -> torch.LongTensor:
        class_labels = torch.as_tensor(class_labels, device=device, dtype=torch.long).reshape(-1)

        if num_images_per_prompt > 1:
            class_labels = class_labels.repeat_interleave(num_images_per_prompt)

        return class_labels

    def prepare_latents(
        self,
        batch_size: int,
        latent_channels: int,
        latent_size: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None,
    ) -> torch.Tensor:
        shape = (batch_size, latent_channels, latent_size, latent_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested a batch size of "
                f"{batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        latents = latents.to(device=device, dtype=dtype)
        if latents.shape != shape:
            raise ValueError(f"Expected `latents` to have shape {shape}, but got {tuple(latents.shape)}.")

        return latents

    def _prepare_timesteps(
        self, timestep: torch.Tensor | float, batch_size: int, sample: torch.Tensor
    ) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timestep = torch.tensor([timestep], dtype=dtype, device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        else:
            timestep = timestep.to(sample.device)

        return timestep.expand(batch_size)

    @torch.no_grad()
    def __call__(
        self,
        class_labels: int | list[int] | torch.Tensor,
        guidance_scale: float = 1.0,
        guidance_start: float = 0.0,
        guidance_end: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        num_inference_steps: int = 50,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> RAEDiTPipelineOutput | tuple:
        r"""
        The call function to the pipeline for generation.

        Args:
            class_labels (`int`, `list[int]`, or `torch.Tensor`):
                The class ids for the images to generate.
            guidance_scale (`float`, *optional*, defaults to `1.0`):
                Classifier-free guidance scale. Guidance is enabled when `guidance_scale > 1`.
            guidance_start (`float`, *optional*, defaults to `0.0`):
                Lower bound of the normalized timestep interval in which classifier-free guidance is active.
            guidance_end (`float`, *optional*, defaults to `1.0`):
                Upper bound of the normalized timestep interval in which classifier-free guidance is active.
            num_images_per_prompt (`int`, *optional*, defaults to `1`):
                Number of images to generate per class label.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Random generator used for latent sampling.
            latents (`torch.Tensor`, *optional*):
                Pre-generated latent noise tensor of shape `(batch, channels, height, width)`.
            num_inference_steps (`int`, *optional*, defaults to `50`):
                Number of denoising steps.
            output_type (`str`, *optional*, defaults to `"pil"`):
                Output format. Choose from `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return an [`RAEDiTPipelineOutput`] instead of a tuple.
        """

        if num_images_per_prompt < 1:
            raise ValueError(f"`num_images_per_prompt` must be >= 1, but got {num_images_per_prompt}.")
        if guidance_scale < 1.0:
            raise ValueError(f"`guidance_scale` must be >= 1.0, but got {guidance_scale}.")
        if not 0.0 <= guidance_start <= guidance_end <= 1.0:
            raise ValueError(
                f"`guidance_start` and `guidance_end` must satisfy 0 <= guidance_start <= guidance_end <= 1, but got "
                f"{guidance_start} and {guidance_end}."
            )
        if output_type not in {"latent", "np", "pil", "pt"}:
            raise ValueError(f"Unsupported `output_type`: {output_type}.")
        if guidance_scale > 1.0 and self.transformer.config.class_dropout_prob <= 0:
            raise ValueError(
                "Classifier-free guidance requires `transformer.config.class_dropout_prob > 0` so a null class token exists."
            )

        self._guidance_scale = guidance_scale

        device = self._execution_device
        dtype = self.transformer.dtype

        class_labels = self._prepare_class_labels(
            class_labels, num_images_per_prompt=num_images_per_prompt, device=device
        )
        batch_size = class_labels.shape[0]

        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size,
            latent_channels=latent_channels,
            latent_size=latent_size,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self._num_timesteps = len(self.scheduler.timesteps)

        for timestep in self.progress_bar(self.scheduler.timesteps):
            if self.do_classifier_free_guidance:
                latent_model_input = torch.cat([latents, latents], dim=0)
                null_class_labels = torch.full(
                    (batch_size,),
                    self.transformer.config.num_classes,
                    device=device,
                    dtype=class_labels.dtype,
                )
                class_labels_input = torch.cat([class_labels, null_class_labels], dim=0)
            else:
                latent_model_input = latents
                class_labels_input = class_labels

            timestep_input = self._prepare_timesteps(timestep, latent_model_input.shape[0], latent_model_input)
            timestep_input = timestep_input / self.scheduler.config.num_train_timesteps
            model_output = self.transformer(
                latent_model_input,
                timestep=timestep_input,
                class_labels=class_labels_input,
            ).sample

            if self.do_classifier_free_guidance:
                cond_model_output, uncond_model_output = model_output.chunk(2, dim=0)
                guided_model_output = uncond_model_output + guidance_scale * (cond_model_output - uncond_model_output)
                guidance_mask = (timestep_input[:batch_size] >= guidance_start) & (
                    timestep_input[:batch_size] <= guidance_end
                )
                guidance_mask = guidance_mask.view(-1, *([1] * (cond_model_output.ndim - 1)))
                model_output = torch.where(guidance_mask, guided_model_output, cond_model_output)

            latents = self.scheduler.step(model_output, timestep, latents).prev_sample

        if output_type == "latent":
            output = latents
        else:
            images = self.vae.decode(latents.to(dtype=self.vae.dtype)).sample.clamp(0, 1)
            output = self.image_processor.postprocess(images, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (output,)

        return RAEDiTPipelineOutput(images=output)
