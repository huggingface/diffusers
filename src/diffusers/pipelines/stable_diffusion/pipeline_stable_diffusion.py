import inspect

import torch

from tqdm.auto import tqdm

from ...pipeline_utils import DiffusionPipeline


class StableDiffusionPipeline(DiffusionPipeline):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        generator=None,
        torch_device=None,
        eta=0.0,
        guidance_scale=1.0,
        num_inference_steps=50,
        output_type="pil",
    ):
        # eta corresponds to η in paper and should be between [0, 1]

        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = len(prompt)

        self.unet.to(torch_device)
        self.vae.to(torch_device)
        self.text_encoder.to(torch_device)

        # get unconditional embeddings for classifier free guidance
        if guidance_scale != 1.0:
            uncond_input = self.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(torch_device))[0]

        # get prompt text embeddings
        text_input = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids.to(torch_device))[0]

        latents = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
            generator=generator,
        )
        latents = latents.to(torch_device)

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_kwrags = {}
        if not accepts_eta:
            extra_kwrags["eta"] = eta

        for t in tqdm(self.scheduler.timesteps):
            if guidance_scale == 1.0:
                # guidance_scale of 1 means no guidance
                latents_input = latents
                context = text_embeddings
            else:
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                latents_input = torch.cat([latents] * 2)
                context = torch.cat([uncond_embeddings, text_embeddings])

            # predict the noise residual
            noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
            # perform guidance
            if guidance_scale != 1.0:
                noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_kwrags)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image}
