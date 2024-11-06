import torch

from app.sana_pipeline import SanaPipeline

from diffusers import SanaTransformer2DModel

import torch
from app.sana_pipeline import (
    SanaPipeline, classify_height_width_bin, prepare_prompt_ar, DPMS, resize_and_crop_tensor, 
    vae_decode, guidance_type_select
)
from torchvision.utils import save_image


class SanaPipelineDiffuser(SanaPipeline):
    def __init__(self, config: str | None = "configs/sana_config/1024ms/Sana_1600M_img1024.yaml"):
        super().__init__(config)
        self.model = SanaTransformer2DModel.from_pretrained(
            "output/Sana_1600M_1024px_diffusers",
            subfolder="transformer",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(device)

    @torch.inference_mode()
    def forward(
            self,
            prompt=None,
            height=1024,
            width=1024,
            negative_prompt="",
            num_inference_steps=20,
            guidance_scale=5,
            pag_guidance_scale=1.,
            num_images_per_prompt=1,
            generator=torch.Generator().manual_seed(42),
            latent=None,
    ):
        self.ori_height, self.ori_width = height, width
        self.height, self.width = classify_height_width_bin(height, width, ratios=self.base_ratios)
        self.latent_size_h, self.latent_size_w = (
            self.height // self.config.vae.vae_downsample_rate,
            self.width // self.config.vae.vae_downsample_rate,
        )
        self.guidance_type = guidance_type_select(self.guidance_type, pag_guidance_scale, self.config.model.attn_type)

        # 1. pre-compute negative embedding
        if negative_prompt != "":
            null_caption_token = self.tokenizer(
                negative_prompt,
                max_length=self.max_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            self.null_caption_embs = self.text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask)[
                0
            ]

        if prompt is None:
            prompt = [""]
        prompts = prompt if isinstance(prompt, list) else [prompt]
        samples = []

        for prompt in prompts:
            # data prepare
            prompts, hw, ar = (
                [],
                torch.tensor([[self.image_size, self.image_size]], dtype=torch.float, device=self.device).repeat(
                    num_images_per_prompt, 1
                ),
                torch.tensor([[1.0]], device=self.device).repeat(num_images_per_prompt, 1),
            )
            for _ in range(num_images_per_prompt):
                prompts.append(prepare_prompt_ar(prompt, self.base_ratios, device=self.device, show=False)[0].strip())

            # prepare text feature
            if not self.config.text_encoder.chi_prompt:
                max_length_all = self.config.text_encoder.model_max_length
                prompts_all = prompts
            else:
                chi_prompt = "\n".join(self.config.text_encoder.chi_prompt)
                prompts_all = [chi_prompt + prompt for prompt in prompts]
                num_chi_prompt_tokens = len(self.tokenizer.encode(chi_prompt))
                max_length_all = (
                        num_chi_prompt_tokens + self.config.text_encoder.model_max_length - 2
                )  # magic number 2: [bos], [_]

            caption_token = self.tokenizer(
                prompts_all, max_length=max_length_all, padding="max_length", truncation=True, return_tensors="pt"
            ).to(self.device)
            select_index = [0] + list(range(-self.config.text_encoder.model_max_length + 1, 0))
            caption_embs = self.text_encoder(caption_token.input_ids, caption_token.attention_mask)[0][:, None][
                           :, :, select_index
                           ].to(dtype)
            print(1111111111, select_index)
            emb_masks = caption_token.attention_mask[:, select_index]
            null_y = self.null_caption_embs.repeat(len(prompts), 1, 1)[:, None].to(dtype)

            # start sampling
            with torch.no_grad():
                n = len(prompts)
                if latent is None:
                    z = torch.randn(
                        n,
                        self.config.vae.vae_latent_dim,
                        self.latent_size_h,
                        self.latent_size_w,
                        generator=generator,
                        device=self.device,
                        dtype=self.weight_dtype,
                    )
                else:
                    z = latent.to(self.weight_dtype).to(device)
                print(z.mean(), z.std())
                model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks)
                scheduler = DPMS(
                    self.model.forward,
                    condition=caption_embs,
                    uncondition=null_y,
                    guidance_type=self.guidance_type,
                    cfg_scale=guidance_scale,
                    pag_scale=pag_guidance_scale,
                    pag_applied_layers=self.config.model.pag_applied_layers,
                    model_type="flow",
                    model_kwargs=model_kwargs,
                    schedule="FLOW",
                )
                scheduler.register_progress_bar(self.progress_fn)
                sample = scheduler.sample(
                    z,
                    steps=num_inference_steps,
                    order=2,
                    skip_type="time_uniform_flow",
                    method="multistep",
                    flow_shift=self.flow_shift,
                )

            sample = sample.to(self.weight_dtype)
            sample = vae_decode(self.config.vae.vae_type, self.vae, sample)
            sample = resize_and_crop_tensor(sample, self.ori_width, self.ori_height)
            samples.append(sample)

            return sample

        return samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--selfmodel", action="store_true", help="save all the pipelien elemets in one.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    prompt = 'a cyberpunk cat with a neon sign that says "Sana"'
    generator = torch.Generator(device=device).manual_seed(42)

    latent = torch.randn(
        1,
        32,
        32,
        32,
        generator=generator,
        device=device,
    )
    # diffusers Sana Model
    pipe_diffuser = SanaPipelineDiffuser()

    image = pipe_diffuser(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=5.0,
        pag_guidance_scale=1.0,
        num_inference_steps=18,
        generator=generator,
        latent=latent,
    )
    save_image(image, 'sana_diffusers.png', nrow=1, normalize=True, value_range=(-1, 1))

    # self implementation
    # sana = SanaPipeline("configs/sana_config/1024ms/Sana_1600M_img1024.yaml")
    # sana.from_pretrained("hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth")

    # image = sana(
    #     prompt=prompt,
    #     height=1024,
    #     width=1024,
    #     guidance_scale=5.0,
    #     pag_guidance_scale=1.0,
    #     num_inference_steps=18,
    #     generator=generator,
    #     latent=latent,
    # )

    # save_image(image, 'sana_self.png', nrow=1, normalize=True, value_range=(-1, 1))
