import argparse
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

from diffusers import FluxPipeline
from diffusers.utils import make_image_grid


class CLIPScore(nn.Module):
    def __init__(self, model_id: str = "openai/clip-vit-large-patch14", device: str = None) -> None:
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_id)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_id)
        if device is not None:
            self.model = self.model.to(device)
        self.model = self.model.to(dtype=torch.float32)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.eval()

    def forward(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        pixel_values = self._prepare_images(images).to(device=device, dtype=torch.float32)
        text_inputs = self.tokenizer(list(prompts), padding=True, truncation=True, return_tensors="pt").to(device)

        image_embeds = self.model.get_image_features(pixel_values=pixel_values)
        text_embeds = self.model.get_text_features(**text_inputs)

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        return (image_embeds * text_embeds).sum(dim=-1)

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = (images + 1) / 2
        pixel_values = torch.clamp(pixel_values, 0, 1)
        crop_size = self.image_processor.crop_size
        if isinstance(crop_size, dict):
            target_height = crop_size["height"]
            target_width = crop_size["width"]
        else:
            target_height = target_width = crop_size
        pixel_values = F.interpolate(
            pixel_values, size=(target_height, target_width), mode="bilinear", align_corners=False
        )
        mean = torch.tensor(
            self.image_processor.image_mean, device=pixel_values.device, dtype=pixel_values.dtype
        ).view(1, -1, 1, 1)
        std = torch.tensor(self.image_processor.image_std, device=pixel_values.device, dtype=pixel_values.dtype).view(
            1, -1, 1, 1
        )
        pixel_values = (pixel_values - mean) / std
        return pixel_values.to(dtype=torch.float32)


class AdversarialFluxPipeline(FluxPipeline):
    def adversarial_refinement(
        self,
        prompt: Union[str, list[str]],
        reward_prompt: Optional[Union[str, list[str]]] = None,
        num_rounds: int = 1,
        step_size: float = 0.1,
        clip_model_id: str = "openai/clip-vit-large-patch14",
        record_intermediate: bool = False,
        **generate_kwargs,
    ):
        if num_rounds < 0:
            raise ValueError("`num_rounds` must be non-negative")

        generate_kwargs = dict(generate_kwargs)
        height, width = self._resolve_height_width(generate_kwargs.get("height"), generate_kwargs.get("width"))
        generate_kwargs["height"] = height
        generate_kwargs["width"] = width
        generate_kwargs["output_type"] = "latent"
        generate_kwargs.setdefault("return_dict", True)

        flux_output = super().__call__(prompt=prompt, **generate_kwargs)
        latents = flux_output.images
        device = latents.device

        if self.reward_model is None:
            self.reward_model = CLIPScore(model_id=clip_model_id, device=device)

        reward_prompts = self._expand_prompts(reward_prompt if reward_prompt is not None else prompt, latents.shape[0])

        with torch.no_grad():
            current_images = self._decode_packed_latents(latents, height, width).to(dtype=torch.float32)
            current_scores = self.reward_model(current_images, reward_prompts)

        intermediate_images: list[Image.Image] = []
        if record_intermediate:
            intermediate_images.append(self.image_processor.postprocess(current_images, output_type="pil")[0])

        score_trace: list[float] = [current_scores.mean().item()]

        for _ in range(num_rounds):
            current_images.requires_grad_(True)
            scores = self.reward_model(current_images, reward_prompts)
            total_score = scores.mean()

            grad = torch.autograd.grad(total_score, current_images)[0]

            with torch.no_grad():
                current_images = current_images + step_size * grad
                current_images = current_images.clamp_(-1.0, 1.0)

            current_images = current_images.detach()

            with torch.no_grad():
                current_scores = self.reward_model(current_images, reward_prompts)

            score_trace.append(current_scores.mean().item())
            if record_intermediate:
                intermediate_images.append(self.image_processor.postprocess(current_images, output_type="pil")[0])

        final_image = self.image_processor.postprocess(current_images, output_type="pil")[0]

        return {
            "final_image": final_image,
            "latents": latents.detach(),
            "score_trace": score_trace,
            "final_scores": current_scores.detach().cpu().tolist(),
            "intermediate_images": intermediate_images,
        }

    def _decode_packed_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        unpacked = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        unpacked = (unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        decoded = self.vae.decode(unpacked, return_dict=False)[0]
        return decoded.to(dtype=torch.float32)

    def _resolve_height_width(self, height: int, width: int) -> tuple[int, int]:
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        return height, width

    @staticmethod
    def _expand_prompts(prompts: Union[str, list[str]], batch_size: int) -> list[str]:
        if isinstance(prompts, str):
            return [prompts] * batch_size
        if len(prompts) != batch_size:
            raise ValueError(f"Expected {batch_size} reward prompts, got {len(prompts)}")
        return prompts


def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial refinement of Flux latents with CLIP score guidance.")
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--prompt", default="Photo of a dog sitting near a sea waiting for its companion to come.")
    parser.add_argument("--reward-prompt", default=None, type=str)
    parser.add_argument("--output", default="flux_adversarial.png")
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip-model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--record-intermediates", action="store_true")
    parser.add_argument("--intermediate-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device)
        generator.manual_seed(args.seed)

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    pipe = AdversarialFluxPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe.to(args.device)

    pipe.reward_model = CLIPScore(model_id=args.clip_model_id, device=args.device)

    record_intermediate = args.record_intermediates or args.intermediate_dir is not None

    result = pipe.adversarial_refinement(
        prompt=args.prompt,
        reward_prompt=args.reward_prompt,
        num_rounds=args.num_rounds,
        step_size=args.step_size,
        clip_model_id=args.clip_model_id,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=generator,
        record_intermediate=record_intermediate,
    )

    result["final_image"].save(args.output)

    if args.intermediate_dir and result["intermediate_images"]:
        output_dir = Path(args.intermediate_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        images = result["intermediate_images"]
        image_grid = make_image_grid(images, cols=len(images), rows=1)
        filename = output_dir / "image_grid.png"
        image_grid.save(filename)

    print("Average CLIP score trace:", result["score_trace"])
    print("Final per-sample CLIP scores:", result["final_scores"])


if __name__ == "__main__":
    main()
