import argparse
import json
from pathlib import Path
from typing import Optional, Union
import torch
from PIL import Image
from reward_scorers import BaseRewardScorer, available_scorers, build_scorer
from diffusers import FluxPipeline
from diffusers.utils import make_image_grid


class AdversarialFluxPipeline(FluxPipeline):
    def adversarial_refinement(
        self,
        prompt: Union[str, list[str]],
        reward_model: BaseRewardScorer,
        reward_prompt: Optional[Union[str, list[str]]] = None,
        num_rounds: int = 1,
        step_size: float = 0.1,
        epsilon: Optional[float] = None,
        attack_type: str = "pgd",
        record_intermediate: bool = False,
        **generate_kwargs,
    ):
        if num_rounds < 0:
            raise ValueError("`num_rounds` must be non-negative.")
        if attack_type not in {"pgd", "fgsm"}:
            raise ValueError("`attack_type` must be either 'pgd' or 'fgsm'.")

        generate_kwargs = dict(generate_kwargs)
        height, width = self._resolve_height_width(generate_kwargs.get("height"), generate_kwargs.get("width"))
        generate_kwargs["height"] = height
        generate_kwargs["width"] = width
        generate_kwargs["output_type"] = "latent"
        generate_kwargs.setdefault("return_dict", True)

        flux_output = super().__call__(prompt=prompt, **generate_kwargs)
        latents = flux_output.images
        device = latents.device

        reward_model = reward_model.to(device)
        reward_model.eval()
        if getattr(reward_model, "supports_gradients", True) is False and (num_rounds != 0 or attack_type == "fgsm"):
            raise ValueError(
                f"Scorer `{reward_model.__class__.__name__}` does not support gradients required for adversarial refinement."
            )

        reward_prompts = self._expand_prompts(reward_prompt if reward_prompt is not None else prompt, latents.shape[0])

        with torch.no_grad():
            current_images = self._decode_packed_latents(latents, height, width).to(dtype=torch.float32)
            current_scores = reward_model(current_images, reward_prompts)

        intermediate_images = []
        if record_intermediate:
            intermediate_images.append(self.image_processor.postprocess(current_images, output_type="pil"))

        score_trace = [current_scores.mean().item()]
        per_sample_scores = [current_scores.detach().cpu().tolist()]

        if num_rounds == 0:
            max_rounds = 0
        else:
            max_rounds = 1 if attack_type == "fgsm" else num_rounds

        for round_index in range(max_rounds):
            current_images.requires_grad_(True)
            scores = reward_model(current_images, reward_prompts)
            total_score = scores.mean()

            grad = torch.autograd.grad(total_score, current_images, retain_graph=False, create_graph=False)[0]

            if attack_type == "fgsm":
                step = epsilon if epsilon is not None else step_size
                update = step * grad.sign()
            else:
                update = step_size * grad

            with torch.no_grad():
                current_images = current_images + update
                current_images = current_images.clamp_(-1.0, 1.0)

            current_images = current_images.detach()

            with torch.no_grad():
                current_scores = reward_model(current_images, reward_prompts)

            score_trace.append(current_scores.mean().item())
            per_sample_scores.append(current_scores.detach().cpu().tolist())

            if record_intermediate:
                intermediate_images.append(self.image_processor.postprocess(current_images, output_type="pil"))

        final_images = self.image_processor.postprocess(current_images, output_type="pil")

        return {
            "images": final_images,
            "latents": latents.detach(),
            "score_trace": score_trace,
            "score_trace_per_sample": per_sample_scores,
            "final_scores": current_scores.detach().cpu().tolist(),
            "intermediate_images": intermediate_images,
        }

    def _decode_packed_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        unpacked = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        unpacked = (unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        decoded = self.vae.decode(unpacked, return_dict=False)[0]
        return decoded

    def _resolve_height_width(self, height: Optional[int], width: Optional[int]) -> tuple[int, int]:
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        return height, width

    @staticmethod
    def _expand_prompts(prompts: Union[str, list[str]], batch_size: int) -> list[str]:
        if isinstance(prompts, str):
            return [prompts] * batch_size
        if len(prompts) != batch_size:
            raise ValueError(f"Expected {batch_size} reward prompts, got {len(prompts)}.")
        return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument(
        "--prompt", type=str, default="Photo of a dog sitting near a sea waiting for its companion to come."
    )
    parser.add_argument("--reward-prompt", type=str, default=None)
    parser.add_argument("--output", default="flux_adversarial.png")
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, help="FGSM epsilon. Falls back to step size when omitted.")
    parser.add_argument("--attack-type", choices=["pgd", "fgsm"], default="pgd")
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scorer", choices=available_scorers(), default="clip")
    parser.add_argument("--scorer-model-id", type=str, default=None)
    parser.add_argument("--record-intermediates", action="store_true")
    parser.add_argument("--intermediate-dir", type=str, default=None)
    parser.add_argument("--metadata-output", type=str, default=None)
    parser.add_argument("--output-root", type=str, required=True)
    return parser.parse_args()


def save_intermediates(intermediate_dir: Path, rounds: list[list[Image.Image]]) -> None:
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    for round_index, images in enumerate(rounds):
        for sample_index, image in enumerate(images):
            filename = intermediate_dir / f"round_{round_index:02d}_sample_{sample_index:02d}.png"
            image.save(filename)

    if rounds and len(rounds[0]) == 1:
        grid = make_image_grid([imgs[0] for imgs in rounds], cols=len(rounds), rows=1)
        grid.save(intermediate_dir / "grid.png")


def dump_metadata(metadata_path: Path, payload: dict[str, object]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file)


def main() -> None:
    args = parse_args()

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device)
        generator.manual_seed(args.seed)

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    pipe = AdversarialFluxPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe.to(args.device)

    reward_model = build_scorer(name=args.scorer, model_id=args.scorer_model_id, device=args.device)

    record_intermediate = args.record_intermediates or args.intermediate_dir is not None

    result = pipe.adversarial_refinement(
        prompt=args.prompt,
        reward_prompt=args.reward_prompt,
        reward_model=reward_model,
        num_rounds=args.num_rounds,
        step_size=args.step_size,
        epsilon=args.epsilon,
        attack_type=args.attack_type,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=generator,
        record_intermediate=record_intermediate,
    )

    images = result["images"]
    output_path = Path(args.output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path)

    if args.intermediate_dir:
        intermediate_dir = output_path / args.intermediate_dir
        if intermediate_dir and result["intermediate_images"]:
            save_intermediates(intermediate_dir, result["intermediate_images"])

    if args.metadata_output:
        metadata_payload = {
            "prompt": args.prompt,
            "reward_prompt": args.reward_prompt or "",
            "scorer": {
                "name": args.scorer,
                "model_id": getattr(reward_model, "model_id", args.scorer_model_id or ""),
            },
            "attack": {
                "type": args.attack_type,
                "num_rounds": args.num_rounds,
                "step_size": args.step_size,
                "epsilon": args.epsilon
                if args.epsilon is not None
                else args.step_size
                if args.attack_type == "fgsm"
                else None,
                "rounds_executed": len(result["score_trace"]) - 1,
            },
            "score_trace": result["score_trace"],
            "score_trace_per_sample": result["score_trace_per_sample"],
            "final_scores": result["final_scores"],
        }
        metadata_path = output_path / args.metadata_output
        if metadata_path:
            dump_metadata(metadata_path, metadata_payload)

    print("Mean score trace:", result["score_trace"])
    print("Final per-sample scores:", result["final_scores"])


if __name__ == "__main__":
    main()
