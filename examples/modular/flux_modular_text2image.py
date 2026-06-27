"""Modular Flux text-to-image inference, based on the Modular Diffusers quickstart.

Runs Flux through the modular pipeline interface. The Flux transformer + text encoders
in bf16 do not all fit on a 32GB GPU at once, so we use `ComponentsManager` auto CPU
offload to keep only the model currently in use on the GPU.

Defaults to the ungated FLUX.1-schnell so it runs without an HF token. Pass --repo
black-forest-labs/FLUX.1-dev (gated) for higher quality; see the argument help below.

Usage:
    python examples/modular/flux_modular_text2image.py \
        --prompt "a cat wizard with a red hat, fantasy, detailed" \
        --output flux_modular.png
"""

import argparse

import torch

from diffusers import ComponentsManager, ModularPipeline


def main():
    parser = argparse.ArgumentParser(description="Modular Flux text-to-image inference.")
    # FLUX.1-schnell is Apache-2.0 / ungated so it runs without an HF token. For
    # FLUX.1-dev (gated, higher quality) pass --repo black-forest-labs/FLUX.1-dev
    # --steps 28 --guidance-scale 3.5 and set HF_TOKEN after accepting its license.
    parser.add_argument("--repo", default="black-forest-labs/FLUX.1-schnell", help="Model repo id.")
    parser.add_argument(
        "--prompt",
        default="cat wizard with red hat, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="flux_modular.png")
    args = parser.parse_args()

    # Auto CPU offload moves each component to the GPU only while it runs, then back to
    # CPU. This keeps peak VRAM around the size of the largest single model (the ~24GB
    # transformer) instead of the sum of all components, so FLUX.1-dev fits in 32GB.
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda:0")

    # Lazy loading: from_pretrained reads the config; weights load on load_components.
    pipe = ModularPipeline.from_pretrained(args.repo, components_manager=manager)
    pipe.load_components(torch_dtype=torch.bfloat16)

    print(pipe.blocks)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]

    image.save(args.output)
    print(f"Saved image to {args.output}")


if __name__ == "__main__":
    main()
