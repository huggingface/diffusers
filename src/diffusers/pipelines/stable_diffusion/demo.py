#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import click
import torch

from aitemplate.testing.benchmark_pt import benchmark_torch_function
from pipeline_stable_diffusion_ait import StableDiffusionAITPipeline


@click.command()
@click.option("--token", default="", help="access token")
@click.option("--width", default=512, help="Width of generated image")
@click.option("--height", default=512, help="Height of generated image")
@click.option("--prompt", default="A vision of paradise, Unreal Engine", help="prompt")
@click.option(
    "--benchmark", type=bool, default=False, help="run stable diffusion e2e benchmark"
)
def run(token, width, height, prompt, benchmark):
    pipe = StableDiffusionAITPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=token,
    ).to("cuda")

    with torch.autocast("cuda"):
        image = pipe(prompt, height, width).images[0]
        if benchmark:
            t = benchmark_torch_function(10, pipe, prompt)
            print(f"sd e2e: {t} ms")

    image.save("example_ait.png")


if __name__ == "__main__":
    run()
