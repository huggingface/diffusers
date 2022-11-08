import math
from typing import Optional

import torch

from ...models.t5_attention import ContinuousContextTransformer
from ...pipeline_utils import DiffusionPipeline, MelPipelineOutput
from ...schedulers import DDPMScheduler


class SpectrogramDiffusionPipeline(DiffusionPipeline):
    def __init__(self, cont_context_trans: ContinuousContextTransformer, scheduler: DDPMScheduler) -> None:
        super().__init__()

        # From MELGAN
        self.min_value = math.log(1e-5)  # Matches MelGAN training.
        self.max_value = 4.0  # Largest value for most examples

        self.register_modules(cont_context_trans=cont_context_trans, scheduler=scheduler)

    def scale_features(self, features, output_range=(-1.0, 1.0), clip=False):
        """Linearly scale features to network outputs range."""
        min_out, max_out = output_range
        if clip:
            features = torch.clip(features, self.min_value, self.max_value)
        # Scale to [0, 1].
        zero_one = (features - self.min_value) / (self.max_value - self.min_value)
        # Scale to [min_out, max_out].
        return zero_one * (max_out - min_out) + min_out

    @torch.no_grad()
    def __call__(
        self,
        encoder_input_tokens,
        encoder_continuous_inputs,
        encoder_continuous_mask,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 1000,
        return_dict: bool = True,
        predict_epsilon: bool = True,
        **kwargs,
    ):
        target_shape = encoder_continuous_inputs.shape
        encoder_continuous_inputs = self.scale_features(encoder_continuous_inputs, output_range=[-1.0, 1.0], clip=True)

        encodings_and_masks = self.cont_context_trans.encode(
            encoder_input_tokens=encoder_input_tokens,
            continuous_inputs=encoder_continuous_inputs,
            continuous_mask=encoder_continuous_mask,
        )

        # Sample gaussian noise to begin loop
        x = torch.randn(target_shape, generator=generator)
        x = x.to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            output = self.cont_context_trans.decode(
                encodings_and_masks=encodings_and_masks,
                input_tokens=x,
                noise_time=t,
            )

            # 2. compute previous output: x_t -> x_t-1
            x = self.scheduler.step(output, t, x, generator=generator, predict_epsilon=predict_epsilon).prev_sample

        mel = self.scale_to_features(x, input_range=[-1.0, 1.0])
        mel = mel.cpu().numpy()

        if not return_dict:
            return (mel,)

        return MelPipelineOutput(mels=mel)
