from typing import List, Optional, Union

import torch

from transformers import T5Model

from ...models import Notes2AudioModel, UNet2DConditionModel
from ...pipeline_utils import DiffusionPipeline


class Notes2AudioPipeline(DiffusionPipeline):
    r"""
    Pipeline for notes(midi)-to-audio generation using music-spectrogram diffusion introduced by magenta in
    notes2audio.

        Args:
        decoder ([` `]):
            Decoder model used to convert the hidden states to a mel spectrogram. Should take as an input the encoder
            hidden states as well a the diffusion noise. Should be the soundstreal MELGan style decoder
        context_encoder ([` `]):
            Encoder used to create the context to smooth the transitions between adjacent audio frames.
        note_encoder (` `):
            model used to encode ??
       scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `decoder` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(self, spectrogram_decoder, context_encoder, note_encoder, vocoder, scheduler):
        self.spectrogram_decoder = spectrogram_decoder
        self.context_encoder = context_encoder
        self.note_encoder = note_encoder
        self.vocoder = vocoder
        self.scheduler = scheduler
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            spectrogram_decoder=spectrogram_decoder,
            context_encoder=context_encoder,
            note_encoder=note_encoder,
            scheduler=scheduler,
            vocoder=vocoder,
        )

    @torch.no_grad()
    def __call__(
        self,
        midi: Union[str, List[str]],
        audio_length: int,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ):
        return
