# Copyright 2022 The Music Spectrogram Diffusion Authors.
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional, Tuple, Union, Any, Callable

import numpy as np
import torch

from ...models import T5FilmDecoder
from ...schedulers import DDPMScheduler
from ...utils import is_note_seq_available, randn_tensor, is_onnx_available

if is_onnx_available():
    from ..onnx_utils import OnnxRuntimeModel
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from .continous_encoder import SpectrogramContEncoder
from .midi_utils import (
    DEFAULT_MAX_SHIFT_SECONDS,
    DEFAULT_NUM_VELOCITY_BINS,
    DEFAULT_STEPS_PER_SECOND,
    FRAME_RATE,
    HOP_SIZE,
    SAMPLE_RATE,
    TARGET_FEATURE_LENGTH,
    Codec,
    EventRange,
    NoteEncodingState,
    NoteRepresentationConfig,
    Tokenizer,
    audio_to_frames,
    encode_and_index_events,
    note_encoding_state_to_events,
    note_event_data_to_events,
    note_representation_processor_chain,
    note_sequence_to_onsets_and_offsets_and_programs,
    program_to_slakh_program,
)
from .notes_encoder import SpectrogramNotesEncoder


if is_note_seq_available():
    import note_seq
else:
    raise ImportError("Please install note-seq via `pip install note-seq`")


class SpectrogramDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        notes_encoder: SpectrogramNotesEncoder,
        continuous_encoder: SpectrogramContEncoder,
        decoder: T5FilmDecoder,
        scheduler: DDPMScheduler,
        melgan: OnnxRuntimeModel if is_onnx_available() else Any,
    ) -> None:
        super().__init__()

        # From MELGAN
        self.min_value = math.log(1e-5)  # Matches MelGAN training.
        self.max_value = 4.0  # Largest value for most examples
        self.n_dims = 128

        self.register_modules(
            notes_encoder=notes_encoder,
            continuous_encoder=continuous_encoder,
            decoder=decoder,
            scheduler=scheduler,
            melgan=melgan,
        )

    def scale_features(self, features, output_range=(-1.0, 1.0), clip=False):
        """Linearly scale features to network outputs range."""
        min_out, max_out = output_range
        if clip:
            features = torch.clip(features, self.min_value, self.max_value)
        # Scale to [0, 1].
        zero_one = (features - self.min_value) / (self.max_value - self.min_value)
        # Scale to [min_out, max_out].
        return zero_one * (max_out - min_out) + min_out

    def scale_to_features(self, outputs, input_range=(-1.0, 1.0), clip=False):
        """Invert by linearly scaling network outputs to features range."""
        min_out, max_out = input_range
        outputs = torch.clip(outputs, min_out, max_out) if clip else outputs
        # Scale to [0, 1].
        zero_one = (outputs - min_out) / (max_out - min_out)
        # Scale to [self.min_value, self.max_value].
        return zero_one * (self.max_value - self.min_value) + self.min_value

    def encode(self, input_tokens, continuous_inputs, continuous_mask):
        tokens_mask = input_tokens > 0
        tokens_encoded, tokens_mask = self.notes_encoder(
            encoder_input_tokens=input_tokens, encoder_inputs_mask=tokens_mask
        )

        continuous_encoded, continuous_mask = self.continuous_encoder(
            encoder_inputs=continuous_inputs, encoder_inputs_mask=continuous_mask
        )

        return [(tokens_encoded, tokens_mask), (continuous_encoded, continuous_mask)]

    def decode(self, encodings_and_masks, input_tokens, noise_time):
        timesteps = noise_time
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=input_tokens.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(input_tokens.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(input_tokens.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        logits = self.decoder(
            encodings_and_masks=encodings_and_masks, decoder_input_tokens=input_tokens, decoder_noise_time=timesteps
        )
        return logits

    @torch.no_grad()
    def __call__(
        self,
        midi_file,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 100,
        return_dict: bool = True,
        output_type: str = "numpy",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
    ) -> Union[AudioPipelineOutput, Tuple]:
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        ns = note_seq.midi_file_to_note_sequence(midi_file)
        ns_sus = note_seq.apply_sustain_control_changes(ns)

        for note in ns_sus.notes:
            if not note.is_drum:
                note.program = program_to_slakh_program(note.program)

        samples = np.zeros(int(ns_sus.total_time * SAMPLE_RATE))

        _, frame_times = audio_to_frames(samples, HOP_SIZE, FRAME_RATE)
        times, values = note_sequence_to_onsets_and_offsets_and_programs(ns_sus)

        codec = Codec(
            max_shift_steps=DEFAULT_MAX_SHIFT_SECONDS * DEFAULT_STEPS_PER_SECOND,
            steps_per_second=DEFAULT_STEPS_PER_SECOND,
            event_ranges=[
                EventRange("pitch", note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH),
                EventRange("velocity", 0, DEFAULT_NUM_VELOCITY_BINS),
                EventRange("tie", 0, 0),
                EventRange("program", note_seq.MIN_MIDI_PROGRAM, note_seq.MAX_MIDI_PROGRAM),
                EventRange("drum", note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH),
            ],
        )
        tokenizer = Tokenizer(codec.num_classes)

        events = encode_and_index_events(
            state=NoteEncodingState(),
            event_times=times,
            event_values=values,
            frame_times=frame_times,
            codec=codec,
            encode_event_fn=note_event_data_to_events,
            encoding_state_to_events_fn=note_encoding_state_to_events,
        )

        note_representation_config = NoteRepresentationConfig(onsets_only=False, include_ties=True)
        events = [note_representation_processor_chain(event, codec, note_representation_config) for event in events]
        input_tokens = [tokenizer.encode(event["inputs"]) for event in events]

        pred_mel = np.zeros([1, TARGET_FEATURE_LENGTH, self.n_dims], dtype=np.float32)
        full_pred_mel = np.zeros([1, 0, self.n_dims], np.float32)
        ones = torch.ones((1, TARGET_FEATURE_LENGTH), dtype=np.bool, device=self.device)

        for i, encoder_input_tokens in enumerate(input_tokens):
            if i == 0:
                encoder_continuous_inputs = torch.from_numpy(pred_mel[:1].copy()).to(
                    device=self.device, dtype=self.decoder.dtype
                )
                # The first chunk has no previous context.
                encoder_continuous_mask = torch.zeros((1, TARGET_FEATURE_LENGTH), dtype=np.bool, device=self.device)
            else:
                # The full song pipeline does not feed in a context feature, so the mask
                # will be all 0s after the feature converter. Because we know we're
                # feeding in a full context chunk from the previous prediction, set it
                # to all 1s.
                encoder_continuous_mask = ones

            encoder_continuous_inputs = self.scale_features(
                encoder_continuous_inputs, output_range=[-1.0, 1.0], clip=True
            )

            encodings_and_masks = self.encode(
                input_tokens=torch.IntTensor([encoder_input_tokens]).to(device=self.device),
                continuous_inputs=encoder_continuous_inputs,
                continuous_mask=encoder_continuous_mask,
            )

            # Sample encoder_continuous_inputs shaped gaussian noise to begin loop
            x = randn_tensor(
                shape=encoder_continuous_inputs.shape,
                generator=generator,
                device=self.device,
                dtype=self.decoder.dtype,
            )

            # set step values
            self.scheduler.set_timesteps(num_inference_steps)

            # Denoising diffusion loop
            for j, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
                output = self.decode(
                    encodings_and_masks=encodings_and_masks,
                    input_tokens=x,
                    noise_time=t / self.scheduler.config.num_train_timesteps,  # rescale to [0, 1)
                )

                # Compute previous output: x_t -> x_t-1
                x = self.scheduler.step(output, t, x, generator=generator).prev_sample

                # call the callback, if provided
                if callback is not None and j % callback_steps == 0:
                    callback(j, t, x)

            mel = self.scale_to_features(x, input_range=[-1.0, 1.0])
            encoder_continuous_inputs = mel[:1]
            pred_mel = mel.cpu().float().numpy()

            full_pred_mel = np.concatenate([full_pred_mel, pred_mel[:1]], axis=1)
            print("Generated segment", i)

        if output_type == "numpy" and not is_onnx_available():
            raise ValueError(
                "Cannot return output in 'np' format if ONNX is not available. Make sure to have ONNX installed or set 'output_type' to 'mel'."
            )
        elif output_type == "numpy" and self.melgan is None:
            raise ValueError(
                "Cannot return output in 'np' format if melgan component is not defined. Make sure to define `self.melgan` or set 'output_type' to 'mel'."
            )

        if output_type == "numpy":
            output = self.melgan(input_features=full_pred_mel.astype(np.float32))
        else:
            output = full_pred_mel

        if not return_dict:
            return (output,)

        return AudioPipelineOutput(audios=output)
