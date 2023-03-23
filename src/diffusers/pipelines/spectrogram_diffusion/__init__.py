# flake8: noqa
from ...utils import is_note_seq_available

from .notes_encoder import SpectrogramNotesEncoder
from .continous_encoder import SpectrogramContEncoder
from .pipeline_spectrogram_diffusion import (
    SpectrogramContEncoder,
    SpectrogramDiffusionPipeline,
    T5FilmDecoder,
)

if is_note_seq_available():
    from .midi_utils import MidiProcessor
