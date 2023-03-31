# flake8: noqa
from ...utils import is_note_seq_available, is_transformers_available
from ...utils import OptionalDependencyNotAvailable


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .notes_encoder import SpectrogramNotesEncoder
    from .continous_encoder import SpectrogramContEncoder
    from .pipeline_spectrogram_diffusion import (
        SpectrogramContEncoder,
        SpectrogramDiffusionPipeline,
        T5FilmDecoder,
    )

try:
    if not (is_transformers_available() and is_torch_available() and is_note_seq_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_transformers_and_torch_and_note_seq_objects import *  # noqa F403
else:
    from .midi_utils import MidiProcessor
