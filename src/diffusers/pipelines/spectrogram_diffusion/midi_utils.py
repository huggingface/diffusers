from typing import Sequence, Tuple, Optional, MutableMapping, List, Callable, Mapping, Any
import dataclasses
import math
from immutabledict import immutabledict

import numpy as np
import note_seq
import torch
import torch.nn.functional as F


INPUT_FEATURE_LENGTH = 2048
TARGET_FEATURE_LENGTH = 256

SAMPLE_RATE = 16000
HOP_SIZE = 320
FRAME_RATE = int(SAMPLE_RATE // HOP_SIZE)

DEFAULT_STEPS_PER_SECOND = 100
DEFAULT_MAX_SHIFT_SECONDS = 10
DEFAULT_NUM_VELOCITY_BINS = 1

SLAKH_CLASS_PROGRAMS = immutabledict(
    {
        "Acoustic Piano": 0,
        "Electric Piano": 4,
        "Chromatic Percussion": 8,
        "Organ": 16,
        "Acoustic Guitar": 24,
        "Clean Electric Guitar": 26,
        "Distorted Electric Guitar": 29,
        "Acoustic Bass": 32,
        "Electric Bass": 33,
        "Violin": 40,
        "Viola": 41,
        "Cello": 42,
        "Contrabass": 43,
        "Orchestral Harp": 46,
        "Timpani": 47,
        "String Ensemble": 48,
        "Synth Strings": 50,
        "Choir and Voice": 52,
        "Orchestral Hit": 55,
        "Trumpet": 56,
        "Trombone": 57,
        "Tuba": 58,
        "French Horn": 60,
        "Brass Section": 61,
        "Soprano/Alto Sax": 64,
        "Tenor Sax": 66,
        "Baritone Sax": 67,
        "Oboe": 68,
        "English Horn": 69,
        "Bassoon": 70,
        "Clarinet": 71,
        "Pipe": 73,
        "Synth Lead": 80,
        "Synth Pad": 88,
    }
)


@dataclasses.dataclass
class NoteRepresentationConfig:
    """Configuration note representations."""

    onsets_only: bool
    include_ties: bool


@dataclasses.dataclass
class NoteEventData:
    pitch: int
    velocity: Optional[int] = None
    program: Optional[int] = None
    is_drum: Optional[bool] = None
    instrument: Optional[int] = None


@dataclasses.dataclass
class NoteEncodingState:
    """Encoding state for note transcription, keeping track of active pitches."""

    # velocity bin for active pitches and programs
    active_pitches: MutableMapping[Tuple[int, int], int] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class EventRange:
    type: str
    min_value: int
    max_value: int


@dataclasses.dataclass
class Event:
    type: str
    value: int


class Tokenizer:
    def __init__(self, regular_ids: int):
        # The special tokens: 0=PAD, 1=EOS, and 2=UNK
        self._num_special_tokens = 3
        self._num_regular_tokens = regular_ids

    def encode(self, token_ids):
        encoded = []
        for token_id in token_ids:
            if not 0 <= token_id < self._num_regular_tokens:
                raise ValueError(
                    f"token_id {token_id} does not fall within valid range of " f"[0, {self._num_regular_tokens})"
                )
            encoded.append(token_id + self._num_special_tokens)

        # Add EOS token
        encoded.append(1)

        # Pad to till INPUT_FEATURE_LENGTH
        encoded = encoded + [0] * (INPUT_FEATURE_LENGTH - len(encoded))

        return encoded


class Codec:
    """Encode and decode events.

    Useful for declaring what certain ranges of a vocabulary should be used for.
    This is intended to be used from Python before encoding or after decoding with
    GenericTokenVocabulary. This class is more lightweight and does not include
    things like EOS or UNK token handling.

    To ensure that 'shift' events are always the first block of the vocab and
    start at 0, that event type is required and specified separately.
    """

    def __init__(self, max_shift_steps: int, steps_per_second: float, event_ranges: List[EventRange]):
        """Define Codec.

        Args:
          max_shift_steps: Maximum number of shift steps that can be encoded.
          steps_per_second: Shift steps will be interpreted as having a duration of
              1 / steps_per_second.
          event_ranges: Other supported event types and their ranges.
        """
        self.steps_per_second = steps_per_second
        self._shift_range = EventRange(type="shift", min_value=0, max_value=max_shift_steps)
        self._event_ranges = [self._shift_range] + event_ranges
        # Ensure all event types have unique names.
        assert len(self._event_ranges) == len(set([er.type for er in self._event_ranges]))

    @property
    def num_classes(self) -> int:
        return sum(er.max_value - er.min_value + 1 for er in self._event_ranges)

    # The next couple methods are simplified special case methods just for shift
    # events that are intended to be used from within autograph functions.

    def is_shift_event_index(self, index: int) -> bool:
        return (self._shift_range.min_value <= index) and (index <= self._shift_range.max_value)

    @property
    def max_shift_steps(self) -> int:
        return self._shift_range.max_value

    def encode_event(self, event: Event) -> int:
        """Encode an event to an index."""
        offset = 0
        for er in self._event_ranges:
            if event.type == er.type:
                if not er.min_value <= event.value <= er.max_value:
                    raise ValueError(
                        f"Event value {event.value} is not within valid range "
                        f"[{er.min_value}, {er.max_value}] for type {event.type}"
                    )
                return offset + event.value - er.min_value
            offset += er.max_value - er.min_value + 1

        raise ValueError(f"Unknown event type: {event.type}")

    def event_type_range(self, event_type: str) -> Tuple[int, int]:
        """Return [min_id, max_id] for an event type."""
        offset = 0
        for er in self._event_ranges:
            if event_type == er.type:
                return offset, offset + (er.max_value - er.min_value)
            offset += er.max_value - er.min_value + 1

        raise ValueError(f"Unknown event type: {event_type}")

    def decode_event_index(self, index: int) -> Event:
        """Decode an event index to an Event."""
        offset = 0
        for er in self._event_ranges:
            if offset <= index <= offset + er.max_value - er.min_value:
                return Event(type=er.type, value=er.min_value + index - offset)
            offset += er.max_value - er.min_value + 1

        raise ValueError(f"Unknown event index: {index}")


def frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
    """
    equivalent of tf.signal.frame
    """
    signal_length = signal.shape[axis]
    if pad_end:
        frames_overlap = frame_length - frame_step
        rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
        pad_size = int(frame_length - rest_samples)

        if pad_size != 0:
            pad_axis = [0] * signal.ndim
            pad_axis[axis] = pad_size
            signal = F.pad(signal, pad_axis, "constant", pad_value)
    frames = signal.unfold(axis, frame_length, frame_step)
    return frames


def program_to_slakh_program(program):
    # this is done very hackily, probably should use a custom mapping
    for slakh_program in sorted(SLAKH_CLASS_PROGRAMS.values(), reverse=True):
        if program >= slakh_program:
            return slakh_program


def audio_to_frames(
    samples,
    hop_size: int,
    frame_rate: int,
) -> Tuple[Sequence[Sequence[int]], torch.Tensor]:
    """Convert audio samples to non-overlapping frames and frame times."""
    frame_size = hop_size
    samples = np.pad(samples, [0, frame_size - len(samples) % frame_size], mode="constant")

    # Split audio into frames.
    frames = frame(
        torch.Tensor(samples).unsqueeze(0),
        frame_length=frame_size,
        frame_step=frame_size,
        pad_end=False,  # TODO check why its off by 1 here when True
    )

    num_frames = len(samples) // frame_size

    times = np.arange(num_frames) / frame_rate
    return frames, times


def note_sequence_to_onsets_and_offsets_and_programs(
    ns: note_seq.NoteSequence,
) -> Tuple[Sequence[float], Sequence[NoteEventData]]:
    """Extract onset & offset times and pitches & programs from a NoteSequence.

    The onset & offset times will not necessarily be in sorted order.

    Args:
      ns: NoteSequence from which to extract onsets and offsets.

    Returns:
      times: A list of note onset and offset times.
      values: A list of NoteEventData objects where velocity is zero for note
          offsets.
    """
    # Sort by program and pitch and put offsets before onsets as a tiebreaker for
    # subsequent stable sort.
    notes = sorted(ns.notes, key=lambda note: (note.is_drum, note.program, note.pitch))
    times = [note.end_time for note in notes if not note.is_drum] + [note.start_time for note in notes]
    values = [
        NoteEventData(pitch=note.pitch, velocity=0, program=note.program, is_drum=False)
        for note in notes
        if not note.is_drum
    ] + [
        NoteEventData(pitch=note.pitch, velocity=note.velocity, program=note.program, is_drum=note.is_drum)
        for note in notes
    ]
    return times, values


def num_velocity_bins_from_codec(codec: Codec):
    """Get number of velocity bins from event codec."""
    lo, hi = codec.event_type_range("velocity")
    return hi - lo


# segment an array into segments of length n
def segment(a, n):
    return [a[i : i + n] for i in range(0, len(a), n)]


def velocity_to_bin(velocity, num_velocity_bins):
    if velocity == 0:
        return 0
    else:
        return math.ceil(num_velocity_bins * velocity / note_seq.MAX_MIDI_VELOCITY)


def note_event_data_to_events(
    state: Optional[NoteEncodingState],
    value: NoteEventData,
    codec: Codec,
) -> Sequence[Event]:
    """Convert note event data to a sequence of events."""
    if value.velocity is None:
        # onsets only, no program or velocity
        return [Event("pitch", value.pitch)]
    else:
        num_velocity_bins = num_velocity_bins_from_codec(codec)
        velocity_bin = velocity_to_bin(value.velocity, num_velocity_bins)
        if value.program is None:
            # onsets + offsets + velocities only, no programs
            if state is not None:
                state.active_pitches[(value.pitch, 0)] = velocity_bin
            return [Event("velocity", velocity_bin), Event("pitch", value.pitch)]
        else:
            if value.is_drum:
                # drum events use a separate vocabulary
                return [Event("velocity", velocity_bin), Event("drum", value.pitch)]
            else:
                # program + velocity + pitch
                if state is not None:
                    state.active_pitches[(value.pitch, value.program)] = velocity_bin
                return [
                    Event("program", value.program),
                    Event("velocity", velocity_bin),
                    Event("pitch", value.pitch),
                ]


def note_encoding_state_to_events(state: NoteEncodingState) -> Sequence[Event]:
    """Output program and pitch events for active notes plus a final tie event."""
    events = []
    for pitch, program in sorted(state.active_pitches.keys(), key=lambda k: k[::-1]):
        if state.active_pitches[(pitch, program)]:
            events += [Event("program", program), Event("pitch", pitch)]
    events.append(Event("tie", 0))
    return events


def encode_and_index_events(
    state, event_times, event_values, codec, frame_times, encode_event_fn, encoding_state_to_events_fn=None
):
    """Encode a sequence of timed events and index to audio frame times.

    Encodes time shifts as repeated single step shifts for later run length
    encoding.

    Optionally, also encodes a sequence of "state events", keeping track of the
    current encoding state at each audio frame. This can be used e.g. to prepend
    events representing the current state to a targets segment.

    Args:
      state: Initial event encoding state.
      event_times: Sequence of event times.
      event_values: Sequence of event values.
      encode_event_fn: Function that transforms event value into a sequence of one
          or more Event objects.
      codec: An Codec object that maps Event objects to indices.
      frame_times: Time for every audio frame.
      encoding_state_to_events_fn: Function that transforms encoding state into a
          sequence of one or more Event objects.

    Returns:
      events: Encoded events and shifts.
      event_start_indices: Corresponding start event index for every audio frame.
          Note: one event can correspond to multiple audio indices due to sampling
          rate differences. This makes splitting sequences tricky because the same
          event can appear at the end of one sequence and the beginning of
          another.
      event_end_indices: Corresponding end event index for every audio frame. Used
          to ensure when slicing that one chunk ends where the next begins. Should
          always be true that event_end_indices[i] = event_start_indices[i + 1].
      state_events: Encoded "state" events representing the encoding state before
          each event.
      state_event_indices: Corresponding state event index for every audio frame.
    """
    indices = np.argsort(event_times, kind="stable")
    event_steps = [round(event_times[i] * codec.steps_per_second) for i in indices]
    event_values = [event_values[i] for i in indices]

    events = []
    state_events = []
    event_start_indices = []
    state_event_indices = []

    cur_step = 0
    cur_event_idx = 0
    cur_state_event_idx = 0

    def fill_event_start_indices_to_cur_step():
        while (
            len(event_start_indices) < len(frame_times)
            and frame_times[len(event_start_indices)] < cur_step / codec.steps_per_second
        ):
            event_start_indices.append(cur_event_idx)
            state_event_indices.append(cur_state_event_idx)

    for event_step, event_value in zip(event_steps, event_values):
        while event_step > cur_step:
            events.append(codec.encode_event(Event(type="shift", value=1)))
            cur_step += 1
            fill_event_start_indices_to_cur_step()
            cur_event_idx = len(events)
            cur_state_event_idx = len(state_events)
        if encoding_state_to_events_fn:
            # Dump state to state events *before* processing the next event, because
            # we want to capture the state prior to the occurrence of the event.
            for e in encoding_state_to_events_fn(state):
                state_events.append(codec.encode_event(e))

        for e in encode_event_fn(state, event_value, codec):
            events.append(codec.encode_event(e))

    # After the last event, continue filling out the event_start_indices array.
    # The inequality is not strict because if our current step lines up exactly
    # with (the start of) an audio frame, we need to add an additional shift event
    # to "cover" that frame.
    while cur_step / codec.steps_per_second <= frame_times[-1]:
        events.append(codec.encode_event(Event(type="shift", value=1)))
        cur_step += 1
        fill_event_start_indices_to_cur_step()
        cur_event_idx = len(events)

    # Now fill in event_end_indices. We need this extra array to make sure that
    # when we slice events, each slice ends exactly where the subsequent slice
    # begins.
    event_end_indices = event_start_indices[1:] + [len(events)]

    events = np.array(events).astype(np.int32)
    state_events = np.array(state_events).astype(np.int32)
    event_start_indices = segment(np.array(event_start_indices).astype(np.int32), TARGET_FEATURE_LENGTH)
    event_end_indices = segment(np.array(event_end_indices).astype(np.int32), TARGET_FEATURE_LENGTH)
    state_event_indices = segment(np.array(state_event_indices).astype(np.int32), TARGET_FEATURE_LENGTH)

    outputs = []
    for start_indices, end_indices, event_indices in zip(event_start_indices, event_end_indices, state_event_indices):
        outputs.append(
            {
                "inputs": events,
                "event_start_indices": start_indices,
                "event_end_indices": end_indices,
                "state_events": state_events,
                "state_event_indices": event_indices,
            }
        )

    return outputs
