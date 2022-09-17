import math

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


module = hub.KerasLayer("https://tfhub.dev/google/soundstream/mel/decoder/music/1")

# 1. Convert the TF weights of SOUNDSTREAM to PyTorch
# This will give us the necessary vocoder


# 2. Convert JAX T5 weights to Pytorch using the transformers script
# This will give us the necessary encoder and decoder
# Then encoder corresponds to the note encoder  and the decoder part is the spectrogram decoder

# 3. Convert eh Context Encoder weights to Pytorch
# The context encoder should be pretty straightforward to convert

# 4. Implement tests to make sure that the models work properly


SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 320
WIN_LENGTH = 640
N_MEL_CHANNELS = 128
MEL_FMIN = 0.0
MEL_FMAX = int(SAMPLE_RATE // 2)
CLIP_VALUE_MIN = 1e-5
CLIP_VALUE_MAX = 1e8

MEL_BASIS = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=N_MEL_CHANNELS,
    num_spectrogram_bins=N_FFT // 2 + 1,
    sample_rate=SAMPLE_RATE,
    lower_edge_hertz=MEL_FMIN,
    upper_edge_hertz=MEL_FMAX,
)


def calculate_spectrogram(samples):
    """Calculate mel spectrogram using the parameters the model expects."""
    fft = tf.signal.stft(
        samples,
        frame_length=WIN_LENGTH,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
        pad_end=True,
    )
    fft_modulus = tf.abs(fft)

    output = tf.matmul(fft_modulus, MEL_BASIS)

    output = tf.clip_by_value(output, clip_value_min=CLIP_VALUE_MIN, clip_value_max=CLIP_VALUE_MAX)
    output = tf.math.log(output)
    return output


# Load a music sample from the GTZAN dataset.
gtzan = tfds.load("gtzan", split="train")
# Convert an example from int to float.
samples = tf.cast(next(iter(gtzan))["audio"] / 32768, dtype=tf.float32)
# Add batch dimension.
samples = tf.expand_dims(samples, axis=0)
# Compute a mel-spectrogram.
spectrogram = calculate_spectrogram(samples)
# Reconstruct the audio from a mel-spectrogram using a SoundStream decoder.
reconstructed_samples = module(spectrogram)
