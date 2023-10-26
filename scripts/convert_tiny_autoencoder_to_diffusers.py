import argparse

import safetensors.torch

from diffusers import AutoencoderTiny


"""
Example - From the diffusers root directory:

Download the weights:
```sh
$ wget -q https://huggingface.co/madebyollin/taesd/resolve/main/taesd_encoder.safetensors
$ wget -q https://huggingface.co/madebyollin/taesd/resolve/main/taesd_decoder.safetensors
```

Convert the model:
```sh
$ python scripts/convert_tiny_autoencoder_to_diffusers.py \
    --encoder_ckpt_path  taesd_encoder.safetensors \
    --decoder_ckpt_path taesd_decoder.safetensors \
    --dump_path taesd-diffusers
```
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument(
        "--encoder_ckpt_path",
        default=None,
        type=str,
        required=True,
        help="Path to the encoder ckpt.",
    )
    parser.add_argument(
        "--decoder_ckpt_path",
        default=None,
        type=str,
        required=True,
        help="Path to the decoder ckpt.",
    )
    parser.add_argument(
        "--use_safetensors", action="store_true", help="Whether to serialize in the safetensors format."
    )
    args = parser.parse_args()

    print("Loading the original state_dicts of the encoder and the decoder...")
    encoder_state_dict = safetensors.torch.load_file(args.encoder_ckpt_path)
    decoder_state_dict = safetensors.torch.load_file(args.decoder_ckpt_path)

    print("Populating the state_dicts in the diffusers format...")
    tiny_autoencoder = AutoencoderTiny()
    new_state_dict = {}

    # Modify the encoder state dict.
    for k in encoder_state_dict:
        new_state_dict.update({f"encoder.layers.{k}": encoder_state_dict[k]})

    # Modify the decoder state dict.
    for k in decoder_state_dict:
        layer_id = int(k.split(".")[0]) - 1
        new_k = str(layer_id) + "." + ".".join(k.split(".")[1:])
        new_state_dict.update({f"decoder.layers.{new_k}": decoder_state_dict[k]})

    # Assertion tests with the original implementation can be found here:
    # https://gist.github.com/sayakpaul/337b0988f08bd2cf2b248206f760e28f
    tiny_autoencoder.load_state_dict(new_state_dict)
    print("Population successful, serializing...")
    tiny_autoencoder.save_pretrained(args.dump_path, safe_serialization=args.use_safetensors)
