"""
# Cosmos 2 Predict

Download checkpoint
```bash
hf download nvidia/Cosmos-Predict2-2B-Text2Image
```

convert checkpoint
```bash
transformer_ckpt_path=~/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2-2B-Text2Image/snapshots/acdb5fde992a73ef0355f287977d002cbfd127e0/model.pt

python scripts/recipes/cosmos.py \
    --transformer_ckpt_path $transformer_ckpt_path \
    --transformer_type Cosmos-2.0-Diffusion-2B-Text2Image \
    --text_encoder_path google-t5/t5-11b \
    --tokenizer_path google-t5/t5-11b \
    --vae_type wan2.1 \
    --output_path converted/cosmos-p2-t2i-2b \
    --save_pipeline
```

# Cosmos 2.5 Predict

Download checkpoint
```bash
hf download nvidia/Cosmos-Predict2.5-2B
```

Convert checkpoint
```bash
# pre-trained
transformer_ckpt_path=~/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/865baf084d4c9e850eac59a021277d5a9b9e8b63/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt

python scripts/recipes/cosmos.py \
    --transformer_type Cosmos-2.5-Predict-Base-2B \
    --transformer_ckpt_path $transformer_ckpt_path \
    --vae_type wan2.1 \
    --output_path converted/2b/d20b7120-df3e-4911-919d-db6e08bad31c \
    --save_pipeline

# post-trained
transformer_ckpt_path=~/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/865baf084d4c9e850eac59a021277d5a9b9e8b63/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt

python scripts/recipes/cosmos.py \
    --transformer_type Cosmos-2.5-Predict-Base-2B \
    --transformer_ckpt_path $transformer_ckpt_path \
    --vae_type wan2.1 \
    --output_path converted/2b/81edfebe-bd6a-4039-8c1d-737df1a790bf \
    --save_pipeline
```

## 14B

```bash
hf download nvidia/Cosmos-Predict2.5-14B
```

```bash
# pre-trained
transformer_ckpt_path=~/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2.5-14B/snapshots/71ebf3e8af30ecfe440bf0481115975fcc052b46/base/pre-trained/54937b8c-29de-4f04-862c-e67b04ec41e8_ema_bf16.pt

python scripts/recipes/cosmos.py \
    --transformer_type Cosmos-2.5-Predict-Base-14B \
    --transformer_ckpt_path $transformer_ckpt_path \
    --vae_type wan2.1 \
    --output_path converted/14b/54937b8c-29de-4f04-862c-e67b04ec41e8/ \
    --save_pipeline

# post-trained
transformer_ckpt_path=~/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2.5-14B/snapshots/71ebf3e8af30ecfe440bf0481115975fcc052b46/base/post-trained/e21d2a49-4747-44c8-ba44-9f6f9243715f_ema_bf16.pt

python scripts/recipes/cosmos.py \
    --transformer_type Cosmos-2.5-Predict-Base-14B \
    --transformer_ckpt_path $transformer_ckpt_path \
    --vae_type wan2.1 \
    --output_path converted/14b/e21d2a49-4747-44c8-ba44-9f6f9243715f/ \
    --save_pipeline
```

# Cosmos 2.5 Transfer

Download checkpoint
```bash
hf download nvidia/Cosmos-Transfer2.5-2B
```

Convert checkpoint
```bash
# depth
transformer_ckpt_path=~/.cache/huggingface/hub/models--nvidia--Cosmos-Transfer2.5-2B/snapshots/eb5325b77d358944da58a690157dd2b8071bbf85/general/depth/626e6618-bfcd-4d9a-a077-1409e2ce353f_ema_bf16.pt

python scripts/recipes/cosmos.py \
    --transformer_type Cosmos-2.5-Transfer-General-2B \
    --transformer_ckpt_path $transformer_ckpt_path \
    --vae_type wan2.1 \
    --output_path converted/transfer/2b/general/depth/pipeline \
    --save_pipeline

python scripts/recipes/cosmos.py \
    --transformer_type Cosmos-2.5-Transfer-General-2B \
    --transformer_ckpt_path $transformer_ckpt_path \
    --vae_type wan2.1 \
    --output_path converted/transfer/2b/general/depth/models

# edge
transformer_ckpt_path=~/.cache/huggingface/hub/models--nvidia--Cosmos-Transfer2.5-2B/snapshots/eb5325b77d358944da58a690157dd2b8071bbf85/general/edge/61f5694b-0ad5-4ecd-8ad7-c8545627d125_ema_bf16.pt

python scripts/recipes/cosmos.py \
    --transformer_type Cosmos-2.5-Transfer-General-2B \
    --transformer_ckpt_path $transformer_ckpt_path \
    --vae_type wan2.1 \
    --output_path converted/transfer/2b/general/edge/pipeline \
    --save_pipeline

python scripts/recipes/cosmos.py \
    --transformer_type Cosmos-2.5-Transfer-General-2B \
    --transformer_ckpt_path $transformer_ckpt_path \
    --vae_type wan2.1 \
    --output_path converted/transfer/2b/general/edge/models

# blur
transformer_ckpt_path=~/.cache/huggingface/hub/models--nvidia--Cosmos-Transfer2.5-2B/snapshots/eb5325b77d358944da58a690157dd2b8071bbf85/general/blur/ba2f44f2-c726-4fe7-949f-597069d9b91c_ema_bf16.pt

python scripts/recipes/cosmos.py \
    --transformer_type Cosmos-2.5-Transfer-General-2B \
    --transformer_ckpt_path $transformer_ckpt_path \
    --vae_type wan2.1 \
    --output_path converted/transfer/2b/general/blur/pipeline \
    --save_pipeline

python scripts/recipes/cosmos.py \
    --transformer_type Cosmos-2.5-Transfer-General-2B \
    --transformer_ckpt_path $transformer_ckpt_path \
    --vae_type wan2.1 \
    --output_path converted/transfer/2b/general/blur/models

# seg
transformer_ckpt_path=~/.cache/huggingface/hub/models--nvidia--Cosmos-Transfer2.5-2B/snapshots/eb5325b77d358944da58a690157dd2b8071bbf85/general/seg/5136ef49-6d8d-42e8-8abf-7dac722a304a_ema_bf16.pt

python scripts/recipes/cosmos.py \
    --transformer_type Cosmos-2.5-Transfer-General-2B \
    --transformer_ckpt_path $transformer_ckpt_path \
    --vae_type wan2.1 \
    --output_path converted/transfer/2b/general/seg/pipeline \
    --save_pipeline

python scripts/recipes/cosmos.py \
    --transformer_type Cosmos-2.5-Transfer-General-2B \
    --transformer_ckpt_path $transformer_ckpt_path \
    --vae_type wan2.1 \
    --output_path converted/transfer/2b/general/seg/models
```
"""

import argparse
import pathlib
from typing import Any, Dict

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, T5EncoderModel, T5TokenizerFast

from diffusers import (
    AutoencoderKLCosmos,
    AutoencoderKLWan,
    Cosmos2TextToImagePipeline,
    Cosmos2VideoToWorldPipeline,
    CosmosControlNetModel,
    CosmosTextToWorldPipeline,
    CosmosTransformer3DModel,
    CosmosVideoToWorldPipeline,
    EDMEulerScheduler,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
)
from diffusers.loaders.conversion import get_conversion
from diffusers.loaders.conversion.configs.cosmos import CONTROLNET_CONFIGS, TRANSFORMER_CONFIGS, VAE_CONFIGS
from diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict import Cosmos2_5_PredictBasePipeline
from diffusers.pipelines.cosmos.pipeline_cosmos2_5_transfer import Cosmos2_5_TransferPipeline


def get_state_dict(saved_dict: Dict[str, Any]) -> dict[str, Any]:
    state_dict = saved_dict
    if "model" in saved_dict.keys():
        state_dict = state_dict["model"]
    if "module" in saved_dict.keys():
        state_dict = state_dict["module"]
    if "state_dict" in saved_dict.keys():
        state_dict = state_dict["state_dict"]
    return state_dict


def convert_transformer(transformer_type: str, state_dict=None, weights_only=True):
    config = dict(TRANSFORMER_CONFIGS[transformer_type])
    config["original_format"] = "cosmos1" if "Cosmos-1.0" in transformer_type else "cosmos2"
    return CosmosTransformer3DModel.from_single_file(state_dict, config=config)


def convert_controlnet(transformer_type: str, control_state_dict, base_state_dict, weights_only=True):
    config = CONTROLNET_CONFIGS[transformer_type]
    conversion = get_conversion("CosmosControlNetModel", config)
    base_config = {**TRANSFORMER_CONFIGS[transformer_type], "original_format": "cosmos2"}
    base = get_conversion("CosmosTransformer3DModel", base_config).to_original(base_state_dict)
    state = {key.removeprefix("net."): value for key, value in control_state_dict.items()}
    # The original ControlNet omits modules shared with the separately loaded base transformer.
    for key in conversion.original_keys:
        if key.startswith("base."):
            state[key] = base[key.removeprefix("base.")]
    return CosmosControlNetModel.from_single_file(state, config=config)


def convert_vae(vae_type: str):
    model_name = VAE_CONFIGS[vae_type]["name"]
    snapshot_directory = snapshot_download(model_name, repo_type="model")
    directory = pathlib.Path(snapshot_directory)

    autoencoder_file = directory / "autoencoder.jit"
    mean_std_file = directory / "mean_std.pt"

    original_state_dict = torch.jit.load(autoencoder_file.as_posix()).state_dict()
    if mean_std_file.exists():
        mean_std = torch.load(mean_std_file, map_location="cpu", weights_only=True)
    else:
        mean_std = (None, None)

    config = dict(VAE_CONFIGS[vae_type]["diffusers_config"])
    config.update(
        {
            "latents_mean": mean_std[0].detach().cpu().numpy().tolist(),
            "latents_std": mean_std[1].detach().cpu().numpy().tolist(),
        }
    )
    return AutoencoderKLCosmos.from_single_file(original_state_dict, config=config)


def save_pipeline_cosmos_1_0(args, transformer, vae):
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_path, torch_dtype=torch.bfloat16)
    tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer_path)
    # The original code initializes EDM config with sigma_min=0.0002, but does not make use of it anywhere directly.
    # So, the sigma_min values that is used is the default value of 0.002.
    scheduler = EDMEulerScheduler(
        sigma_min=0.002,
        sigma_max=80,
        sigma_data=0.5,
        sigma_schedule="karras",
        num_train_timesteps=1000,
        prediction_type="epsilon",
        rho=7.0,
        final_sigmas_type="sigma_min",
    )

    pipe_cls = CosmosTextToWorldPipeline if "Text2World" in args.transformer_type else CosmosVideoToWorldPipeline
    pipe = pipe_cls(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        safety_checker=lambda *args, **kwargs: None,
    )
    pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")


def save_pipeline_cosmos_2_0(args, transformer, vae):
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_path, torch_dtype=torch.bfloat16)
    tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer_path)

    scheduler = FlowMatchEulerDiscreteScheduler(use_karras_sigmas=True)

    pipe_cls = Cosmos2TextToImagePipeline if "Text2Image" in args.transformer_type else Cosmos2VideoToWorldPipeline
    pipe = pipe_cls(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        safety_checker=lambda *args, **kwargs: None,
    )
    pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")


def save_pipeline_cosmos2_5_predict(args, transformer, vae):
    text_encoder_path = args.text_encoder_path or "nvidia/Cosmos-Reason1-7B"
    tokenizer_path = args.tokenizer_path or "Qwen/Qwen2.5-VL-7B-Instruct"

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        text_encoder_path, torch_dtype="auto", device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    scheduler = UniPCMultistepScheduler(
        use_karras_sigmas=True,
        use_flow_sigmas=True,
        prediction_type="flow_prediction",
        sigma_max=200.0,
        sigma_min=0.01,
    )

    pipe = Cosmos2_5_PredictBasePipeline(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        safety_checker=lambda *args, **kwargs: None,
    )
    pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")


def save_pipeline_cosmos2_5_transfer(args, transformer, controlnet, vae):
    text_encoder_path = args.text_encoder_path or "nvidia/Cosmos-Reason1-7B"
    tokenizer_path = args.tokenizer_path or "Qwen/Qwen2.5-VL-7B-Instruct"

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        text_encoder_path, torch_dtype="auto", device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    scheduler = UniPCMultistepScheduler(
        use_karras_sigmas=True,
        use_flow_sigmas=True,
        prediction_type="flow_prediction",
        sigma_max=200.0,
        sigma_min=0.01,
    )

    pipe = Cosmos2_5_TransferPipeline(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        controlnet=controlnet,
        vae=vae,
        scheduler=scheduler,
        safety_checker=lambda *args, **kwargs: None,
    )
    pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer_type", type=str, default=None, choices=list(TRANSFORMER_CONFIGS.keys()))
    parser.add_argument(
        "--transformer_ckpt_path", type=str, default=None, help="Path to original transformer checkpoint"
    )
    parser.add_argument(
        "--vae_type", type=str, default="wan2.1", choices=["wan2.1", *list(VAE_CONFIGS.keys())], help="Type of VAE"
    )
    parser.add_argument("--text_encoder_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--save_pipeline", action="store_true")
    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model should be saved")
    parser.add_argument("--dtype", default="bf16", help="Torch dtype to save the transformer in.")
    return parser.parse_args()


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


if __name__ == "__main__":
    args = get_args()

    transformer = None
    controlnet = None
    dtype = DTYPE_MAPPING[args.dtype]

    if args.save_pipeline:
        assert args.transformer_ckpt_path is not None
        assert args.vae_type is not None

    raw_state_dict = None
    if args.transformer_ckpt_path is not None:
        weights_only = "Cosmos-1.0" in args.transformer_type
        raw_state_dict = get_state_dict(
            torch.load(args.transformer_ckpt_path, map_location="cpu", weights_only=weights_only)
        )

    if raw_state_dict is not None:
        if "Transfer" in args.transformer_type:
            base_state_dict = {}
            control_state_dict = {}
            for k, v in raw_state_dict.items():
                plain_key = k.removeprefix("net.") if k.startswith("net.") else k
                if "control" in plain_key.lower():
                    control_state_dict[k] = v
                else:
                    base_state_dict[k] = v
            assert len(base_state_dict.keys() & control_state_dict.keys()) == 0

            # Convert transformer first to get the processed base state dict
            transformer = convert_transformer(
                args.transformer_type, state_dict=base_state_dict, weights_only=weights_only
            )
            transformer = transformer.to(dtype=dtype)

            # Get converted transformer state dict to copy shared weights to controlnet
            converted_base_state_dict = transformer.state_dict()

            # Convert controlnet with both control-specific and shared weights from transformer
            controlnet = convert_controlnet(
                args.transformer_type, control_state_dict, converted_base_state_dict, weights_only=weights_only
            )
            controlnet = controlnet.to(dtype=dtype)

            if not args.save_pipeline:
                transformer.save_pretrained(
                    pathlib.Path(args.output_path) / "transformer", safe_serialization=True, max_shard_size="5GB"
                )
                controlnet.save_pretrained(
                    pathlib.Path(args.output_path) / "controlnet", safe_serialization=True, max_shard_size="5GB"
                )
        else:
            transformer = convert_transformer(
                args.transformer_type, state_dict=raw_state_dict, weights_only=weights_only
            )
            transformer = transformer.to(dtype=dtype)
            if not args.save_pipeline:
                transformer.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    if args.vae_type is not None:
        if "Cosmos-1.0" in args.transformer_type:
            vae = convert_vae(args.vae_type)
        elif "Cosmos-2.0" in args.transformer_type or "Cosmos-2.5" in args.transformer_type:
            vae = AutoencoderKLWan.from_pretrained(
                "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=torch.float32
            )
        else:
            raise AssertionError(f"{args.transformer_type} not supported")

        if not args.save_pipeline:
            vae.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")
    else:
        vae = None

    if args.save_pipeline:
        if "Cosmos-1.0" in args.transformer_type:
            assert args.text_encoder_path is not None
            assert args.tokenizer_path is not None
            save_pipeline_cosmos_1_0(args, transformer, vae)
        elif "Cosmos-2.0" in args.transformer_type:
            assert args.text_encoder_path is not None
            assert args.tokenizer_path is not None
            save_pipeline_cosmos_2_0(args, transformer, vae)
        elif "Cosmos-2.5" in args.transformer_type:
            if "Predict" in args.transformer_type:
                save_pipeline_cosmos2_5_predict(args, transformer, vae)
            elif "Transfer" in args.transformer_type:
                save_pipeline_cosmos2_5_transfer(args, transformer, None, vae)
            else:
                raise AssertionError(f"{args.transformer_type} not supported")
        else:
            raise AssertionError(f"{args.transformer_type} not supported")
