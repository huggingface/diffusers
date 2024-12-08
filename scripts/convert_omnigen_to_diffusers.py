import argparse
import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, OmniGenPipeline, OmniGenTransformer2DModel


def main(args):
    # checkpoint from https://huggingface.co/Shitao/OmniGen-v1

    if not os.path.exists(args.origin_ckpt_path):
        print("Model not found, downloading...")
        cache_folder = os.getenv('HF_HUB_CACHE')
        args.origin_ckpt_path = snapshot_download(repo_id=args.origin_ckpt_path,
                                                  cache_dir=cache_folder,
                                                  ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5',
                                                                   'model.pt'])
        print(f"Downloaded model to {args.origin_ckpt_path}")

    ckpt = os.path.join(args.origin_ckpt_path, 'model.safetensors')
    ckpt = load_file(ckpt, device="cpu")

    mapping_dict = {
        "pos_embed": "patch_embedding.pos_embed",
        "x_embedder.proj.weight": "patch_embedding.output_image_proj.weight",
        "x_embedder.proj.bias": "patch_embedding.output_image_proj.bias",
        "input_x_embedder.proj.weight": "patch_embedding.input_image_proj.weight",
        "input_x_embedder.proj.bias": "patch_embedding.input_image_proj.bias",
        "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
        "final_layer.adaLN_modulation.1.bias": "norm_out.linear.bias",
        "final_layer.linear.weight": "proj_out.weight",
        "final_layer.linear.bias": "proj_out.bias",

    }

    converted_state_dict = {}
    for k, v in ckpt.items():
        if k in mapping_dict:
            converted_state_dict[mapping_dict[k]] = v
        else:
            converted_state_dict[k] = v

    # transformer_config = AutoConfig.from_pretrained(args.origin_ckpt_path)
    # print(type(transformer_config.__dict__))
    # print(transformer_config.__dict__)

    transformer_config = {
        "_name_or_path": "Phi-3-vision-128k-instruct",
        "architectures": [
            "Phi3ForCausalLM"
        ],
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 3072,
        "initializer_range": 0.02,
        "intermediate_size": 8192,
        "max_position_embeddings": 131072,
        "model_type": "phi3",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "original_max_position_embeddings": 4096,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "long_factor": [
                1.0299999713897705,
                1.0499999523162842,
                1.0499999523162842,
                1.0799999237060547,
                1.2299998998641968,
                1.2299998998641968,
                1.2999999523162842,
                1.4499999284744263,
                1.5999999046325684,
                1.6499998569488525,
                1.8999998569488525,
                2.859999895095825,
                3.68999981880188,
                5.419999599456787,
                5.489999771118164,
                5.489999771118164,
                9.09000015258789,
                11.579999923706055,
                15.65999984741211,
                15.769999504089355,
                15.789999961853027,
                18.360000610351562,
                21.989999771118164,
                23.079999923706055,
                30.009998321533203,
                32.35000228881836,
                32.590003967285156,
                35.56000518798828,
                39.95000457763672,
                53.840003967285156,
                56.20000457763672,
                57.95000457763672,
                59.29000473022461,
                59.77000427246094,
                59.920005798339844,
                61.190006256103516,
                61.96000671386719,
                62.50000762939453,
                63.3700065612793,
                63.48000717163086,
                63.48000717163086,
                63.66000747680664,
                63.850006103515625,
                64.08000946044922,
                64.760009765625,
                64.80001068115234,
                64.81001281738281,
                64.81001281738281
            ],
            "short_factor": [
                1.05,
                1.05,
                1.05,
                1.1,
                1.1,
                1.1,
                1.2500000000000002,
                1.2500000000000002,
                1.4000000000000004,
                1.4500000000000004,
                1.5500000000000005,
                1.8500000000000008,
                1.9000000000000008,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.000000000000001,
                2.1000000000000005,
                2.1000000000000005,
                2.2,
                2.3499999999999996,
                2.3499999999999996,
                2.3499999999999996,
                2.3499999999999996,
                2.3999999999999995,
                2.3999999999999995,
                2.6499999999999986,
                2.6999999999999984,
                2.8999999999999977,
                2.9499999999999975,
                3.049999999999997,
                3.049999999999997,
                3.049999999999997
            ],
            "type": "su"
        },
        "rope_theta": 10000.0,
        "sliding_window": 131072,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.38.1",
        "use_cache": True,
        "vocab_size": 32064,
        "_attn_implementation": "sdpa"
    }
    transformer = OmniGenTransformer2DModel(
        transformer_config=transformer_config,
        patch_size=2,
        in_channels=4,
        pos_embed_max_size=192,
    )
    transformer.load_state_dict(converted_state_dict, strict=True)
    transformer.to(torch.bfloat16)

    num_model_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total number of transformer parameters: {num_model_params}")

    scheduler = FlowMatchEulerDiscreteScheduler()

    vae = AutoencoderKL.from_pretrained(os.path.join(args.origin_ckpt_path, "vae"), torch_dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.origin_ckpt_path)

    pipeline = OmniGenPipeline(
        tokenizer=tokenizer, transformer=transformer, vae=vae, scheduler=scheduler
    )
    pipeline.save_pretrained(args.dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--origin_ckpt_path", default="Shitao/OmniGen-v1", type=str, required=False,
        help="Path to the checkpoint to convert."
    )

    parser.add_argument("--dump_path", default="OmniGen-v1-diffusers", type=str, required=False,
                        help="Path to the output pipeline.")

    args = parser.parse_args()
    main(args)
