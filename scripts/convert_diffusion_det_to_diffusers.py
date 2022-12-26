"""
This script ports models from Diffusion-Det (https://github.com/ShoufaChen/DiffusionDet) to diffusers.
"""

import os
import torch
import numpy as np
import torch
from transformers import SwinModel


MODEL_DOWNLOAD_MAP = {
    'diffdet_coco_swinbase': 'https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_swinbase.pth'
}

def download(name):
    if not os.path.isfile(f'./{name}.pth'):
        os.system(f'wget {MODEL_DOWNLOAD_MAP[name]} ./')
    return f'./{name}.pth' 

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = None
    if args.encoder_type == 'swin':
        model_path = download('diffdet_coco_swinbase')
    assert model_path is not None

    checkpoint = torch.load(model_path)
    for key in checkpoint['model']:
        print(key)


def convert_pretrained_HFSwinBackbone():
  orig_ckpt_path = download('diffdet_coco_swinbase')
  assert orig_ckpt_path is not None
  orig_model = torch.load(orig_ckpt_path)
  orig_model_state = orig_model['model']
  hf_model = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
  hf_model_state = hf_model.state_dict()
  state_dict = {}
  def _load_weights(src_name, dest_name):
    state_dict.update({
        dest_name: orig_model_state[src_name]
    })
  # load positional embedding weights.
  _load_weights('backbone.bottom_up.patch_embed.proj.weight', 'embeddings.patch_embeddings.projection.weight')
  _load_weights('backbone.bottom_up.patch_embed.proj.bias', 'embeddings.patch_embeddings.projection.bias')
  _load_weights('backbone.bottom_up.patch_embed.norm.weight', 'embeddings.norm.weight')
  _load_weights('backbone.bottom_up.patch_embed.norm.bias', 'embeddings.norm.bias')

  # load SwinEncoder:
  # loading layer-0
  def _load_swin_layers(layer_id, num_blocks, downsample=True):
    for block_id in range(num_blocks):
      src_name_prefix = f'backbone.bottom_up.layers.{layer_id}.blocks.{block_id}'
      dest_name_prefix = f'encoder.layers.{layer_id}.blocks.{block_id}'
      _load_weights(f'{src_name_prefix}.norm1.weight', f'{dest_name_prefix}.layernorm_before.weight')
      _load_weights(f'{src_name_prefix}.norm1.bias', f'{dest_name_prefix}.layernorm_before.bias')
      _load_weights(f'{src_name_prefix}.attn.relative_position_bias_table', f'{dest_name_prefix}.attention.self.relative_position_bias_table')
      _load_weights(f'{src_name_prefix}.attn.relative_position_index', f'{dest_name_prefix}.attention.self.relative_position_index')

      # self attention weights.
      orig_size = orig_model_state[f'{src_name_prefix}.attn.qkv.weight'].shape[0]
      q_w, k_w, v_w = torch.split(orig_model_state[f'{src_name_prefix}.attn.qkv.weight'], [orig_size//3, orig_size//3, orig_size//3], dim=0)
      q_b, k_b, v_b = torch.split(orig_model_state[f'{src_name_prefix}.attn.qkv.bias'], [orig_size//3, orig_size//3, orig_size//3], dim=0)
      state_dict.update({
          f'{dest_name_prefix}.attention.self.query.weight': q_w,
          f'{dest_name_prefix}.attention.self.query.bias': q_b,

          f'{dest_name_prefix}.attention.self.key.weight': k_w,
          f'{dest_name_prefix}.attention.self.key.bias': k_b,

          f'{dest_name_prefix}.attention.self.value.weight': v_w,
          f'{dest_name_prefix}.attention.self.value.bias': v_b,
      })
      
      _load_weights(f'{src_name_prefix}.attn.proj.weight', f'{dest_name_prefix}.attention.output.dense.weight')
      _load_weights(f'{src_name_prefix}.attn.proj.bias', f'{dest_name_prefix}.attention.output.dense.bias')

      _load_weights(f'{src_name_prefix}.norm2.weight', f'{dest_name_prefix}.layernorm_after.weight')
      _load_weights(f'{src_name_prefix}.norm2.bias', f'{dest_name_prefix}.layernorm_after.bias')

      _load_weights(f'{src_name_prefix}.mlp.fc1.weight', f'{dest_name_prefix}.intermediate.dense.weight')
      _load_weights(f'{src_name_prefix}.mlp.fc1.bias', f'{dest_name_prefix}.intermediate.dense.bias')

      _load_weights(f'{src_name_prefix}.mlp.fc2.weight', f'{dest_name_prefix}.output.dense.weight')
      _load_weights(f'{src_name_prefix}.mlp.fc2.bias', f'{dest_name_prefix}.output.dense.bias')

    if downsample:
      _load_weights(f'backbone.bottom_up.layers.{layer_id}.downsample.reduction.weight', f'encoder.layers.{layer_id}.downsample.reduction.weight')
      _load_weights(f'backbone.bottom_up.layers.{layer_id}.downsample.norm.weight', f'encoder.layers.{layer_id}.downsample.norm.weight')
      _load_weights(f'backbone.bottom_up.layers.{layer_id}.downsample.norm.bias', f'encoder.layers.{layer_id}.downsample.norm.bias')

  _load_swin_layers(layer_id=0, num_blocks=2)
  _load_swin_layers(layer_id=1, num_blocks=2)
  _load_swin_layers(layer_id=2, num_blocks=18)
  _load_swin_layers(layer_id=3, num_blocks=2, downsample=False)

  # TODO(ishan): check absent layernorm's after layer-{0,1,2}
  _load_weights('backbone.bottom_up.norm3.weight', 'layernorm.weight')
  _load_weights('backbone.bottom_up.norm3.bias', 'layernorm.bias')
  
  for key in hf_model_state:
    if key in state_dict:
      assert hf_model_state[key].shape == state_dict[key].shape, f'Shape mismatch: {hf_model_state[key].shape}, {state_dict[key].shape}'
    else:
      print(f'{key} not set')

  hf_model.load_state_dict(state_dict)
  return hf_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--encoder_type", default='swin', type=str, required=False, help="Encoder type (resnet/swin).")
    parser.add_argument(
        "--save", default=False, type=bool, required=False, help="Whether to save the converted model or not."
    )
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="Path to the output model.")
    args = parser.parse_args()

    if args.encoder_type != 'swin':
        return f'{args.encoder_type} encoder type not supported.'
    
    hf_swin_model = convert_pretrained_HFSwinBackbone()
