from diffusers import UNetUnconditionalModel, DDPMScheduler, DDPMPipeline
import argparse
import json
import torch


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return '.'.join(path.split('.')[n_shave_prefix_segments:])
    else:
        return '.'.join(path.split('.')[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    mapping = []
    for old_item in old_list:
        new_item = old_item
        new_item = new_item.replace('block.', 'resnets.')
        new_item = new_item.replace('conv_shorcut', 'conv1')
        new_item = new_item.replace('nin_shortcut', 'conv_shortcut')
        new_item = new_item.replace('temb_proj', 'time_emb_proj')

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({'old': old_item, 'new': new_item})

    return mapping


def renew_attention_paths(old_list, n_shave_prefix_segments=0, in_mid=False):
    mapping = []
    for old_item in old_list:
        new_item = old_item

        # In `model.mid`, the layer is called `attn`.
        if not in_mid:
            new_item = new_item.replace('attn', 'attentions')
        new_item = new_item.replace('.k.', '.key.')
        new_item = new_item.replace('.v.', '.value.')
        new_item = new_item.replace('.q.', '.query.')

        new_item = new_item.replace('proj_out', 'proj_attn')
        new_item = new_item.replace('norm', 'group_norm')

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)
        mapping.append({'old': old_item, 'new': new_item})

    return mapping


def assign_to_checkpoint(paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None):
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    if attention_paths_to_split is not None:
        if config is None:
            raise ValueError("Please specify the config if setting 'attention_paths_to_split' to 'True'.")

        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

            old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            checkpoint[path_map['query']] = query.reshape(target_shape).squeeze()
            checkpoint[path_map['key']] = key.reshape(target_shape).squeeze()
            checkpoint[path_map['value']] = value.reshape(target_shape).squeeze()

    for path in paths:
        new_path = path['new']

        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        new_path = new_path.replace('down.', 'downsample_blocks.')
        new_path = new_path.replace('up.', 'upsample_blocks.')

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement['old'], replacement['new'])

        if 'attentions' in new_path:
            checkpoint[new_path] = old_checkpoint[path['old']].squeeze()
        else:
            checkpoint[new_path] = old_checkpoint[path['old']]


def convert_ddpm_checkpoint(checkpoint, config):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """
    new_checkpoint = {}

    new_checkpoint['time_embedding.linear_1.weight'] = checkpoint['temb.dense.0.weight']
    new_checkpoint['time_embedding.linear_1.bias'] = checkpoint['temb.dense.0.bias']
    new_checkpoint['time_embedding.linear_2.weight'] = checkpoint['temb.dense.1.weight']
    new_checkpoint['time_embedding.linear_2.bias'] = checkpoint['temb.dense.1.bias']

    new_checkpoint['conv_norm_out.weight'] = checkpoint['norm_out.weight']
    new_checkpoint['conv_norm_out.bias'] = checkpoint['norm_out.bias']

    new_checkpoint['conv_in.weight'] = checkpoint['conv_in.weight']
    new_checkpoint['conv_in.bias'] = checkpoint['conv_in.bias']
    new_checkpoint['conv_out.weight'] = checkpoint['conv_out.weight']
    new_checkpoint['conv_out.bias'] = checkpoint['conv_out.bias']

    num_downsample_blocks = len({'.'.join(layer.split('.')[:2]) for layer in checkpoint if 'down' in layer})
    downsample_blocks = {layer_id: [key for key in checkpoint if f'down.{layer_id}' in key] for layer_id in range(num_downsample_blocks)}

    num_upsample_blocks = len({'.'.join(layer.split('.')[:2]) for layer in checkpoint if 'up' in layer})
    upsample_blocks = {layer_id: [key for key in checkpoint if f'up.{layer_id}' in key] for layer_id in range(num_upsample_blocks)}

    for i in range(num_downsample_blocks):
        block_id = (i - 1) // (config['num_res_blocks'] + 1)

        if any('downsample' in layer for layer in downsample_blocks[i]):
            new_checkpoint[f'downsample_blocks.{i}.downsamplers.0.conv.weight'] = checkpoint[f'down.{i}.downsample.conv.weight']
            new_checkpoint[f'downsample_blocks.{i}.downsamplers.0.conv.bias'] = checkpoint[f'down.{i}.downsample.conv.bias']
            new_checkpoint[f'downsample_blocks.{i}.downsamplers.0.op.weight'] = checkpoint[f'down.{i}.downsample.conv.weight']
            new_checkpoint[f'downsample_blocks.{i}.downsamplers.0.op.bias'] = checkpoint[f'down.{i}.downsample.conv.bias']

        if any('block' in layer for layer in downsample_blocks[i]):
            num_blocks = len({'.'.join(shave_segments(layer, 2).split('.')[:2]) for layer in downsample_blocks[i] if 'block' in layer})
            blocks = {layer_id: [key for key in downsample_blocks[i] if f'block.{layer_id}' in key] for layer_id in range(num_blocks)}

            if num_blocks > 0:
                for j in range(config['num_res_blocks']):
                    paths = renew_resnet_paths(blocks[j])
                    assign_to_checkpoint(paths, new_checkpoint, checkpoint)

        if any('attn' in layer for layer in downsample_blocks[i]):
            num_attn = len({'.'.join(shave_segments(layer, 2).split('.')[:2]) for layer in downsample_blocks[i] if 'attn' in layer})
            attns = {layer_id: [key for key in downsample_blocks[i] if f'attn.{layer_id}' in key] for layer_id in range(num_blocks)}

            if num_attn > 0:
                for j in range(config['num_res_blocks']):
                    paths = renew_attention_paths(attns[j])
                    assign_to_checkpoint(paths, new_checkpoint, checkpoint, config=config)

    mid_block_1_layers = [key for key in checkpoint if "mid.block_1" in key]
    mid_block_2_layers = [key for key in checkpoint if "mid.block_2" in key]
    mid_attn_1_layers = [key for key in checkpoint if "mid.attn_1" in key]

    # Mid new 2
    paths = renew_resnet_paths(mid_block_1_layers)
    assign_to_checkpoint(paths, new_checkpoint, checkpoint, additional_replacements=[
        {'old': 'mid.', 'new': 'mid_new_2.'}, {'old': 'block_1', 'new': 'resnets.0'}
    ])

    paths = renew_resnet_paths(mid_block_2_layers)
    assign_to_checkpoint(paths, new_checkpoint, checkpoint, additional_replacements=[
        {'old': 'mid.', 'new': 'mid_new_2.'}, {'old': 'block_2', 'new': 'resnets.1'}
    ])

    paths = renew_attention_paths(mid_attn_1_layers, in_mid=True)
    assign_to_checkpoint(paths, new_checkpoint, checkpoint, additional_replacements=[
        {'old': 'mid.', 'new': 'mid_new_2.'}, {'old': 'attn_1', 'new': 'attentions.0'}
    ])

    for i in range(num_upsample_blocks):
        block_id = num_upsample_blocks - 1 - i

        if any('upsample' in layer for layer in upsample_blocks[i]):
            new_checkpoint[f'upsample_blocks.{block_id}.upsamplers.0.conv.weight'] = checkpoint[f'up.{i}.upsample.conv.weight']
            new_checkpoint[f'upsample_blocks.{block_id}.upsamplers.0.conv.bias'] = checkpoint[f'up.{i}.upsample.conv.bias']

        if any('block' in layer for layer in upsample_blocks[i]):
            num_blocks = len({'.'.join(shave_segments(layer, 2).split('.')[:2]) for layer in upsample_blocks[i] if 'block' in layer})
            blocks = {layer_id: [key for key in upsample_blocks[i] if f'block.{layer_id}' in key] for layer_id in range(num_blocks)}

            if num_blocks > 0:
                for j in range(config['num_res_blocks'] + 1):
                    replace_indices = {'old': f'upsample_blocks.{i}', 'new': f'upsample_blocks.{block_id}'}
                    paths = renew_resnet_paths(blocks[j])
                    assign_to_checkpoint(paths, new_checkpoint, checkpoint, additional_replacements=[replace_indices])

        if any('attn' in layer for layer in upsample_blocks[i]):
            num_attn = len({'.'.join(shave_segments(layer, 2).split('.')[:2]) for layer in upsample_blocks[i] if 'attn' in layer})
            attns = {layer_id: [key for key in upsample_blocks[i] if f'attn.{layer_id}' in key] for layer_id in range(num_blocks)}

            if num_attn > 0:
                for j in range(config['num_res_blocks'] + 1):
                    replace_indices = {'old': f'upsample_blocks.{i}', 'new': f'upsample_blocks.{block_id}'}
                    paths = renew_attention_paths(attns[j])
                    assign_to_checkpoint(paths, new_checkpoint, checkpoint, additional_replacements=[replace_indices])

    new_checkpoint = {k.replace('mid_new_2', 'mid'): v for k, v in new_checkpoint.items()}
    return new_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )

    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the architecture.",
    )

    parser.add_argument(
        "--dump_path", default=None, type=str, required=True, help="Path to the output model."
    )

    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint_path)

    with open(args.config_file) as f:
        config = json.loads(f.read())

    converted_checkpoint = convert_ddpm_checkpoint(checkpoint, config)

    if "ddpm" in config:
        del config["ddpm"]

    model = UNetUnconditionalModel(**config)
    model.load_state_dict(converted_checkpoint)

    scheduler = DDPMScheduler.from_config("/".join(args.checkpoint_path.split("/")[:-1]))

    pipe = DDPMPipeline(unet=model, scheduler=scheduler)
    pipe.save_pretrained(args.dump_path)
