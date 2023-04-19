import os
import glob
import re
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from typing import *
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionAdapterPipeline, T2IAdapter, MultiAdapter



def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResnetBlock_light(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.block1 = nn.Conv2d(in_c, in_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(in_c, in_c, 3, 1, 1)

    def forward(self, x):
        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)

        return h + x


class extractor(nn.Module):
    def __init__(self, in_c, inter_c, out_c, nums_rb, down=False):
        super().__init__()
        self.in_conv = nn.Conv2d(in_c, inter_c, 1, 1, 0)
        self.body = []
        for _ in range(nums_rb):
            self.body.append(ResnetBlock_light(inter_c))
        self.body = nn.Sequential(*self.body)
        self.out_conv = nn.Conv2d(inter_c, out_c, 1, 1, 0)
        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=False)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        x = self.in_conv(x)
        x = self.body(x)
        x = self.out_conv(x)

        return x


class Adapter_light(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=4, cin=64*3):
        super(Adapter_light, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(8)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            if i == 0:
                self.body.append(extractor(in_c=cin, inter_c=channels[i]//4, out_c=channels[i], nums_rb=nums_rb, down=False))
            else:
                self.body.append(extractor(in_c=channels[i-1], inter_c=channels[i]//4, out_c=channels[i], nums_rb=nums_rb, down=True))
        self.body = nn.ModuleList(self.body)

    def forward(self, x):
        # unshuffle
        x = self.unshuffle(x)
        # extract features
        features = []
        for i in range(len(self.channels)):
            x = self.body[i](x)
            features.append(x)

        return features


def compare_light_model():

    def convert(old):
        mapping = {
            "body.0.in_conv.bias": "conv_in.bias",
            "body.0.in_conv.weight": "conv_in.weight",
            "body.0.out_conv.bias": "body.3.out_conv.bias",
            "body.0.out_conv.weight": "body.3.out_conv.weight",
            "body.1.in_conv.bias": "body.4.in_conv.bias",
            "body.1.in_conv.weight": "body.4.in_conv.weight",
            "body.1.out_conv.bias": "body.7.out_conv.bias",
            "body.1.out_conv.weight": "body.7.out_conv.weight",
            "body.2.in_conv.bias": "body.8.in_conv.bias",
            "body.2.in_conv.weight": "body.8.in_conv.weight",
            "body.2.out_conv.bias": "body.11.out_conv.bias",
            "body.2.out_conv.weight": "body.11.out_conv.weight",
            "body.3.in_conv.bias": "body.12.in_conv.bias",
            "body.3.in_conv.weight": "body.12.in_conv.weight",
            "body.3.out_conv.bias": "body.15.out_conv.bias",
            "body.3.out_conv.weight": "body.15.out_conv.weight",
        }
        cvr_state = {}
        resblock = re.compile(r"body\.(\d+)\.body\.(\d+)\.(.+)")
        for k, v in old.items():
            m = resblock.match(k)
            if m:
                new_group = int(m.group(1)) * 4 + int(m.group(2))
                cvr_state[f"body.{new_group}.{m.group(3)}"] = v
            else:
                cvr_state[mapping[k]] = v
        return cvr_state


    X = torch.zeros([1, 3, 512, 512], dtype=torch.float32)
    ad = Adapter_light()
    print(ad)
    trainable_params = sum(
        p.numel() for p in ad.parameters() if p.requires_grad
    )
    print('trainable_params: ', trainable_params)
    y = ad(X)
    print('feat: ', [yy.shape for yy in y])

    old_state = ad.state_dict()
    for k in sorted(old_state.keys()):
        print(k)
    
    print('-' * 100)

    ad = T2IAdapter(
        block_out_channels=[320, 640, 1280, 1280],
        block_mid_channels=[80, 160, 320, 320],
        channels_in=int(3 * 64), 
        num_res_blocks=4, 
        kernel_size=3, 
        proj_kernel_size=1,
        res_block_skip=True, 
        use_conv=False
    )
    trainable_params = sum(
        p.numel() for p in ad.parameters() if p.requires_grad
    )
    
    # new_state = ad.state_dict()
    # for k in sorted(new_state.keys()):
    #     print(k)

    print('trainable_params: ', trainable_params)
    y = ad(X)
    print('feat: ', [yy.shape for yy in y.values()])

    # ad = Adapter(
    #     block_out_channels=[320, 640, 1280, 1280][:4],
    #     channels_in=int(3 * 64), 
    #     num_res_blocks=2, 
    #     kernel_size=1, 
    #     res_block_skip=True, 
    #     use_conv=False
    # )
    # ad = Adapter.from_pretrained("RzZ/sd-v1-4-adapter-keypose")
    print(ad)
    ad.load_state_dict(convert(torch.load('/home/ron/Downloads/Adapter/t2iadapter_color_sd14v1.pth')))
    # torch.save(ad.state_dict(), '/home/ron/Downloads/Adapter/diffuers_adapter_color.pth')
    ad.save_pretrained("/home/ron/Projects/tmp/sd-v1-4-adapter-color")


def test_adapter():
    adapter_ckpt = "/home/ron/Downloads/t2iadapter_seg_sd14v1.pth"
    adapter = T2IAdapter(
        block_out_channels=[320, 640, 1280, 1280][:4],
        channels_in=3, 
        num_res_blocks=2, 
        kernel_size=1, 
        res_block_skip=True, 
        use_conv=False
    )
    weight = torch.load(adapter_ckpt)
    mapping = {}
    for k in weight.keys():
        print(k)
        if 'down_opt.op' in k:
            mapping[k] = k.replace('down_opt.op', 'down_opt.conv')
    print('mapping: ', mapping)
    for old, new in mapping.items():
        weight[new] = weight.pop(old)

    adapter.load_state_dict(weight)


def test_pipeline(device='cpu'):

    def inputs(revision):
        if revision == 'seg':
            mask = Image.open("motor.png")
            prompt = [
                "A black Honda motorcycle parked in front of a garage",
                # "A red-blue Honda motorcycle parked in front of a garage",
                # "A green Honda motorcycle parked in a desert",
            ]
        elif revision == 'keypose':
            mask = Image.open("/home/ron/Downloads/iron.png")
            prompt = [
                'a man waling on the street',
                # 'a bear waling on the street',
                # 'a astronaut waling on the street',
            ]
        elif revision == 'openpose':
            mask = Image.open("/home/ron/Downloads/openpose.png")
            prompt = [
                'iron man standing on the mars',
            ]
        elif revision == 'depth':
            mask = Image.open("/home/ron/Downloads/desk_depth_512.png")
            prompt = [
                'An office room with nice view',
            ]
        elif revision == 'canny':
            mask = Image.open("/home/ron/Downloads/vermeer_canny_edged.png")
            mask = torch.tensor(np.array(mask)).float() / 255
            mask = mask[..., :3].mean(dim=-1, keepdim=True)
            mask = mask.permute(2, 0, 1)
            prompt = [
                'disco dancer with colorful lights',
            ]
        elif revision == 'sketch':
            mask = Image.open("/home/ron/Downloads/sketch_car.png").convert('L')
            # mask = torch.tensor(np.array(mask)).float() / 255
            # mask = mask[..., :3].max(dim=-1, keepdim=True).values
            # mask = mask.permute(2, 0, 1)
            prompt = [
                'blue limousine with wings, high quality, photorealistic',
                'red limousine with wings, high quality, photorealistic',
            ]
        elif revision == 'color':
            # mask = Image.open("/home/ron/Downloads/color_0002.png")
            # mask = Image.open("/home/ron/Pictures/artem-chebokha-204-4-3-surf-1538.jpeg")
            mask = Image.open("color_ref.png")
            mask = mask.resize((8, 8)).resize((512, 512), resample=Image.Resampling.NEAREST)
            mask.save('color_palette.png')
            prompt = [
                'At night, glowing cubes in front of the beach'
                # 'A photo of scenery',
            ]
        elif revision == 'keypose_depth':
            mask1 = Image.open("/home/ron/Downloads/iron.png")
            mask2 = Image.open("/home/ron/Downloads/desk_depth_512.png")

            mask1 = mask1.resize((384, 512))

            # mask1 = torch.from_numpy(np.array(mask1))
            # mask2 = torch.from_numpy(np.array(mask2))
            
            # mask = torch.cat([mask1, mask2], dim=-1)
            # mask = mask.permute(2, 0, 1) / 255
            # print(mask.shape, '#############')

            mask = [mask1, mask2]
            
            prompt = [
                'a man waling in an office room with nice view',
                'a man waling in an office room with nice view',
                'a man waling in an office room with nice view',
            ]
        return mask, prompt

    # model_name = "CompVis/stable-diffusion-v1-4"
    model_name = "CompVis/stable-diffusion-v1-4"
    revision = "keypose"
    mask, prompt = inputs(revision)
    # generator = torch.Generator(device=device).manual_seed(1)
    # generator = None
    num_images_per_prompt = 1

    if device =='cuda':
        if '_' in revision:
            # c = MultiAdapter.from_pretrained(f"RzZ/sd-v1-4-adapter-{revision.replace('_', '-')}")
            c = [
                    T2IAdapter.from_pretrained("RzZ/sd-v1-4-adapter-keypose"),
                    T2IAdapter.from_pretrained("RzZ/sd-v1-4-adapter-depth"),
            ]
        else:
            c = T2IAdapter.from_pretrained(f"RzZ/sd-v1-4-adapter-{revision}")
            c = c.to(torch.float16)
        
        pipe = StableDiffusionAdapterPipeline.from_pretrained(
            model_name, revision='main', torch_dtype=torch.float16, safety_checker=None,
            adapter=c,
            adapter_weights=[0.8, 0.8],
        )
        pipe.to("cuda")
    else:
        c = [
            T2IAdapter.from_pretrained("RzZ/sd-v1-4-adapter-keypose"),
            T2IAdapter.from_pretrained("RzZ/sd-v1-4-adapter-depth"),
        ]
        pipe = StableDiffusionAdapterPipeline.from_pretrained(
            model_name, adapter=c, torch_dtype=torch.float32, safety_checker=None,
            adapter_weights=[0.8, 0.8],
        )

    
    for k in range(0, 1):
        generator = torch.Generator(device=device).manual_seed(k)
        images = pipe(
            prompt, 
            [mask] * len(prompt), 
            output_type='pil',
            generator=generator,
            num_inference_steps=50,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        # np.save('sample_output.npy', images)
        images[0].save('sample_output.png')
        grid = int(len(images)**0.5) + 1

        try:
            plt.subplot(grid, grid, 1)
            plt.imshow(mask)
        except TypeError:
            pass

        for i, image in enumerate(images):
            plt.subplot(grid, grid, 2 + i)
            plt.imshow(image)
            plt.title(prompt[i // num_images_per_prompt], fontsize=24)
        plt.show()


def fix_ckpt_name(hub_repo_dir):
    import json
    
    print('-' * 100)
    print('Open: ', hub_repo_dir)
    bin_path = os.path.join(hub_repo_dir, "diffusion_pytorch_model.bin")
    state = torch.load(bin_path)
    mapping = {
        'down_opt': 'down_opt'
    }
    wrong_keys = [k for k in state.keys() if 'skep' in k]
    print(state.keys())
    print("wrong_keys: ", wrong_keys)
    state = {
        k.replace('skep', 'skip'): v
        for k, v in state.items()
    }
    # cfg_path = os.path.join(hub_repo_dir, "config.json")
    # with open(cfg_path, mode='r') as f:
    #     config = json.load(f)
    # config['_class_name'] = 


if __name__ == "__main__":
    # test_adapter()
    # test_pipeline(device='cuda')
    # compare_light_model()
    for folder in glob.glob("/home/ron/Projects/tmp/sd-v1-4-adapter-*"):
        fix_ckpt_name(folder)