from os import setegid
import numpy as np
import torch
import time
import sys
from PIL import Image
from diffusers import StableDiffusionPipeline
import math

seed = 666
seed_eval = 77
prompt = "a photo of an astronaut riding a horse on mars"
model_id = "./stable-diffusion-v1-4"
generator = torch.Generator("cpu").manual_seed(seed)
device = "cpu"
torch.backends.quantized.engine = 'onednn'

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

start1 = time.time()
pipe = StableDiffusionPipeline.from_pretrained(model_id)
print('weight load latency:', time.time() - start1)
pipe = pipe.to(device)

# print('=====1',pipe.feature_extractor)
# print('==ww===2',pipe.safety_checker)
# print('=====3',pipe.scheduler)
# print('==ww===4',pipe.text_encoder)
# print('=====5',pipe.tokenizer)
# print('==ww===6',pipe.unet)
# print('==ww===7', pipe.vae)

num_images = int(sys.argv[2])
_rows = int(math.sqrt(num_images))
prompt = [sys.argv[1]] * num_images
start1 = time.time()
fp32_images = pipe(prompt, guidance_scale=7.5, num_inference_steps=50, generator=generator).images
print('fp32 inference latency:', time.time() - start1)
grid = image_grid(fp32_images, rows=_rows, cols=num_images//_rows)
grid.save("astronaut_rides_horse-fp32.png")

# generate eval set
generator_eval = torch.Generator("cpu").manual_seed(seed_eval)
eval_images = pipe(prompt, guidance_scale=7.5, num_inference_steps=50, generator=generator_eval).images

# prompt = "a photo of an astronaut riding a horse on mars"
# start1 = time.time()
# image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator)["sample"][0]
# print('fp32 inference latency:', time.time() - start1)
# # pipe.safety_checker = lambda images, clip_input: (images, False)
# image.save("astronaut_rides_horse-fp32.png")

# with torch.autograd.profiler.profile() as prof:
#     pipe(prompt, guidance_scale=7.5)["sample"][0]
# prof.export_chrome_trace("fp32" + "_result.json")
# table_res = prof.key_averages().table(sort_by="self_cpu_time_total")
# print(table_res)

from neural_compressor.conf.config import QuantConf
from neural_compressor.experimental import Quantization, common
from neural_compressor.utils.pytorch import load

quant_config = QuantConf()
# Default approach is "post_training_static_quant"
quant_config.usr_cfg.model.framework = "pytorch_fx"
quant_config.usr_cfg.quantization.approach = "post_training_static_quant"

class CalibDataLoader(object):
    def __init__(self):
        self.batch_size = 2
        self.data = "a photo of an astronaut riding a horse on mars"

    def __iter__(self):
        yield self.data, None

def inc_quant(model, name, calibration_func, eval_func):
    # quantizer = Quantization(quant_config)
    quantizer = Quantization("./conf.yaml")
    quantizer.model = common.Model(model)
    # quantizer.calib_dataloader = CalibDataLoader()
    quantizer.q_func = calibration_func
    quantizer.eval_func = eval_func
    q_model = quantizer()
    q_model.save('./static-{}'.format(name))
    return q_model.model

def inc_load(model, name):
    int8_model = load('./static-{}'.format(name), model)
    return int8_model


def benchmark(pipe):
    import time
    warmup = 2
    total = 5
    total_time = 0.
    with torch.no_grad():
        for i in range(total):
            if i == warmup:
                start = time.time()
            # num_images = int(sys.argv[2])
            # prompt = [sys.argv[1]] * num_images
            prompt = "a photo of an astronaut riding a horse on mars"
            start2 = time.time()
            pipe(prompt, num_inference_steps=50, guidance_scale=7.5)["sample"][0]
            end2 = time.time()
            total_time += end2 - start2
            print('int8 inference latency: ', str(end2 - start2) + 's')
    print('avg_latency: ', (total_time) / (total - warmup), 's')


attr_list = ['unet']
for name in attr_list:
    model = getattr(pipe, name)
    def calibration_func(model):
        setattr(pipe, name, model)
        with torch.no_grad():
            num_images = int(sys.argv[2]) * 2
            prompt = [sys.argv[1]] * num_images
            # generator = torch.Generator("cpu").manual_seed(seed)
            new_images = pipe(prompt, guidance_scale=7.5, num_inference_steps=50, generator=generator).images
            # pipe.safety_checker = lambda images, clip_input: (images, False)

    def eval_func(model):
        setattr(pipe, name, model)
        with torch.no_grad():
            loss = torch.nn.MSELoss()
            num_images = int(sys.argv[2])
            prompt = [sys.argv[1]] * num_images
            generator_eval = torch.Generator("cpu").manual_seed(seed_eval)
            new_images = pipe(prompt, guidance_scale=7.5, num_inference_steps=50, generator=generator_eval).images
            mse_score = 0
            for i in range(num_images):
                new = torch.from_numpy(np.array(new_images[i]))
                old = torch.from_numpy(np.array(eval_images[i]))
                new = new.to(dtype=torch.float32)
                old = old.to(dtype=torch.float32)
                mse_score += loss(new, old)
            mse_score = mse_score.item()
            print("="*200)
            print(mse_score)
            return mse_score

    model = inc_quant(model, name, calibration_func, eval_func)
    # print(model)
    # model = inc_load(model, name)
    setattr(pipe, name, model)

'''
print('=====1',pipe.feature_extractor)
print('==1-embed, 145-linear===2',pipe.safety_checker)
print('=====3',pipe.scheduler)
print('==2-embed, 72-linear===4',pipe.text_encoder)
print('=====5',pipe.tokenizer)
print('==conv+184-linear===6',pipe.unet)
print('==conv 8-linear===7', pipe.vae)
'''


num_images = int(sys.argv[2])
prompt = [sys.argv[1]] * num_images
start1 = time.time()
generator = torch.Generator("cpu").manual_seed(seed)
int8_images = pipe(prompt, guidance_scale=7.5, num_inference_steps=50, generator=generator).images
print('int8 inference latency:', time.time() - start1)
pipe.safety_checker = lambda int8_images, clip_input: (images, False)
grid = image_grid(int8_images, rows=_rows, cols=num_images//_rows)
grid.save("astronaut_rides_horse-static-int8.png")

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt, guidance_scale=7.5, num_inference_steps=50, generator=generator)["sample"][0]
# # pipe.safety_checker = lambda images, clip_input: (images, False)
# image.save("astronaut_rides_horse-int8.png")

# with torch.autograd.profiler.profile() as prof:
#     pipe(prompt, guidance_scale=7.5)["sample"][0]
# prof.export_chrome_trace("int8" + "_result.json")
# table_res = prof.key_averages().table(sort_by="self_cpu_time_total")
# print(table_res)

benchmark(pipe)
